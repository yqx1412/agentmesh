"""``agentmesh up``: start several agents, each as its own ``agentmesh serve`` process.

Each process gets its own spec (model, tools, workspace, port) and its own log file. The
launcher waits until every agent serves its Agent Card, prints one line per agent, and
stops them all on Ctrl+C or as soon as one of them exits.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from agentmesh.spec import AgentSpec, SpecError, load_spec

# `agentmesh up` with no arguments: the four M2 specialists.
DEFAULT_FLEET = ("research", "analyst", "verifier", "knowledge")
CARD_PATH = "/.well-known/agent-card.json"


@dataclass
class Member:
    path: Path
    spec: AgentSpec
    log: Path
    proc: subprocess.Popen | None = None


def default_specs(agents_dir: Path) -> list[Path]:
    return [agents_dir / f"{name}.toml" for name in DEFAULT_FLEET]


def plan(paths: list[Path], log_dir: Path) -> list[Member]:
    """Load and check every spec before starting anything: valid, distinct ports, ports free."""
    members = [Member(p, load_spec(p), log_dir / f"{p.stem}.log") for p in paths]
    seen: dict[int, str] = {}
    for m in members:
        if m.spec.port in seen:
            raise SpecError(f"{m.path.name} and {seen[m.spec.port]} both use port {m.spec.port}")
        seen[m.spec.port] = m.path.name
        if not port_free(m.spec.host, m.spec.port):
            raise SpecError(f"{m.path.name}: {m.spec.host}:{m.spec.port} is already in use")
    return members


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def start(members: list[Member], trace_dir: Path | None = None) -> None:
    for m in members:
        m.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "agentmesh", "serve", str(m.path)]
        if trace_dir is not None:
            cmd += ["--trace-dir", str(trace_dir / m.path.stem)]
        log = m.log.open("w", encoding="utf-8")
        # Unbuffered, so the log shows the startup line and errors as they happen.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        m.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()  # the child holds its own handle


def wait_ready(members: list[Member], timeout: float = 60.0) -> dict[str, dict]:
    """Poll each agent's card until all answer. Returns card JSON by spec stem."""
    cards: dict[str, dict] = {}
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2.0) as http:
        while len(cards) < len(members):
            for m in members:
                if m.path.stem in cards:
                    continue
                assert m.proc is not None
                if m.proc.poll() is not None:
                    raise RuntimeError(
                        f"{m.spec.name} exited with code {m.proc.returncode}; see {m.log}"
                    )
                try:
                    r = http.get(m.spec.url + CARD_PATH)
                    if r.status_code == 200:
                        cards[m.path.stem] = r.json()
                except httpx.HTTPError:
                    pass
            if len(cards) < len(members):
                if time.monotonic() > deadline:
                    missing = [m.spec.name for m in members if m.path.stem not in cards]
                    raise TimeoutError(f"no Agent Card after {timeout:g} s from: {missing}")
                time.sleep(0.3)
    return cards


def stop(members: list[Member], grace: float = 10.0) -> None:
    for m in members:
        if m.proc is not None and m.proc.poll() is None:
            m.proc.terminate()
    deadline = time.monotonic() + grace
    for m in members:
        if m.proc is None:
            continue
        try:
            m.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            m.proc.kill()
            m.proc.wait()


def describe(m: Member, card: dict) -> str:
    skills = ", ".join(s["id"] for s in card.get("skills", []))
    tools = ", ".join(m.spec.builtin_tools) or "-"
    return (
        f"{card['name']:<16} {m.spec.url:<23} pid {m.proc.pid if m.proc else '?':<6} "
        f"{m.spec.model:<10} built-in: {tools} | skills: {skills}"
    )


def watch(members: list[Member]) -> int:
    """Block until Ctrl+C (exit 0) or until any agent exits (exit 1); then stop the rest."""
    try:
        while True:
            for m in members:
                assert m.proc is not None
                if m.proc.poll() is not None:
                    print(
                        f"{m.spec.name} exited with code {m.proc.returncode}; stopping the "
                        f"others (log: {m.log})",
                        file=sys.stderr,
                    )
                    return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("stopping...", file=sys.stderr)
        return 0
    finally:
        stop(members)
