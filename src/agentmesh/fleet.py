"""``agentmesh up``: start several agents, each as its own ``agentmesh serve`` process.

Each process gets its own spec (model, tools, workspace, port) and its own log file. The
launcher waits until every agent serves its Agent Card, prints one line per agent, then
supervises them: an agent that exits is restarted (with backoff, up to a limit) while the
others keep running, so one dead agent never takes the fleet down. Ctrl+C stops them all.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
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
    cmd: list[str] | None = None  # set by start(); tests may set it directly
    restarts: int = 0
    down: bool = False  # exited and out of restarts


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


def start(
    members: list[Member], trace_dir: Path | None = None, extra: list[str] | None = None
) -> None:
    """``extra`` goes on every ``serve`` command line (the M5 auth options)."""
    for m in members:
        if m.cmd is None:
            m.cmd = [sys.executable, "-m", "agentmesh", "serve", str(m.path)]
            if trace_dir is not None:
                m.cmd += ["--trace-dir", str(trace_dir / m.path.stem)]
            m.cmd += extra or []
        launch(m)


def launch(m: Member, append: bool = False) -> None:
    """Start (or restart) one member's process. A restart appends to the same log."""
    assert m.cmd is not None
    m.log.parent.mkdir(parents=True, exist_ok=True)
    log = m.log.open("a" if append else "w", encoding="utf-8")
    if append:
        log.write(f"\n--- restart {m.restarts} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log.flush()
    # Unbuffered, so the log shows the startup line and errors as they happen.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    m.proc = subprocess.Popen(m.cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
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


def watch(
    members: list[Member],
    max_restarts: int = 3,
    backoff: float = 2.0,
    poll: float = 0.5,
    stop_event: threading.Event | None = None,
) -> int:
    """Supervise the fleet until Ctrl+C or ``stop_event`` (exit 0), then stop every agent.

    An agent that exits is restarted after ``backoff * 2**n`` seconds, at most
    ``max_restarts`` times; the other agents keep running throughout. An agent out of
    restarts stays down. Returns 1 only when every agent is down.
    """
    due: dict[int, float] = {}  # member index -> monotonic time its restart is due
    try:
        while stop_event is None or not stop_event.is_set():
            now = time.monotonic()
            for i, m in enumerate(members):
                assert m.proc is not None
                if m.down or m.proc.poll() is None:
                    continue
                if i in due:
                    if now >= due[i]:
                        del due[i]
                        m.restarts += 1
                        launch(m, append=True)
                        print(
                            f"{m.spec.name} ({m.spec.url}) restarted "
                            f"({m.restarts}/{max_restarts}), pid {m.proc.pid}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue
                code = m.proc.returncode
                if m.restarts >= max_restarts:
                    m.down = True
                    print(
                        f"{m.spec.name} ({m.spec.url}) exited with code {code} and is out of "
                        f"restarts; it stays down, the others keep running (log: {m.log})",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                delay = backoff * 2**m.restarts
                due[i] = now + delay
                print(
                    f"{m.spec.name} ({m.spec.url}) exited with code {code}; restarting in "
                    f"{delay:g} s, the others keep running (log: {m.log})",
                    file=sys.stderr,
                    flush=True,
                )
            if all(m.down for m in members):
                print("every agent is down; exiting", file=sys.stderr)
                return 1
            time.sleep(poll)
        return 0
    except KeyboardInterrupt:
        print("stopping...", file=sys.stderr)
        return 0
    finally:
        stop(members)
