"""``agentmesh watch``: a live, read-only page of what every agent is doing.

It reads the files the agents already write under one trace root (``agentmesh up
--trace-dir runs/fleet-traces`` gives one folder per agent):

- ``<agent>/a2a.jsonl``: the A2A lifecycle (submitted, working, completed, failed, ...).
- ``<agent>/<timestamp>-<task id>.jsonl``: the AgentOS trace (model steps, tool calls).

So it sees every task, whichever client sent it, without being an A2A client itself. It also
polls each agent's card to show whether the agent is up.

The page has no login. It binds to 127.0.0.1 only and refuses requests whose ``Host`` isn't
a loopback name, so another web page can't read it through DNS rebinding. Everything it
shows (fetched pages, tool output) is inserted as text, never as HTML.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from agentmesh.journal import JOURNAL_NAME

PAGE = Path(__file__).with_name("watch.html")
CARD_PATH = "/.well-known/agent-card.json"
TERMINAL = {"completed", "failed", "rejected", "canceled"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
# The folder `agentmesh orchestrate --trace-root` writes under the same root.
ORCHESTRATOR = "orchestrator"

# How much of a tool output, model reply or answer to send to the page.
MAX_TEXT = 4000
# Newest tasks kept per agent.
MAX_TASKS = 25

_TRACE_NAME = re.compile(r"^\d{8}-\d{6}-(?P<id>.+)\.jsonl$")


def _clip(value: Any, limit: int = MAX_TEXT) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


@dataclass
class _Cached:
    mtime_ns: int
    size: int
    records: list[dict[str, Any]]


@dataclass
class TraceReader:
    """Parses JSONL files, re-reading one only when its size or mtime changes."""

    _cache: dict[Path, _Cached] = field(default_factory=dict)

    def records(self, path: Path) -> list[dict[str, Any]]:
        try:
            st = path.stat()
        except OSError:
            return []
        hit = self._cache.get(path)
        if hit and (hit.mtime_ns, hit.size) == (st.st_mtime_ns, st.st_size):
            return hit.records
        out: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a line still being written
                    if isinstance(rec, dict):
                        out.append(rec)
        except OSError:
            return []
        self._cache[path] = _Cached(st.st_mtime_ns, st.st_size, out)
        return out


def trace_steps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """AgentOS trace records, reduced to what the page shows."""
    steps = []
    for r in records:
        ev = r.get("event")
        if ev == "run_start":
            steps.append(
                {
                    "ts": r.get("ts"),
                    "kind": "start",
                    "model": r.get("model"),
                    "tools": r.get("tools") or [],
                }
            )
        elif ev == "llm_response":
            calls = [
                {"name": c.get("name", "?"), "arguments": _clip(c.get("arguments", {}))}
                for c in r.get("tool_calls") or []
            ]
            steps.append(
                {
                    "ts": r.get("ts"),
                    "kind": "model",
                    "step": r.get("step"),
                    "content": _clip(r.get("content") or ""),
                    "tool_calls": calls,
                    "prompt_tokens": r.get("prompt_tokens"),
                    "completion_tokens": r.get("completion_tokens"),
                }
            )
        elif ev == "tool_result":
            call = r.get("call") or {}
            steps.append(
                {
                    "ts": r.get("ts"),
                    "kind": "tool",
                    "step": r.get("step"),
                    "name": call.get("name", "?"),
                    "ok": bool(r.get("ok")),
                    "output": _clip(r.get("output") or ""),
                }
            )
        elif ev == "run_end":
            steps.append(
                {
                    "ts": r.get("ts"),
                    "kind": "end",
                    "stop_reason": r.get("stop_reason"),
                    "answer": _clip(r.get("answer") or ""),
                }
            )
    return steps


def _state_from_trace(steps: list[dict[str, Any]]) -> str:
    """For traces with no journal (older runs): the state the trace implies."""
    end = next((s for s in reversed(steps) if s["kind"] == "end"), None)
    if end is None:
        return "working"
    return "completed" if end.get("stop_reason") == "final_answer" else "failed"


def collect_agent(folder: Path, reader: TraceReader) -> list[dict[str, Any]]:
    """Every task one agent's folder knows about, newest first."""
    tasks: dict[str, dict[str, Any]] = {}

    for rec in reader.records(folder / JOURNAL_NAME):
        tid = rec.get("task")
        if not tid:
            continue
        t = tasks.setdefault(
            tid,
            {
                "id": tid,
                "context": rec.get("context"),
                "events": [],
                "steps": [],
                "text": "",
                "journal": True,
            },
        )
        state = rec.get("state", "?")
        t["events"].append(
            {
                "ts": rec.get("ts"),
                "state": state,
                "message": _clip(rec.get("message") or ""),
            }
        )
        if state == "submitted":
            t["text"] = _clip(rec.get("text") or "")
            for key in ("orchestration", "step"):
                if rec.get(key):
                    t[key] = str(rec[key])
        if state == "completed":
            t["answer"] = _clip(rec.get("answer") or "")
        if rec.get("stats"):
            t["stats"] = rec["stats"]

    for path in folder.glob("*.jsonl"):
        m = _TRACE_NAME.match(path.name)
        if path.name == JOURNAL_NAME or m is None:
            continue
        steps = trace_steps(reader.records(path))
        tid = m.group("id")
        t = tasks.get(tid)
        if t is None:  # a run from before the journal existed, or outside A2A
            start = next((s for s in steps if s["kind"] == "start"), None)
            raw = reader.records(path)
            text = next((r.get("task", "") for r in raw if r.get("event") == "run_start"), "")
            t = tasks[tid] = {
                "id": tid,
                "context": None,
                "journal": False,
                "text": _clip(text),
                "events": [
                    {
                        "ts": (start or {}).get("ts") or path.stat().st_mtime,
                        "state": "working",
                        "message": "",
                    }
                ],
            }
            end = next((s for s in steps if s["kind"] == "end"), None)
            if end is not None:
                t["events"].append(
                    {
                        "ts": end["ts"],
                        "state": _state_from_trace(steps),
                        "message": end.get("stop_reason") or "",
                    }
                )
                if end.get("stop_reason") == "final_answer":
                    t["answer"] = end["answer"]
        t["steps"] = steps
        t["trace"] = path.name

    for t in tasks.values():
        t["events"].sort(key=lambda e: e["ts"] or 0)
        t["state"] = t["events"][-1]["state"] if t["events"] else "?"
        t["started"] = t["events"][0]["ts"] if t["events"] else None
        ends = [e["ts"] for e in t["events"] if e["state"] in TERMINAL]
        t["ended"] = ends[-1] if ends else None
        t["tool_calls"] = sum(1 for s in t.get("steps", []) if s["kind"] == "tool")
    ordered = sorted(tasks.values(), key=lambda t: t["started"] or 0, reverse=True)
    return ordered[:MAX_TASKS]


@dataclass
class AgentInfo:
    key: str  # folder name under the trace root, e.g. "analyst"
    name: str
    url: str | None = None
    model: str | None = None
    up: bool | None = None  # None: not probed / no URL


class Watcher:
    """Builds the page's state and keeps each agent's up/down reading fresh."""

    def __init__(self, root: Path, agents: list[AgentInfo], probe_every: float = 3.0) -> None:
        self.root = root
        self.agents = {a.key: a for a in agents}
        self.reader = TraceReader()
        self.probe_every = probe_every
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def state(self) -> dict[str, Any]:
        with self._lock:
            known = dict(self.agents)
            if self.root.is_dir():
                for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
                    known.setdefault(d.name, AgentInfo(key=d.name, name=d.name))
            if ORCHESTRATOR in known:  # the lane that starts the others goes first
                known = {ORCHESTRATOR: known.pop(ORCHESTRATOR), **known}
            out = []
            for key, info in known.items():
                tasks = collect_agent(self.root / key, self.reader)
                name = info.name
                if info.name == key:  # no spec: take the name the journal recorded
                    name = next(
                        (
                            r.get("agent")
                            for r in self.reader.records(self.root / key / JOURNAL_NAME)
                            if r.get("agent")
                        ),
                        key,
                    )
                out.append(
                    {
                        "key": key,
                        "name": name,
                        "url": info.url,
                        "model": info.model,
                        "up": info.up,
                        "busy": any(t["state"] not in TERMINAL for t in tasks),
                        "tasks": tasks,
                    }
                )
        return {"now": time.time(), "root": str(self.root), "agents": out}

    def probe_once(self, http: httpx.Client) -> None:
        for info in list(self.agents.values()):
            if not info.url:
                continue
            try:
                info.up = http.get(info.url + CARD_PATH).status_code == 200
            except httpx.HTTPError:
                info.up = False

    def run_probes(self) -> None:
        with httpx.Client(timeout=1.5) as http:
            while not self._stop.is_set():
                self.probe_once(http)
                self._stop.wait(self.probe_every)

    def stop(self) -> None:
        self._stop.set()


def _host_ok(host_header: str | None) -> bool:
    """True for ``127.0.0.1``, ``localhost`` or ``[::1]``, with or without a port."""
    if not host_header:
        return False
    h = host_header.strip().lower()
    host = h[: h.find("]") + 1] if h.startswith("[") else h.split(":", 1)[0]
    return host in LOOPBACK_HOSTS


def make_handler(watcher: Watcher) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'self'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if not _host_ok(self.headers.get("Host")):
                self._send(403, b"forbidden host\n", "text/plain; charset=utf-8")
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                body = json.dumps(watcher.state(), ensure_ascii=False).encode()
                self._send(200, body, "application/json; charset=utf-8")
            else:
                self._send(404, b"not found\n", "text/plain; charset=utf-8")

    return Handler


def serve(watcher: Watcher, port: int) -> ThreadingHTTPServer:
    """Bind on 127.0.0.1 only. The caller runs ``serve_forever``."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(watcher))
    server.daemon_threads = True
    return server
