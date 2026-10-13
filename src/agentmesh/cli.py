"""``agentmesh serve <spec> | up | watch | card <url> | ask <url> <task>``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from google.protobuf.json_format import MessageToDict

from agentmesh import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentmesh", description="Agents over A2A.")
    p.add_argument("--version", action="version", version=f"agentmesh {__version__}")
    sub = p.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run one agent as an A2A server")
    serve.add_argument("spec", type=Path, help="Agent spec (TOML)")
    serve.add_argument("--port", type=int, help="Override the spec's port")
    serve.add_argument("--trace-dir", type=Path, help="Write an AgentOS trace per task here")

    up = sub.add_parser("up", help="Start several agents, one process each (default: M2 four)")
    up.add_argument("specs", type=Path, nargs="*", help="Agent specs (default: the specialists)")
    up.add_argument("--agents-dir", type=Path, default=Path("agents"), help="Default: agents")
    up.add_argument("--log-dir", type=Path, help="Default: runs/fleet/<timestamp>")
    up.add_argument("--trace-dir", type=Path, help="Per-agent AgentOS traces under this folder")
    up.add_argument("--ready-timeout", type=float, default=60.0, help="Seconds (default: 60)")

    watch = sub.add_parser("watch", help="Live page of every agent's A2A tasks and steps")
    watch.add_argument(
        "--trace-root",
        type=Path,
        default=Path("runs") / "fleet-traces",
        help="The folder given to `up --trace-dir` (default: runs/fleet-traces)",
    )
    watch.add_argument("--agents-dir", type=Path, default=Path("agents"), help="Default: agents")
    watch.add_argument("--port", type=int, default=9130, help="Default: 9130 (127.0.0.1 only)")

    card = sub.add_parser("card", help="Fetch and print an agent's card")
    card.add_argument("url")

    ask = sub.add_parser("ask", help="Find an agent by its card and give it a task")
    ask.add_argument("url")
    ask.add_argument("task")
    ask.add_argument("--no-stream", action="store_true", help="One blocking request")
    ask.add_argument("--timeout", type=float, default=900.0, help="Seconds (default: 900)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    if args.command == "up":
        return _up(args)
    if args.command == "watch":
        return _watch(args)
    if args.command == "card":
        return asyncio.run(_card(args))
    if args.command == "ask":
        return asyncio.run(_ask(args))
    build_parser().print_help()
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from agentmesh.runtime import agentos_runner
    from agentmesh.server import build_app
    from agentmesh.spec import SpecError, load_spec

    try:
        spec = load_spec(args.spec)
    except SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2
    if args.port:
        spec = spec.model_copy(update={"port": args.port})
    journal = None
    if args.trace_dir is not None:
        from agentmesh.journal import JOURNAL_NAME, JsonlJournal

        journal = JsonlJournal(args.trace_dir / JOURNAL_NAME, spec.name)
    app = build_app(spec, agentos_runner(spec, args.trace_dir), journal)
    print(f"{spec.name} on {spec.url} (model {spec.model}, workspace {spec.workspace})")
    uvicorn.run(app, host=spec.host, port=spec.port, log_level="warning")
    return 0


def _up(args: argparse.Namespace) -> int:
    import time

    from agentmesh import fleet
    from agentmesh.spec import SpecError

    paths = args.specs or fleet.default_specs(args.agents_dir)
    log_dir = args.log_dir or Path("runs") / "fleet" / time.strftime("%Y%m%d-%H%M%S")
    try:
        members = fleet.plan(paths, log_dir)
    except SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2
    fleet.start(members, args.trace_dir)
    try:
        cards = fleet.wait_ready(members, args.ready_timeout)
    except (RuntimeError, TimeoutError) as exc:
        print(f"startup failed: {exc}", file=sys.stderr)
        fleet.stop(members)
        return 1
    for m in members:
        print(fleet.describe(m, cards[m.path.stem]))
    print(f"{len(members)} agents up; logs in {log_dir}. Ctrl+C stops them all.", flush=True)
    return fleet.watch(members)


def _watch(args: argparse.Namespace) -> int:
    import threading

    from agentmesh import fleet, watch
    from agentmesh.spec import SpecError, load_spec

    agents = []
    for path in fleet.default_specs(args.agents_dir):
        if not path.exists():
            continue
        try:
            spec = load_spec(path)
        except SpecError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        agents.append(watch.AgentInfo(path.stem, spec.name, spec.url, spec.model))
    watcher = watch.Watcher(args.trace_root, agents)
    try:
        server = watch.serve(watcher, args.port)
    except OSError as exc:
        print(f"cannot listen on 127.0.0.1:{args.port}: {exc}", file=sys.stderr)
        return 1
    threading.Thread(target=watcher.run_probes, daemon=True).start()
    print(f"watching {args.trace_root} on http://127.0.0.1:{args.port}/ (Ctrl+C stops)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        server.server_close()
    return 0


async def _card(args: argparse.Namespace) -> int:
    from agentmesh.client import discover

    async with httpx.AsyncClient(timeout=30) as http:
        card = await discover(args.url, http)
    print(json.dumps(MessageToDict(card), indent=2))
    return 0


async def _ask(args: argparse.Namespace) -> int:
    from agentmesh.client import ask, discover, state_name

    async with httpx.AsyncClient(timeout=args.timeout) as http:
        card = await discover(args.url, http)
        skills = ", ".join(s.id for s in card.skills)
        print(f"found {card.name!r} at {args.url}, skills: {skills}", file=sys.stderr)

        def show(event) -> None:
            kind = event.WhichOneof("payload")
            if kind == "status_update":
                s = event.status_update.status
                text = " ".join(p.text for p in s.message.parts) if s.HasField("message") else ""
                print(f"  [{state_name(s.state)}] {text}", file=sys.stderr)
            elif kind == "task":
                t = event.task
                print(f"  [{state_name(t.status.state)}] task {t.id}", file=sys.stderr)

        run = await ask(card, args.task, http, streaming=not args.no_stream, on_event=show)
    if run.answer:
        print(run.answer)
    if run.final_message:
        print(f"[{run.state}] {run.final_message}", file=sys.stderr)
    if run.stats:
        print(f"[{run.state}] {json.dumps(run.stats)}", file=sys.stderr)
    return 0 if run.state == "completed" else 3
