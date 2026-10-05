"""``agentmesh serve <spec> | card <url> | ask <url> <task>``."""

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
    app = build_app(spec, agentos_runner(spec, args.trace_dir))
    print(f"{spec.name} on {spec.url} (model {spec.model}, workspace {spec.workspace})")
    uvicorn.run(app, host=spec.host, port=spec.port, log_level="warning")
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
