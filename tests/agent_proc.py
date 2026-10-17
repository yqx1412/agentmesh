"""A test agent in its own process, so a test can kill it mid-task (tests/test_m4.py).

    python agent_proc.py <name> <port> <skill> <seconds> <journal path>

Every task takes ``seconds``, reporting progress every 0.1 s, and answers
"done by <name>". It stops early when its task is cancelled. No model, no tools.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import uvicorn

from agentmesh.executor import Cancelled, RunOutcome
from agentmesh.journal import JsonlJournal
from agentmesh.server import build_app
from agentmesh.spec import AgentSpec, SkillSpec


def main() -> None:
    name, port, skill, seconds, journal = sys.argv[1:6]
    spec = AgentSpec(
        name=name,
        description=f"{name}, a test agent",
        port=int(port),
        workspace=Path(journal).parent / "ws",
        skills=[SkillSpec(id=skill, name=skill, description=f"does {skill}")],
    )

    def run(task, progress, flag):
        end = time.monotonic() + float(seconds)
        n = 0
        while time.monotonic() < end:
            if flag.is_set():
                raise Cancelled
            n += 1
            progress(f"tick {n}")
            time.sleep(0.1)
        return RunOutcome(answer=f"done by {name}", stop_reason="final_answer", steps=n)

    app = build_app(spec, run, JsonlJournal(Path(journal), name))
    uvicorn.run(app, host="127.0.0.1", port=int(port), log_level="warning")


if __name__ == "__main__":
    main()
