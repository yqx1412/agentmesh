"""The A2A lifecycle journal: one JSON line per task state change, beside the agent's traces.

``agentmesh serve --trace-dir D`` writes ``D/a2a.jsonl``. Each record is::

    {"ts": 1791733808.8, "agent": "Analyst Agent", "task": "<uuid>", "context": "<uuid>",
     "state": "submitted" | "working" | "completed" | "failed" | "rejected" | "canceled",
     ...state-specific fields: text, message, answer, stats}

The AgentOS trace for the same task is ``D/<timestamp>-<task id>.jsonl``; ``agentmesh
watch`` joins the two.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

JOURNAL_NAME = "a2a.jsonl"


class JsonlJournal:
    """Appends records to ``path``. Thread-safe; each write is one complete line."""

    def __init__(self, path: Path, agent: str) -> None:
        self.path = path
        self.agent = agent
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: dict[str, Any]) -> None:
        line = json.dumps(
            {"ts": round(time.time(), 3), "agent": self.agent, **record},
            ensure_ascii=False,
            default=str,
        )
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
