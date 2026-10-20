"""The bridge from A2A to AgentOS: an ``AgentExecutor`` that runs one AgentOS task per A2A task.

Lifecycle, as the A2A client sees it:

- ``submitted``: the task object is enqueued as soon as the request arrives.
- ``working``: once the agent starts, then again for every model step and tool call, each
  with a one-line status message (visible to streaming clients).
- ``completed``: the answer as a ``result`` artifact, with run statistics in its metadata.
- ``failed``: the agent stopped without a final answer, or the model or a tool server failed.
- ``canceled``: the client cancelled; the agent stops before its next model call.
- ``rejected``: the request has no text, or (with logins on, M5) the caller may not use the
  skill it asked for; see ``access.py``. A signing agent adds an ``attestation`` artifact
  to every completed task.

AgentOS is synchronous, so each run happens in a worker thread. Progress crosses back into
the event loop through ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState

from agentmesh.access import GRANT, Access, Refused

log = logging.getLogger(__name__)

Progress = Callable[[str], None]

# Receives one dict per lifecycle change: {"task", "context", "state", ...}. See journal.py.
Journal = Callable[[dict[str, Any]], None]

# The A2A task the current run belongs to. Set in execute(); asyncio.to_thread copies it into
# the worker thread, so the runner can name its trace file after the task.
TASK_ID: ContextVar[str | None] = ContextVar("agentmesh_task_id", default=None)

# Longest answer or task text copied into the journal.
_PREVIEW = 2000


@dataclass
class RunOutcome:
    """What one AgentOS run produced, independent of which agent class ran it."""

    answer: str
    stop_reason: str
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.stop_reason == "final_answer"

    def stats(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class Cancelled(Exception):
    """The A2A task was cancelled; raised inside the worker thread at the next model call."""


# (task text, progress callback, cancel flag) -> outcome. Runs in a worker thread.
Runner = Callable[[str, Progress, threading.Event], RunOutcome]


def _agent_text(updater: TaskUpdater, text: str):
    return updater.new_agent_message([Part(text=text)])


def _meta_str(context: RequestContext, key: str, limit: int = 100) -> str:
    msg = context.message
    if msg is None or not msg.HasField("metadata") or key not in msg.metadata.fields:
        return ""
    value = msg.metadata.fields[key]
    return value.string_value[:limit] if value.WhichOneof("kind") == "string_value" else ""


def _meta_strings(context: RequestContext, key: str) -> list[str]:
    msg = context.message
    if msg is None or not msg.HasField("metadata") or key not in msg.metadata.fields:
        return []
    value = msg.metadata.fields[key]
    if value.WhichOneof("kind") != "list_value":
        return []
    return [
        v.string_value for v in value.list_value.values if v.WhichOneof("kind") == "string_value"
    ]


def _origin(context: RequestContext) -> dict[str, str]:
    """The orchestration run and step that sent this task, from the message metadata."""
    out = {}
    for key in ("orchestration", "step"):
        if value := _meta_str(context, key):
            out[key] = value
    return out


def _caller(context: RequestContext) -> str:
    """The logged-in principal ("" without auth)."""
    user = context.call_context.user if context.call_context else None
    return user.user_name if user is not None and user.is_authenticated else ""


class AgentOSExecutor(AgentExecutor):
    def __init__(
        self, runner: Runner, journal: Journal | None = None, access: Access | None = None
    ) -> None:
        self.runner = runner
        self.journal = journal
        self.access = access  # None: no logins (tests, `serve --insecure`)
        self._cancel_flags: dict[str, threading.Event] = {}

    def _log(self, task_id: str, context_id: str, state: str, **data: Any) -> None:
        if self.journal is None:
            return
        try:
            self.journal({"task": task_id, "context": context_id, "state": state, **data})
        except Exception:  # a broken journal must never fail the task
            log.exception("journal write failed")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        def note(state: str, **data: Any) -> None:
            self._log(task.id, task.context_id, state, **data)

        text = get_message_text(context.message).strip() if context.message else ""
        caller, skill = _caller(context), _meta_str(context, "skill")
        who = {k: v for k, v in (("caller", caller), ("skill", skill)) if v}
        note("submitted", text=text[:_PREVIEW], **_origin(context), **who)
        if not text:
            msg = "The request has no text to act on."
            await updater.reject(_agent_text(updater, msg))
            note("rejected", message=msg)
            return

        grant = None
        if self.access is not None:
            try:
                grant = self.access.admit(caller, skill, _meta_strings(context, "attestations"))
            except Refused as exc:
                msg = f"Refused: {exc}"
                await updater.reject(_agent_text(updater, msg))
                note("rejected", message=msg)
                return
        GRANT.set(grant)

        TASK_ID.set(task.id)
        await updater.start_work(_agent_text(updater, "Starting the agent."))
        note("working", message="Starting the agent.")
        loop = asyncio.get_running_loop()
        progress: asyncio.Queue[str | None] = asyncio.Queue()
        flag = self._cancel_flags.setdefault(task.id, threading.Event())

        def report(line: str) -> None:  # called from the worker thread
            loop.call_soon_threadsafe(progress.put_nowait, line)

        def work() -> RunOutcome:
            try:
                return self.runner(text, report, flag)
            finally:
                loop.call_soon_threadsafe(progress.put_nowait, None)

        job = asyncio.ensure_future(asyncio.to_thread(work))
        try:
            while (line := await progress.get()) is not None:
                await updater.update_status(
                    TaskState.TASK_STATE_WORKING, _agent_text(updater, line)
                )
                note("working", message=line)
            outcome = await job
        except asyncio.CancelledError:
            # The framework cancels execute() and then calls cancel(). The thread cannot be
            # killed, so tell it to stop at its next model call.
            flag.set()
            raise
        except Cancelled:
            return  # cancel() publishes the canceled state
        except Exception as exc:
            log.exception("agent run failed")
            msg = f"Agent error: {type(exc).__name__}: {exc}"
            await updater.failed(_agent_text(updater, msg))
            note("failed", message=msg)
            return
        finally:
            self._cancel_flags.pop(task.id, None)

        if outcome.ok:
            if grant is not None and (report_text := grant.write_report()):
                outcome.answer = f"{outcome.answer.rstrip()}\n\n{report_text}"
            await updater.add_artifact(
                [Part(text=outcome.answer)], name="result", metadata=outcome.stats()
            )
            signed = None
            if self.access is not None:
                signed = self.access.sign(
                    task=task.id, caller=caller, input=text, answer=outcome.answer
                )
            if signed:
                await updater.add_artifact([Part(text=signed)], name="attestation")
            await updater.complete()
            extra = {"signed_as": self.access.signer} if signed and self.access else {}
            note("completed", answer=outcome.answer[:_PREVIEW], stats=outcome.stats(), **extra)
        else:
            detail = f"stopped without a final answer ({outcome.stop_reason})"
            if outcome.answer:
                detail += f"; last reply: {outcome.answer[:500]}"
            await updater.failed(_agent_text(updater, detail))
            note("failed", message=detail, stats=outcome.stats())

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or (context.current_task.id if context.current_task else None)
        if task_id and task_id in self._cancel_flags:
            self._cancel_flags[task_id].set()
        context_id = context.context_id or (
            context.current_task.context_id if context.current_task else ""
        )
        updater = TaskUpdater(event_queue, task_id or "", context_id or "")
        with contextlib.suppress(RuntimeError):  # already terminal
            await updater.cancel(new_text_message("Cancelled by the client."))
            self._log(
                task_id or "", context_id or "", "canceled", message="Cancelled by the client."
            )
