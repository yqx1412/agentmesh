"""The bridge from A2A to AgentOS: an ``AgentExecutor`` that runs one AgentOS task per A2A task.

Lifecycle, as the A2A client sees it:

- ``submitted``: the task object is enqueued as soon as the request arrives.
- ``working``: once the agent starts, then again for every model step and tool call, each
  with a one-line status message (visible to streaming clients).
- ``completed``: the answer as a ``result`` artifact, with run statistics in its metadata.
- ``failed``: the agent stopped without a final answer, or the model or a tool server failed.
- ``canceled``: the client cancelled; the agent stops before its next model call.

AgentOS is synchronous, so each run happens in a worker thread. Progress crosses back into
the event loop through ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


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


class AgentOSExecutor(AgentExecutor):
    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self._cancel_flags: dict[str, threading.Event] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        text = get_message_text(context.message).strip() if context.message else ""
        if not text:
            await updater.reject(_agent_text(updater, "The request has no text to act on."))
            return

        await updater.start_work(_agent_text(updater, "Starting the agent."))
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
            await updater.failed(_agent_text(updater, f"Agent error: {type(exc).__name__}: {exc}"))
            return
        finally:
            self._cancel_flags.pop(task.id, None)

        if outcome.ok:
            await updater.add_artifact(
                [Part(text=outcome.answer)], name="result", metadata=outcome.stats()
            )
            await updater.complete()
        else:
            detail = f"stopped without a final answer ({outcome.stop_reason})"
            if outcome.answer:
                detail += f"; last reply: {outcome.answer[:500]}"
            await updater.failed(_agent_text(updater, detail))

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
