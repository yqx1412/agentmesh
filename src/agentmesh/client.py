"""A plain A2A client: read an agent's card, send it a task, follow the task to the end."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_message_text, new_text_message
from a2a.types import AgentCard, Role, SendMessageRequest, StreamResponse, TaskState
from google.protobuf.json_format import MessageToDict

TERMINAL = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}


def state_name(state: int) -> str:
    return TaskState.Name(state).removeprefix("TASK_STATE_").lower()


@dataclass
class TaskRun:
    """Everything a client saw for one task, in order."""

    task_id: str = ""
    states: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    answer: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    final_message: str = ""

    @property
    def state(self) -> str:
        return self.states[-1] if self.states else ""


async def discover(url: str, http: httpx.AsyncClient) -> AgentCard:
    return await A2ACardResolver(httpx_client=http, base_url=url).get_agent_card()


def _note_state(run: TaskRun, state: int) -> None:
    name = state_name(state)
    if not run.states or run.states[-1] != name:
        run.states.append(name)


async def ask(
    card: AgentCard,
    text: str,
    http: httpx.AsyncClient,
    *,
    streaming: bool = True,
    on_event: Callable[[StreamResponse], None] | None = None,
) -> TaskRun:
    client = await create_client(
        agent=card, client_config=ClientConfig(streaming=streaming, httpx_client=http)
    )
    run = TaskRun()
    request = SendMessageRequest(message=new_text_message(text, role=Role.ROLE_USER))
    try:
        async for event in client.send_message(request):
            if on_event:
                on_event(event)
            kind = event.WhichOneof("payload")
            if kind == "task":
                t = event.task
                run.task_id = t.id
                _note_state(run, t.status.state)
                for a in t.artifacts:
                    _take_artifact(run, a)
                if t.status.HasField("message"):
                    run.final_message = get_message_text(t.status.message)
            elif kind == "status_update":
                s = event.status_update.status
                _note_state(run, s.state)
                if s.HasField("message"):
                    line = get_message_text(s.message)
                    if s.state == TaskState.TASK_STATE_WORKING:
                        run.progress.append(line)
                    else:
                        run.final_message = line
            elif kind == "artifact_update":
                _take_artifact(run, event.artifact_update.artifact)
            elif kind == "message":
                run.answer = get_message_text(event.message)
    finally:
        await client.close()
    return run


def _take_artifact(run: TaskRun, artifact) -> None:
    if artifact.name == "result":
        run.answer = "".join(p.text for p in artifact.parts)
        raw = MessageToDict(artifact.metadata) if artifact.HasField("metadata") else {}
        # protobuf Struct stores every number as a double; give the counters back as ints.
        run.stats = {
            k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in raw.items()
        }
