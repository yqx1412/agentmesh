"""A plain A2A client: read an agent's card, send it a task, follow the task to the end.

Two ways to follow a task:

- ``ask``: stream it. The connection stays open and every update arrives on it.
- ``ask_push``: send it with ``return_immediately`` and a webhook (a ``PushReceiver``). The
  agent POSTs each update to the webhook; nothing stays open. If no notification arrives
  for ``poll_every`` seconds the client asks the agent for the task, so a lost notification
  costs time, not the task, and a dead agent is noticed.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_message_text, new_text_message
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    GetTaskRequest,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    StreamResponse,
    TaskPushNotificationConfig,
    TaskState,
)
from google.protobuf.json_format import MessageToDict, ParseDict
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

TERMINAL = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}
TERMINAL_NAMES = {"completed", "failed", "canceled", "rejected"}

# The header the A2A SDK's push sender puts the config's token in.
TOKEN_HEADER = "X-A2A-Notification-Token"


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
    pushes: int = 0  # notifications received (ask_push only)
    polls: int = 0  # get_task calls made because no notification came (ask_push only)

    @property
    def state(self) -> str:
        return self.states[-1] if self.states else ""

    @property
    def done(self) -> bool:
        return self.state in TERMINAL_NAMES


async def discover(url: str, http: httpx.AsyncClient) -> AgentCard:
    return await A2ACardResolver(httpx_client=http, base_url=url).get_agent_card()


def _note_state(run: TaskRun, state: int) -> None:
    name = state_name(state)
    if not run.states or run.states[-1] != name:
        run.states.append(name)


def apply_event(run: TaskRun, event: StreamResponse) -> None:
    """Fold one A2A event (streamed, pushed or polled) into ``run``."""
    kind = event.WhichOneof("payload")
    if kind == "task":
        t = event.task
        run.task_id = t.id
        _note_state(run, t.status.state)
        for a in t.artifacts:
            _take_artifact(run, a)
        if t.status.HasField("message"):
            line = get_message_text(t.status.message)
            if t.status.state == TaskState.TASK_STATE_WORKING:
                if not run.progress or run.progress[-1] != line:
                    run.progress.append(line)
            else:
                run.final_message = line
    elif kind == "status_update":
        s = event.status_update.status
        run.task_id = run.task_id or event.status_update.task_id
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


def _message(text: str, metadata: dict[str, Any] | None):
    message = new_text_message(text, role=Role.ROLE_USER)
    if metadata:
        message.metadata.update(metadata)
    return message


async def ask(
    card: AgentCard,
    text: str,
    http: httpx.AsyncClient,
    *,
    streaming: bool = True,
    on_event: Callable[[StreamResponse], None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskRun:
    """Send ``text`` and follow the task to the end. ``metadata`` rides on the message
    (the orchestrator uses it to tag the tasks it starts)."""
    client = await create_client(
        agent=card, client_config=ClientConfig(streaming=streaming, httpx_client=http)
    )
    run = TaskRun()
    request = SendMessageRequest(message=_message(text, metadata))
    # No client.close(): the SDK's JSON-RPC transport closes only the httpx client, which
    # belongs to the caller (the orchestrator sends many tasks through one).
    async for event in client.send_message(request):
        if on_event:
            on_event(event)
        apply_event(run, event)
    return run


async def ask_push(
    card: AgentCard,
    text: str,
    http: httpx.AsyncClient,
    receiver: PushReceiver,
    *,
    on_event: Callable[[StreamResponse], None] | None = None,
    metadata: dict[str, Any] | None = None,
    poll_every: float = 15.0,
) -> TaskRun:
    """Like ``ask``, but the agent reports by push notification instead of a stream."""
    client = await create_client(
        agent=card, client_config=ClientConfig(streaming=False, httpx_client=http)
    )
    run = TaskRun()
    token, inbox = receiver.expect()
    config = SendMessageConfiguration(
        return_immediately=True,
        task_push_notification_config=TaskPushNotificationConfig(url=receiver.url, token=token),
    )
    request = SendMessageRequest(message=_message(text, metadata), configuration=config)
    try:
        async for event in client.send_message(request):
            if on_event:
                on_event(event)
            apply_event(run, event)
        while not run.done:
            try:
                event = await asyncio.wait_for(inbox.get(), poll_every)
                run.pushes += 1
            except TimeoutError:
                # Raises if the agent is gone, which is how a dead agent is noticed here.
                run.polls += 1
                event = StreamResponse(task=await client.get_task(GetTaskRequest(id=run.task_id)))
            if on_event:
                on_event(event)
            apply_event(run, event)
    finally:
        receiver.forget(token)
    return run


async def cancel(card: AgentCard, task_id: str, http: httpx.AsyncClient) -> str:
    """Ask the agent to cancel a task. Returns the state it reports, or why that failed.
    Never raises: cancelling is cleanup, and the agent may already be gone."""
    try:
        client = await create_client(
            agent=card, client_config=ClientConfig(streaming=False, httpx_client=http)
        )
        task = await client.cancel_task(CancelTaskRequest(id=task_id))
        return state_name(task.status.state)
    except Exception as exc:
        return f"cancel failed: {type(exc).__name__}: {exc}"


def _take_artifact(run: TaskRun, artifact) -> None:
    if artifact.name == "result":
        run.answer = "".join(p.text for p in artifact.parts)
        raw = MessageToDict(artifact.metadata) if artifact.HasField("metadata") else {}
        # protobuf Struct stores every number as a double; give the counters back as ints.
        run.stats = {
            k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in raw.items()
        }


class PushReceiver:
    """A webhook for A2A push notifications, bound to loopback.

    Each task gets its own random token (``expect``); a POST without a known token is
    refused, so only an agent we gave a task can deliver to that task's inbox. ``app`` is a
    plain ASGI app; ``serve`` runs it on ``127.0.0.1``.
    """

    def __init__(self, url: str = "") -> None:
        self.url = url
        self._inboxes: dict[str, asyncio.Queue[StreamResponse]] = {}
        self.refused = 0
        self.app = Starlette(routes=[Route("/", self._receive, methods=["POST"])])
        self._server: Any = None
        self._serving: asyncio.Task[None] | None = None

    def expect(self) -> tuple[str, asyncio.Queue[StreamResponse]]:
        token = secrets.token_urlsafe(24)
        inbox: asyncio.Queue[StreamResponse] = asyncio.Queue()
        self._inboxes[token] = inbox
        return token, inbox

    def forget(self, token: str) -> None:
        self._inboxes.pop(token, None)

    async def _receive(self, request: Request) -> Response:
        inbox = self._inboxes.get(request.headers.get(TOKEN_HEADER, ""))
        if inbox is None:
            self.refused += 1
            return Response(status_code=403)
        try:
            event = ParseDict(await request.json(), StreamResponse(), ignore_unknown_fields=True)
        except Exception:
            return Response(status_code=400)
        inbox.put_nowait(event)
        return Response(status_code=204)

    async def serve(self, port: int = 0) -> PushReceiver:
        """Start listening on ``127.0.0.1:<port>`` (0 = any free port); sets ``url``."""
        import uvicorn

        config = uvicorn.Config(self.app, host="127.0.0.1", port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None  # leave Ctrl+C to the caller
        self._serving = asyncio.create_task(self._server.serve())
        while not self._server.started:
            if self._serving.done():
                self._serving.result()  # raises the startup error
            await asyncio.sleep(0.02)
        sock = self._server.servers[0].sockets[0]
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}/"
        return self

    async def close(self) -> None:
        if self._server is not None and self._serving is not None:
            self._server.should_exit = True
            with contextlib.suppress(Exception):
                await self._serving

    async def __aenter__(self) -> PushReceiver:
        return await self.serve()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
