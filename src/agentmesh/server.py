"""The A2A server: an Agent Card at ``/.well-known/agent-card.json`` and JSON-RPC at ``/``.

Push notifications (M4): a client can give a task a webhook URL, and the agent POSTs every
state change of that task to it, so the client doesn't have to hold a stream open. The
mesh runs on one machine, so webhooks are accepted only on loopback addresses. That keeps
an agent from being used to POST at other hosts (the library's own validator does the
opposite: it allows only public hosts).
"""

from __future__ import annotations

import ipaddress
import urllib.parse

import httpx
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

from agentmesh.executor import AgentOSExecutor, Journal, Runner
from agentmesh.spec import AgentSpec

# How long an agent waits for a webhook to accept one notification.
PUSH_TIMEOUT = 10.0


def is_loopback_url(url: str) -> bool:
    """An http(s) URL whose host is a literal loopback address or ``localhost``."""
    try:
        parsed = urllib.parse.urlparse(url)
        parsed.port  # noqa: B018 - raises on a malformed port
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False  # any other name could resolve anywhere


async def loopback_only(url: str) -> bool:
    return is_loopback_url(url)


def build_card(spec: AgentSpec) -> AgentCard:
    return AgentCard(
        name=spec.name,
        description=spec.description,
        version=spec.version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=spec.url, protocol_version="1.0")
        ],
        skills=[
            AgentSkill(
                id=s.id,
                name=s.name,
                description=s.description,
                tags=s.tags,
                examples=s.examples,
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
            for s in spec.skills
        ],
    )


def build_app(
    spec: AgentSpec,
    runner: Runner,
    journal: Journal | None = None,
    *,
    push_http: httpx.AsyncClient | None = None,
) -> Starlette:
    """``push_http`` is the client notifications are POSTed with (tests pass one that
    routes in-process); by default the agent makes its own."""
    card = build_card(spec)
    push_store = InMemoryPushNotificationConfigStore()
    sender = BasePushNotificationSender(
        push_http or httpx.AsyncClient(timeout=PUSH_TIMEOUT),
        push_store,
        push_url_validator=loopback_only,
    )
    handler = DefaultRequestHandler(
        agent_executor=AgentOSExecutor(runner, journal),
        task_store=InMemoryTaskStore(),
        agent_card=card,
        push_config_store=push_store,
        push_sender=sender,
        push_url_validator=loopback_only,
    )
    return Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])
