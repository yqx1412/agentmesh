"""The A2A server: an Agent Card at ``/.well-known/agent-card.json`` and JSON-RPC at ``/``."""

from __future__ import annotations

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from starlette.applications import Starlette

from agentmesh.executor import AgentOSExecutor, Runner
from agentmesh.spec import AgentSpec


def build_card(spec: AgentSpec) -> AgentCard:
    return AgentCard(
        name=spec.name,
        description=spec.description,
        version=spec.version,
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
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


def build_app(spec: AgentSpec, runner: Runner) -> Starlette:
    card = build_card(spec)
    handler = DefaultRequestHandler(
        agent_executor=AgentOSExecutor(runner),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    return Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])
