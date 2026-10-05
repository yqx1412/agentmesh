"""The default runner: a real AgentOS agent (Ollama model, built-in and MCP tools)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from agentos.agent import Tracer
from agentos.bench.runner import make_agent
from agentos.builtin_tools import builtin_tools
from agentos.config import AgentOSConfig, load_config
from agentos.llm import ChatResponse, Message, OllamaLLM
from agentos.mcp_client import MCPManager
from agentos.tools import Policy, ToolRegistry

from agentmesh.executor import Cancelled, Progress, Runner, RunOutcome
from agentmesh.spec import AgentSpec


class ProgressTracer(Tracer):
    """AgentOS's JSONL tracer, plus one human-readable progress line per step and tool call."""

    def __init__(self, path: Path | None, progress: Progress) -> None:
        super().__init__(path)
        self.progress = progress

    def emit(self, event: str, **data: Any) -> None:
        super().emit(event, **data)
        if event == "llm_response":
            calls = [c["name"] for c in data.get("tool_calls") or []]
            what = f"calling {', '.join(calls)}" if calls else "writing the answer"
            self.progress(f"step {data.get('step')}: {what}")
        elif event == "tool_result":
            name = (data.get("call") or {}).get("name", "?")
            self.progress(f"{name}: {'ok' if data.get('ok') else 'error'}")


class CancellableLLM:
    """Checks the cancel flag before every model call, the only safe point to stop."""

    def __init__(self, inner: OllamaLLM, flag: threading.Event) -> None:
        self.inner = inner
        self.model = inner.model
        self.flag = flag

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse:
        if self.flag.is_set():
            raise Cancelled()
        return self.inner.chat(messages, tools)


def agentos_runner(spec: AgentSpec, trace_dir: Path | None = None) -> Runner:
    config = load_config(spec.agentos_config) if spec.agentos_config else AgentOSConfig()
    spec.workspace.mkdir(parents=True, exist_ok=True)

    def run(task: str, progress: Progress, flag: threading.Event) -> RunOutcome:
        trace = None
        if trace_dir is not None:
            trace = trace_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{threading.get_ident()}.jsonl"
        llm = OllamaLLM(spec.model, spec.ollama_url, num_ctx=spec.num_ctx)
        try:
            # Tool servers start per task, so one task's MCP state never leaks into the next.
            with MCPManager(config.enabled_servers(spec.workspace)) as mcp:
                registry = ToolRegistry(
                    [*builtin_tools(spec.workspace), *mcp.tools()],
                    policy=Policy(auto_approve=spec.allow),
                )
                agent = make_agent(
                    spec.kind,
                    CancellableLLM(llm, flag),
                    registry,
                    max_steps=spec.max_steps,
                    tracer=ProgressTracer(trace, progress),
                    context_budget=int(0.75 * spec.num_ctx),
                )
                r = agent.run(task)
        finally:
            llm.close()
        return RunOutcome(
            answer=r.answer,
            stop_reason=r.stop_reason,
            steps=r.steps,
            tool_calls=r.tool_calls,
            tool_errors=r.tool_errors,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
        )

    return run
