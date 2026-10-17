"""The default runner: a real AgentOS agent (Ollama model, built-in and MCP tools)."""

from __future__ import annotations

import dataclasses
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from agentos.agent import DEFAULT_SYSTEM_PROMPT, Agent, Tracer
from agentos.bench.runner import make_agent
from agentos.builtin_tools import builtin_tools
from agentos.config import AgentOSConfig, load_config
from agentos.llm import ChatResponse, Message, OllamaLLM
from agentos.mcp_client import MCPManager
from agentos.sandbox import sandbox_tools
from agentos.tools import Policy, Tool, ToolError, ToolRegistry

from agentmesh.access import GRANT, Grant
from agentmesh.executor import TASK_ID, Cancelled, Progress, Runner, RunOutcome
from agentmesh.spec import SANDBOX_TOOLS, AgentSpec


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


def _matches(tool_name: str | None, wanted: str) -> bool:
    """MCP tools are registered as ``<server>__<tool>``; accept either form."""
    return tool_name is not None and (tool_name == wanted or tool_name.endswith(f"__{wanted}"))


def missing_calls(messages: list[Message], required: dict[str, int]) -> dict[str, int]:
    """How many more successful calls of each required tool the history needs."""
    missing = {}
    for wanted, need in required.items():
        done = sum(
            1
            for m in messages
            if m.role == "tool"
            and _matches(m.tool_name, wanted)
            and not m.content.startswith("ERROR:")
        )
        if done < need:
            missing[wanted] = need - done
    return missing


_QUOTE = re.compile(r"[\"\u201c]([^\"\u201c\u201d\n]{20,})[\"\u201d]")


def _norm(text: str) -> str:
    """Lower-case words only, so quotes survive whitespace, markup and punctuation changes."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def invented_quotes(answer: str, messages: list[Message]) -> list[str]:
    """Quoted strings in ``answer`` that appear in no successful tool result."""
    seen = " ".join(
        _norm(m.content)
        for m in messages
        if m.role == "tool" and not m.content.startswith("ERROR:")
    )
    return [q for q in _QUOTE.findall(answer) if _norm(q) not in seen]


class AnswerGuardLLM:
    """Sends an unacceptable final answer back to the model with a reminder.

    A reply with no tool calls is the agent's final answer. It's refused when it's empty,
    while the history lacks the spec's required tool results, or (with ``check_quotes``) while
    it quotes text no tool returned. The model is then asked again with one extra user
    message (not kept in
    the agent's history). After ``max_nudges`` the answer goes through anyway, so a tool that
    keeps failing can't trap the task.
    """

    def __init__(
        self,
        inner: Any,
        required: dict[str, int] | None = None,
        *,
        check_quotes: bool = False,
        max_nudges: int = 3,
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self.required = required or {}
        self.check_quotes = check_quotes
        self.max_nudges = max_nudges
        self.nudges = 0

    def _problem(self, answer: str, messages: list[Message]) -> str | None:
        if not answer.strip():
            return (
                "Your reply was empty. Do the task now: call the tools it needs, then reply "
                "with the result."
            )
        missing = missing_calls(messages, self.required)
        if missing:
            need = ", ".join(f"{n} more successful {t} call(s)" for t, n in missing.items())
            return f"Not yet: before answering you need {need}. Call the tool now."
        if self.check_quotes:
            bad = invented_quotes(answer, messages)
            if bad:
                listed = "; ".join(f'"{q}"' for q in bad)
                return (
                    "These quotes are not in any page you fetched: "
                    f"{listed}. Quote only text copied exactly from a fetched page. If no "
                    "passage supports or contradicts a claim, its verdict is NOT FOUND. "
                    "Fetch again or read further if you need to, then rewrite the answer."
                )
        return None

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse:
        resp = self.inner.chat(messages, tools)
        extra: list[Message] = []
        while not resp.message.tool_calls and self.nudges < self.max_nudges:
            # Tool results fetched during earlier nudges count too.
            problem = self._problem(resp.message.content, [*messages, *extra])
            if problem is None:
                break
            self.nudges += 1
            extra += [resp.message, Message(role="user", content=problem)]
            resp = self.inner.chat([*messages, *extra], tools)
        return resp


# The M2 name, kept for the existing tests and callers.
RequireToolsLLM = AnswerGuardLLM


def _mcp_log(trace_dir: Path | None) -> TextIO:
    """Where MCP servers' stderr goes: ``mcp.log`` beside the traces, or the null device."""
    if trace_dir is None:
        return open(os.devnull, "w", encoding="utf-8")
    trace_dir.mkdir(parents=True, exist_ok=True)
    return (trace_dir / "mcp.log").open("a", encoding="utf-8")


def apply_grant(tools: list[Tool], grant: Grant | None) -> list[Tool]:
    """M5: only the tools the task's skill allows, and every attested tool gated on a
    source the Verifier signed off on (see ``access.Grant.spend``). No grant, no change."""
    if grant is None:
        return tools
    out = []
    for tool in tools:
        if not grant.allows_tool(tool.name):
            continue
        if grant.is_attested(tool.name):
            tool = dataclasses.replace(tool, fn=_attested(tool, grant))
        out.append(tool)
    return out


def _attested(tool: Tool, grant: Grant):
    inner = tool.fn

    def call(args: Any) -> Any:
        url, why = grant.spend(tool.name, args)
        if why is not None:
            raise ToolError(why)
        try:
            return inner(args)
        except Exception:
            grant.refund(url)  # a failed write doesn't use up the claim
            raise

    return call


def select_tools(spec: AgentSpec, config: AgentOSConfig) -> list[Tool]:
    """The built-in tools the spec names, in the spec's order (sandbox ones included)."""
    available = {t.name: t for t in builtin_tools(spec.workspace)}
    if any(name in SANDBOX_TOOLS for name in spec.builtin_tools):
        available.update({t.name: t for t in sandbox_tools(spec.workspace, config.sandbox)})
    return [available[name] for name in spec.builtin_tools]


def system_prompt(spec: AgentSpec) -> str:
    if not spec.instructions:
        return DEFAULT_SYSTEM_PROMPT
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{spec.instructions.strip()}"


def agentos_runner(spec: AgentSpec, trace_dir: Path | None = None) -> Runner:
    config = load_config(spec.agentos_config) if spec.agentos_config else AgentOSConfig()
    spec.workspace.mkdir(parents=True, exist_ok=True)
    prompt = system_prompt(spec)

    def run(task: str, progress: Progress, flag: threading.Event) -> RunOutcome:
        grant = GRANT.get()
        trace = None
        if trace_dir is not None:
            # Named after the A2A task (see executor.TASK_ID) so `agentmesh watch` can join
            # the trace to the task's lifecycle; the thread id is the fallback outside A2A.
            run_id = TASK_ID.get() or str(threading.get_ident())
            trace = trace_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id}.jsonl"
        llm = OllamaLLM(spec.model, spec.ollama_url, num_ctx=spec.num_ctx, think=spec.think)
        try:
            # Tool servers start per task, so one task's MCP state never leaks into the next.
            with (
                _mcp_log(trace_dir) as mcp_log,
                MCPManager(config.enabled_servers(spec.workspace), errlog=mcp_log) as mcp,
            ):
                registry = ToolRegistry(
                    apply_grant([*select_tools(spec, config), *mcp.tools()], grant),
                    policy=Policy(auto_approve=spec.allow),
                )
                model: Any = CancellableLLM(llm, flag)
                # Always guarded: an empty final answer is refused for every agent, and the
                # spec's required tools and quote check are added where it asks for them.
                model = AnswerGuardLLM(
                    model, spec.require_before_answer, check_quotes=spec.check_quotes
                )
                agent = make_agent(
                    spec.kind,
                    model,
                    registry,
                    max_steps=spec.max_steps,
                    tracer=ProgressTracer(trace, progress),
                    context_budget=int(0.75 * spec.num_ctx),
                )
                if isinstance(agent, Agent):
                    agent.system_prompt = prompt
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
