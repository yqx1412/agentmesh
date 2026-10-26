"""The default runner: a real AgentOS agent (Ollama model, built-in and MCP tools)."""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import os
import re
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

from agentos.agent import DEFAULT_SYSTEM_PROMPT, Agent, Tracer
from agentos.bench.runner import make_agent
from agentos.builtin_tools import builtin_tools
from agentos.config import AgentOSConfig, load_config
from agentos.llm import ChatResponse, Message, OllamaLLM
from agentos.mcp_client import MCPManager
from agentos.sandbox import sandbox_tools
from agentos.tools import Policy, Tool, ToolError, ToolRegistry

from agentmesh.access import GRANT, Grant, read_write_result
from agentmesh.executor import TASK_ID, Cancelled, Progress, Runner, RunOutcome
from agentmesh.numbers import TRIVIAL, numbers
from agentmesh.spec import SANDBOX_TOOLS, AgentSpec
from agentmesh.urls import PART_FROM, normalize_url, urls_in


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
# "4k", "128K": a number of thousands, as pages write context lengths.
_THOUSANDS = re.compile(r"(?<![\w.,])(\d+)\s?[kK]\b")


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


def _page_text(content: str) -> str:
    """A fetch result's page text: fetch_url returns JSON whose "text" holds the page, and
    its escaped newlines would otherwise glue a number to the "n" of "\\n"."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(data, dict):
        return " ".join(str(data.get(k, "")) for k in ("title", "text"))
    return content


def unfetched_numbers(answer: str, messages: list[Message], fetch_tool: str) -> list[str]:
    """Numbers in ``answer`` that the user's message doesn't give and no fetched page
    behind them contains. Search snippets don't count: they are not sources.

    Which pages count: on a line that cites URLs (one of Research's claims), the fetched
    pages that line cites; elsewhere (the summary), the fetched pages the answer cites
    anywhere, or every fetched page when it cites none of them.

    Live, Research reported Qwen3-8B's head dimension as 5,120, a number its one fetched
    page (the model card, which doesn't give a head dimension) never mentions. In the
    2026-10-23b M6 rerun a claim gave Ollama's default context as 4,096, citing a page
    Research never fetched; the check passed because 4096 is also Qwen3-8B's hidden_size,
    in the config.json fetched with the card another claim cited."""
    results = _fetch_results(messages, fetch_tool)
    if not results:
        return []  # nothing fetched yet: the required-calls check asks for fetches
    ok = [(urls, text) for urls, text, good in results if good]
    # Always allowed: the task's numbers, and a failed fetch's error ("HTTP 403").
    base = _numbers_in([text for _, text, good in results if not good])
    for m in messages:
        if m.role == "user":
            base |= numbers(m.content).keys()

    def on(cited: set[str]) -> set:
        return _numbers_in([text for urls, text in ok if urls & cited])

    def cites(text: str) -> set[str]:
        return {u for u in urls_in(text) if not _is_local(u)}  # "http://localhost:11434"

    everywhere = cites(answer)
    rest = (
        on(everywhere)
        if any(urls & everywhere for urls, _ in ok)
        else _numbers_in([text for _, text in ok])
    )
    bad: dict = {}
    for line in answer.splitlines():
        cited = cites(line)
        seen = base | (on(cited) if cited else rest)
        for v, t in numbers(line).items():
            if v not in seen and v not in TRIVIAL:
                bad.setdefault(v, t)
    return list(bad.values())


def _numbers_in(pages: list[str]) -> set:
    seen: set = set()
    for p in pages:
        seen |= numbers(p).keys()
        # "4k context" states 4,096 (or 4,000) tokens.
        for m in _THOUSANDS.finditer(p):
            seen |= {Decimal(m.group(1)) * 1000, Decimal(m.group(1)) * 1024}
    return seen


def _fetch_results(messages: list[Message], fetch_tool: str) -> list[tuple[set[str], str, bool]]:
    """Each ``fetch_tool`` result as (the URLs it was fetched under, its page text, whether
    it succeeded). The URLs are the one asked for, the page's own URL after redirects
    (fetch_url's result has a ``url`` field) and, for a Hugging Face card fetched with its
    config.json, both parts' URLs."""
    asked = {
        tc.id: tc.arguments.get("url")
        for m in messages
        if m.role == "assistant"
        for tc in m.tool_calls or []
        if _matches(tc.name, fetch_tool) and isinstance(tc.arguments, dict)
    }
    out = []
    for m in messages:
        if m.role != "tool" or not _matches(m.tool_name, fetch_tool):
            continue
        if m.content.startswith("ERROR:"):
            out.append((set(), _page_text(m.content.removeprefix("ERROR:")), False))
            continue
        urls: set[str] = set()
        url = asked.get(m.tool_call_id or "")
        if isinstance(url, str) and url:
            urls |= _cited_form(url)
        try:
            data = json.loads(m.content)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            final = data.get("url")
            if isinstance(final, str) and final:
                urls |= _cited_form(final)
            # A fetched Hugging Face card carries its config.json and the other way round,
            # each under a "[..., from <url>]" heading (webtools.with_model_config).
            text = data.get("text")
            if isinstance(text, str):
                urls |= {normalize_url(u) for u in PART_FROM.findall(text)}
        out.append((urls, _page_text(m.content), True))
    return out


def fetched_urls(messages: list[Message], fetch_tool: str) -> set[str]:
    """Normalized URLs of every successful ``fetch_tool`` call (see ``_fetch_results``)."""
    return {u for urls, _, good in _fetch_results(messages, fetch_tool) if good for u in urls}


def _cited_form(url: str) -> set[str]:
    """A fetched URL as an answer's citation of it is read: the citation pattern stops at
    a comma (llm-explorer.com's ".../Qwen3-8B,6sQs..." is cited as ".../Qwen3-8B")."""
    return {normalize_url(url), *urls_in(url)}


def _is_local(url: str) -> bool:
    """A loopback or private address ("http://localhost:11434") is a setting the answer
    reports, not a source it cites; fetch_url can't reach one anyway."""
    host = urlsplit(url).hostname or ""
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def unfetched_citations(answer: str, messages: list[Message], fetch_tool: str) -> list[str]:
    """URLs ``answer`` cites that no successful ``fetch_tool`` call returned. The user
    messages' URLs (the task's own) don't count as citations; the guard passes only the
    task's message, see ``AnswerGuardLLM._history``.

    In the 2026-10-23b M6 rerun, two Research answers cited NVIDIA's model card PDF after
    its fetch had failed ("unsupported content type"), and one cited ssdnodes.com, which
    Research never fetched. Only a prompt rule asked for fetched citations."""
    given = {u for m in messages if m.role == "user" for u in urls_in(m.content)}
    fetched = fetched_urls(messages, fetch_tool)
    return [u for u in urls_in(answer) if u not in fetched and u not in given and not _is_local(u)]


@dataclasses.dataclass(frozen=True)
class SendBack:
    """Why the guard refused a final answer: which check, what it flagged, and the message
    the model gets in reply."""

    check: str  # "empty" | "required_calls" | "quotes" | "numbers" | "citations"
    flagged: list[str]
    message: str


class AnswerGuardLLM:
    """Sends an unacceptable final answer back to the model with a reminder.

    A reply with no tool calls is the agent's final answer. It's refused when it's empty,
    while the history lacks the spec's required tool results, or (with ``check_quotes``) while
    it quotes text no tool returned. The model is then asked again with one extra user
    message (not kept in
    the agent's history). After ``max_nudges`` the answer goes through anyway, so a tool that
    keeps failing can't trap the task.

    With a ``tracer``, every refusal is written to the trace as an ``answer_sent_back``
    event (the check, what it flagged, the refused reply and its tokens). The agent's own
    trace only has the reply it keeps, so without this a send-back left no record.
    """

    def __init__(
        self,
        inner: Any,
        required: dict[str, int] | None = None,
        *,
        check_quotes: bool = False,
        check_numbers: str | None = None,
        check_citations: str | None = None,
        max_nudges: int = 3,
        tracer: Tracer | None = None,
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self.required = required or {}
        self.check_quotes = check_quotes
        self.check_numbers = check_numbers
        self.check_citations = check_citations
        # Every tool call and result seen in this task, oldest first, keyed by a stable
        # per-message identity so two distinct results are never merged (one guard serves
        # one task). See _history.
        self.seen: dict[object, Message] = {}
        self.max_nudges = max_nudges
        self.nudges = 0
        self.tracer = tracer

    def _history(self, messages: list[Message]) -> list[Message]:
        """What the checks read: the system prompt, the task, and every tool call and result
        this guard has seen during the task, including ones the agent has since compacted
        away. Nothing else counts as given by the task.

        Compaction replaces old messages with one *user* message holding their clipped
        text. In a 2026-10-26 live run that summary carried Research's search results, so
        the citation check took a search hit (neuralgist.com) for one of the task's own URLs
        and the number check took its "4,096" for a number the task gave; the fetched pages
        themselves were gone. The guard's own send-back message (which names the flagged
        numbers and URLs) is not given by the task either, so a retry can't pass on it.

        Each remembered message keeps its own slot: a tool result is identified by its
        ``tool_call_id`` (an assistant call by its first call's id) when it has one, else by
        object identity. Two distinct results must never merge, or ``missing_calls``, which
        counts result *occurrences*, would under-count the required calls."""
        for m in messages:
            if m.role == "tool":
                key: object = ("tool", m.tool_call_id) if m.tool_call_id else id(m)
            elif m.role == "assistant" and m.tool_calls:
                first = m.tool_calls[0].id
                key = ("call", first) if first else id(m)
            else:
                continue
            self.seen.setdefault(key, m)
        system = [m for m in messages if m.role == "system"]
        task = next(([m] for m in messages if m.role == "user"), [])
        return [*system, *task, *self.seen.values()]

    def _problem(
        self, answer: str, messages: list[Message], done_reason: str = ""
    ) -> SendBack | None:
        if not answer.strip():
            if done_reason == "length":
                # qwen3 with thinking on can spend the whole reply limit (num_predict)
                # thinking and write nothing; in M6's a-weights it deliberated over "7.7"
                # against "7.70" until it was cut off.
                return SendBack(
                    "empty",
                    ["cut off at the reply limit"],
                    "Your reply was cut off at the length limit while you were still "
                    "thinking, so it was empty. Keep your reasoning short: call the tools "
                    "the task needs, then reply with the result.",
                )
            return SendBack(
                "empty",
                [],
                "Your reply was empty. Do the task now: call the tools it needs, then reply "
                "with the result.",
            )
        missing = missing_calls(messages, self.required)
        if missing:
            need = ", ".join(f"{n} more successful {t} call(s)" for t, n in missing.items())
            return SendBack(
                "required_calls",
                [f"{t} x{n}" for t, n in missing.items()],
                f"Not yet: before answering you need {need}. Call the tool now.",
            )
        if self.check_quotes:
            bad = invented_quotes(answer, messages)
            if bad:
                listed = "; ".join(f'"{q}"' for q in bad)
                return SendBack(
                    "quotes",
                    bad,
                    "These quotes are not in any page you fetched: "
                    f"{listed}. Quote only text copied exactly from a fetched page. If no "
                    "passage supports or contradicts a claim, its verdict is NOT FOUND. "
                    "Fetch again or read further if you need to, then rewrite the answer.",
                )
        if self.check_numbers:
            bad = unfetched_numbers(answer, messages, self.check_numbers)
            if bad:
                return SendBack(
                    "numbers",
                    bad,
                    f"These numbers in your answer are not on the fetched pages cited for "
                    f"them: {', '.join(bad)}. Report a value only when a fetched page states "
                    "it, and cite that page on the same claim. Fetch a page that does (a "
                    "model's architecture values such as its head dimension are in its "
                    "config.json, which comes first when you fetch its Hugging Face page, "
                    "https://huggingface.co/<org>/<model>), or say "
                    "plainly that you could not find the value. Then rewrite the answer.",
                )
        if self.check_citations:
            bad = unfetched_citations(answer, messages, self.check_citations)
            if bad:
                return SendBack(
                    "citations",
                    bad,
                    "You cite pages you did not fetch successfully: "
                    f"{', '.join(bad)}. Cite only pages whose fetch returned their text; a "
                    "search result or a fetch that failed is not a source. Fetch the page "
                    "now if you need it, or remove that citation and anything only it "
                    "supports. Then rewrite the answer.",
                )
        return None

    def _reminder(self, messages: list[Message]) -> list[Message]:
        """While required calls are missing, a short note after the history saying what is
        still needed, so the model hears it before answering instead of after.

        Sending a too-early answer back costs a whole extra model call. With #25's logging,
        4 of 9 fleet runs had one: Research answered after one fetch (or none) and the
        Analyst did the arithmetic in its head. The note is not kept in the history and
        sits at the end, so the prompt prefix the model has already read stays the same.
        It only states what is missing. A first wording ending "Until then, reply with a
        tool call" made the Verifier fetch its three pages one per step; a second ("several
        can go in one reply") made Research batch its searches and stop at the minimum."""
        missing = missing_calls(messages, self.required)
        if not missing:
            return []
        need = ", ".join(f"{n} more successful {t} call(s)" for t, n in missing.items())
        return [
            Message(
                role="user",
                content=f"(Status: before your final answer you need at least {need}.)",
            )
        ]

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> ChatResponse:
        history = self._history(messages)
        resp = self.inner.chat([*messages, *self._reminder(history)], tools)
        extra: list[Message] = []
        p_tok = c_tok = 0  # tokens of the replies sent back
        while not resp.message.tool_calls and self.nudges < self.max_nudges:
            # A reply with tool calls ends the loop, so no tool result arrives in here: the
            # history _history built is the same for every retry.
            done_reason = getattr(resp, "done_reason", "")
            problem = self._problem(resp.message.content, history, done_reason)
            if problem is None:
                break
            self.nudges += 1
            if self.tracer is not None:
                self.tracer.emit(
                    "answer_sent_back",
                    check=problem.check,
                    flagged=problem.flagged,
                    nudge=self.nudges,
                    content=resp.message.content,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    done_reason=done_reason,
                )
            p_tok += resp.prompt_tokens
            c_tok += resp.completion_tokens
            extra += [resp.message, Message(role="user", content=problem.message)]
            resp = self.inner.chat([*messages, *extra], tools)
        if p_tok or c_tok:
            # M6: a refused reply still cost tokens; report them with the one the agent keeps.
            resp = resp.model_copy(
                update={
                    "prompt_tokens": resp.prompt_tokens + p_tok,
                    "completion_tokens": resp.completion_tokens + c_tok,
                }
            )
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
            result = inner(args)
        except Exception:
            grant.refund(url)  # a failed write doesn't use up the claim
            raise
        rec = read_write_result(args, result)
        grant.record(rec)
        if rec.new is False:
            # Say it in words: small models read {"new": false} as "stored".
            note = (
                f"ALREADY STORED: this fact was already in the knowledge base (id {rec.id}); "
                "nothing new was added. Report it as a duplicate, not as stored.\n"
            )
            return note + result if isinstance(result, str) else {"note": note, **result}
        return result

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
                tracer = ProgressTracer(trace, progress)
                # Always guarded: an empty final answer is refused for every agent, and the
                # spec's required tools and quote check are added where it asks for them.
                model = AnswerGuardLLM(
                    model,
                    spec.require_before_answer,
                    check_quotes=spec.check_quotes,
                    check_numbers=spec.check_numbers,
                    check_citations=spec.check_citations,
                    tracer=tracer,
                )
                agent = make_agent(
                    spec.kind,
                    model,
                    registry,
                    max_steps=spec.max_steps,
                    tracer=tracer,
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
