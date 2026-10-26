"""M6: run one task set on a single agent and on the whole fleet, and compare them.

Both systems get the same prompt over A2A:

- **single**: one AgentOS agent (``agents/solo.toml``) with the union of the specialists'
  tools, asked directly with the ``user`` login;
- **multi**: the M3 orchestrator with the ``orchestrator`` login, planning across the four
  specialists' Agent Cards (with the M4 retry/fallback defaults).

A run is scored only on what the user would see, the final answer, by deterministic checks
(``answer_contains``, ``answer_number``, ``answer_regex``, ``answer_timestamp``). Cost and
behaviour come from the agents' own traces, which are joined to the run by A2A task id:
tokens (every model call, including replies the answer guard sent back), tool calls, and
the pages actually fetched. For tasks marked ``cites``, every URL the answer cites is
compared with those pages, so "cites a page nobody opened" is measured, not assumed.

Results stream to ``results.jsonl`` (one line per system, task and repeat); re-running into
the same folder skips what is already there. ``report`` turns that file into Markdown.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import yaml
from a2a.types import AgentCard
from agentos.bench.checks import (
    AnswerContains,
    AnswerNumber,
    CheckResult,
    Outcome,
    _Check,
    _clip,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentmesh.access import read_write_result
from agentmesh.client import TaskRun, ask, cancel, task_state
from agentmesh.orchestrator import Chat, Orchestration, RunPolicy, orchestrate
from agentmesh.urls import PART_FROM, normalize_url, urls_in

SYSTEMS = ("single", "multi")

# --- checks ----------------------------------------------------------------------------


class AnswerRegex(_Check):
    type: Literal["answer_regex"]
    pattern: str
    flags: Literal["", "i", "m", "im"] = "im"

    def evaluate(self, o: Outcome) -> CheckResult:
        fl = (re.I if "i" in self.flags else 0) | (re.M if "m" in self.flags else 0)
        if re.search(self.pattern, o.answer, fl):
            return CheckResult(True, f"answer matches /{self.pattern}/")
        return CheckResult(False, f"answer !~ /{self.pattern}/; got {_clip(o.answer)!r}")


_TIMESTAMP = re.compile(r"(?<![\d:])(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?![\d:])")


def timestamps_in(text: str) -> list[int]:
    """Every ``mm:ss`` or ``h:mm:ss`` in a text, in seconds."""
    out = []
    for h, m, s in _TIMESTAMP.findall(text):
        if int(s) < 60:
            out.append(int(h or 0) * 3600 + int(m) * 60 + int(s))
    return out


def _seconds(stamp: str) -> int:
    found = timestamps_in(stamp)
    if len(found) != 1:
        raise ValueError(f"not a timestamp: {stamp!r}")
    return found[0]


class AnswerTimestamp(_Check):
    """Some ``mm:ss`` in the answer falls between ``start`` and ``end`` (inclusive): the
    answer cites the right part of the lecture."""

    type: Literal["answer_timestamp"]
    start: str
    end: str

    def evaluate(self, o: Outcome) -> CheckResult:
        lo, hi = _seconds(self.start), _seconds(self.end)
        found = timestamps_in(o.answer)
        if any(lo <= t <= hi for t in found):
            return CheckResult(True, f"answer cites a time in {self.start}-{self.end}")
        return CheckResult(
            False, f"no timestamp in {self.start}-{self.end}; found {len(found)} timestamp(s)"
        )


@dataclass
class RunView(Outcome):
    """An ``Outcome`` that also carries the run record, for checks that look at what the
    run did (its writes, its remote tasks) rather than at the answer."""

    run: Any = None


class Stored(_Check):
    """A knowledge write whose statement matches ``pattern`` went through (stored as new,
    or found already stored) -- or, with ``expect: false``, none did. Read from the writing
    agent's trace (its add_fact results), not from what the answer says about it."""

    type: Literal["stored"]
    pattern: str
    expect: bool = True

    def evaluate(self, o: Outcome) -> CheckResult:
        writes = getattr(getattr(o, "run", None), "writes", None) or []
        hit = [w for w in writes if re.search(self.pattern, w["statement"], re.I)]
        through = [w for w in hit if w["ok"]]
        refused = len(hit) - len(through)
        if self.expect:
            if through:
                w = through[0]
                how = "stored as new" if w.get("new") else "already stored"
                return CheckResult(True, f"/{self.pattern}/ {how} (id {w.get('id')})")
            return CheckResult(
                False,
                f"no write of /{self.pattern}/ went through ({refused} refused, "
                f"{len(writes)} write(s) in all)",
            )
        if through:
            stored = _clip(through[0]["statement"])
            return CheckResult(False, f"/{self.pattern}/ was stored: {stored!r}")
        return CheckResult(True, f"/{self.pattern}/ not stored ({refused} attempt(s) refused)")


class RemoteStopped(_Check):
    """The eval cancelled the run, and every A2A task the run had started and that was
    still going is ``canceled`` when asked afterwards; no agent made more than
    ``max_calls_after`` model calls after the cancel (an agent stops at its next model
    call, so the one in flight may still finish)."""

    type: Literal["remote_stopped"]
    min_tasks: int = Field(default=1, ge=0)
    max_calls_after: int = Field(default=1, ge=0)

    def evaluate(self, o: Outcome) -> CheckResult:
        run = getattr(o, "run", None)
        if run is None or not run.cancelled_at:
            return CheckResult(False, "the run ended before the eval could cancel it")
        tasks = run.remote
        if len(tasks) < self.min_tasks:
            return CheckResult(
                False, f"{len(tasks)} remote task(s) seen, want {self.min_tasks}; {run.message}"
            )
        going = [t for t in tasks if not t.get("finished_before_cancel")]
        bad = [t for t in going if t.get("state") != "canceled"]
        if bad:
            listed = ", ".join(f"{t['agent']} {t['state']}" for t in bad)
            return CheckResult(False, f"not canceled after the cancel: {listed}")
        if run.calls_after_cancel > self.max_calls_after:
            return CheckResult(
                False,
                f"an agent made {run.calls_after_cancel} model calls after the cancel "
                f"(max {self.max_calls_after})",
            )
        return CheckResult(
            True,
            f"{len(going)} remote task(s) canceled, {len(tasks) - len(going)} had finished; "
            f"at most {run.calls_after_cancel} model call(s) after the cancel",
        )


Check = Annotated[
    AnswerContains | AnswerNumber | AnswerRegex | AnswerTimestamp | Stored | RemoteStopped,
    Field(discriminator="type"),
]

# --- tasks -----------------------------------------------------------------------------


class EvalError(Exception):
    """The task file or the setup is unusable."""


class CancelAt(BaseModel):
    """When the eval cancels a run: ``after`` seconds, or ``at: dispatch``, the moment the
    orchestrator sends its first step and before the agent has replied with a task id (the
    window PR #19 closed; multi only). (Not ``on``: YAML reads a bare ``on`` key as true.)"""

    model_config = ConfigDict(extra="forbid")

    at: Literal["dispatch"] | None = None
    after: float | None = Field(default=None, gt=0)
    # How long to wait after the cancel before asking the agents for the tasks' states.
    grace: float = Field(default=10.0, ge=0)

    @model_validator(mode="after")
    def _one(self) -> CancelAt:
        if (self.at is None) == (self.after is None):
            raise ValueError('cancel needs exactly one of "at" and "after"')
        return self


class EvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: str
    prompt: str = Field(min_length=1)
    checks: list[Check] = Field(min_length=1)
    # The answer must cite web pages; cited URLs are compared with the fetched ones.
    cites: bool = False
    # Offer the planner the skills tagged ``write`` (orchestrate --allow-writes).
    allow_writes: bool = False
    # Run on these systems only (the single agent has no add_fact, so no write task).
    systems: list[Literal["single", "multi"]] | None = None
    cancel: CancelAt | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> EvalTask:
        if self.cancel and self.cancel.at == "dispatch" and self.systems != ["multi"]:
            raise ValueError(f'{self.id}: cancel at "dispatch" needs systems: [multi]')
        return self

    def runs_on(self, system: str) -> bool:
        return self.systems is None or system in self.systems


def load_tasks(path: Path) -> list[EvalTask]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvalError(f"task file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise EvalError(f"{path}: invalid YAML: {exc}") from exc
    items = raw.get("tasks") if isinstance(raw, dict) else None
    if not isinstance(items, list) or not items:
        raise EvalError(f'{path}: needs a non-empty "tasks" list')
    try:
        tasks = [EvalTask.model_validate(t) for t in items]
    except ValidationError as exc:
        raise EvalError(f"{path}: {exc}") from exc
    dupes = sorted(i for i, n in Counter(t.id for t in tasks).items() if n > 1)
    if dupes:
        raise EvalError(f"duplicate task ids: {dupes}")
    return tasks


def select_tasks(tasks: list[EvalTask], patterns: list[str] | None) -> list[EvalTask]:
    if not patterns:
        return tasks
    return [
        t
        for t in tasks
        if any(fnmatch.fnmatch(t.id, p) or fnmatch.fnmatch(t.category, p) for p in patterns)
    ]


def score(task: EvalTask, answer: str, run: Any = None) -> list[dict[str, Any]]:
    outcome = RunView(Path("."), answer, run=run)
    out = []
    for c in task.checks:
        r = c.evaluate(outcome)
        out.append({"type": c.type, "ok": r.ok, "detail": r.detail})
    return out


# --- traces ----------------------------------------------------------------------------


def find_traces(trace_root: Path, task_ids: list[str]) -> list[Path]:
    """Each agent names its trace ``<time>-<A2A task id>.jsonl`` in its own folder."""
    found = []
    for tid in dict.fromkeys(t for t in task_ids if t):
        found += sorted(trace_root.glob(f"*/*-{tid}.jsonl"))
    return found


def trace_stats(paths: list[Path]) -> dict[str, Any]:
    """Model calls, tokens, tool calls and fetched pages across AgentOS traces."""
    s: dict[str, Any] = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "tools": Counter(),
        "sent_back": Counter(),
        "fetched": [],
        "writes": [],
    }
    for p in paths:
        # The orchestrator's trace records its delegations as tool results; they are A2A
        # tasks (counted separately), not tool calls. Its model calls do count.
        delegations = p.parent.name == "orchestrator"
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("event")
            if kind == "answer_sent_back":
                # A reply an answer guard refused: a model call of its own (its tokens are
                # already folded into the llm_response of the reply the agent kept).
                s["llm_calls"] += 1
                s["sent_back"][str(ev.get("check", "?"))] += 1
            elif kind == "llm_response":
                s["llm_calls"] += 1
                s["prompt_tokens"] += int(ev.get("prompt_tokens") or 0)
                s["completion_tokens"] += int(ev.get("completion_tokens") or 0)
            elif kind == "tool_result" and not delegations:
                call = ev.get("call") or {}
                name = str(call.get("name", "?")).split("__")[-1]
                s["tool_calls"] += 1
                s["tools"][name] += 1
                if name == "add_fact":
                    s["writes"].append(_write_entry(call, ev))
                if not ev.get("ok"):
                    s["tool_errors"] += 1
                elif name == "fetch_url":
                    url = (call.get("arguments") or {}).get("url")
                    if url:
                        s["fetched"].append(normalize_url(str(url)))
                    # After a redirect the page's own URL differs from the one asked for
                    # (NVIDIA's model card PDF moves to developer.download.nvidia.com), and
                    # an answer may cite either.
                    final = _result_url(ev.get("output"))
                    if final:
                        s["fetched"].append(normalize_url(final))
                    # A Hugging Face card fetch also returns the model's config.json (and the
                    # other way round), each part under a "[..., from <url>]" heading. The
                    # citation guard counts both parts as fetched; so does the score.
                    s["fetched"] += _result_parts(ev.get("output"))
    s["tools"] = dict(s["tools"])
    s["sent_back"] = dict(s["sent_back"])
    s["fetched"] = list(dict.fromkeys(s["fetched"]))
    return s


_ALREADY = re.compile(r"^ALREADY STORED\b.*?\(id (\d+)\)")


def _write_entry(call: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
    """One add_fact call as the trace saw it: the statement, whether it went through, and
    the id / ``new`` flag the tool returned (a refusal by the write grant is ``ok: false``
    with the reason as output)."""
    args = call.get("arguments") or {}
    output = ev.get("output")
    entry: dict[str, Any] = {
        "statement": read_write_result(args, None).statement,
        "ok": bool(ev.get("ok")),
        "id": None,
        "new": None,
    }
    if not entry["ok"]:
        entry["error"] = str(output)[:300]
    elif isinstance(output, str) and (m := _ALREADY.match(output)):
        # The grant's duplicate note comes before the tool's JSON (runtime._attested).
        entry["id"], entry["new"] = int(m.group(1)), False
    else:
        rec = read_write_result(args, output)
        entry["id"], entry["new"] = rec.id, rec.new
    return entry


def calls_after(paths: list[Path], t: float) -> int:
    """The most model calls any one trace logged after time ``t`` (epoch seconds)."""
    most = 0
    for p in paths:
        n = 0
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                ev.get("event") in ("llm_response", "answer_sent_back")
                and float(ev.get("ts") or 0) > t
            ):
                n += 1
        most = max(most, n)
    return most


def _result_url(output: Any) -> str | None:
    """The ``url`` field of a ``fetch_url`` result (the URL after redirects), if any."""
    if not isinstance(output, str):
        return None
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    url = data.get("url") if isinstance(data, dict) else None
    return url if isinstance(url, str) and url else None


def _result_parts(output: Any) -> list[str]:
    """Normalized URLs of the parts a ``fetch_url`` result's text names with a
    ``[..., from <url>]`` heading (``webtools.with_model_config``)."""
    if not isinstance(output, str):
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    text = data.get("text") if isinstance(data, dict) else None
    if not isinstance(text, str):
        return []
    return [normalize_url(u) for u in PART_FROM.findall(text)]


# --- one run ---------------------------------------------------------------------------


class RunRecord(BaseModel):
    system: str
    task_id: str
    category: str
    repeat: int
    passed: bool
    failure: str = ""  # "" | no_answer | wrong | partial | timeout | error | plan_error
    state: str = ""
    message: str = ""
    checks: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    seconds: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    tools: dict[str, int] = Field(default_factory=dict)
    # answers an agent's guard sent back, by check (see runtime.AnswerGuardLLM)
    sent_back: dict[str, int] = Field(default_factory=dict)
    # multi only: the plan, and the A2A tasks the orchestrator sent (attempts included)
    plan: list[str] = Field(default_factory=list)
    agent_tasks: int = 0
    steps_failed: int = 0
    cited: list[str] = Field(default_factory=list)
    fetched: list[str] = Field(default_factory=list)
    unfetched_citations: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    traces: list[str] = Field(default_factory=list)
    started: str = ""
    # add_fact calls in the agents' traces (see ``_write_entry``)
    writes: list[dict[str, Any]] = Field(default_factory=list)
    # cancel tasks: when the eval cancelled the run (epoch s, 0 = it didn't), each remote
    # task it had started with its state afterwards, and the most model calls any of their
    # traces logged after the cancel
    cancelled_at: float = 0.0
    remote: list[dict[str, Any]] = Field(default_factory=list)
    calls_after_cancel: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.system, self.task_id, self.repeat)


def classify(checks: list[dict[str, Any]], answer: str, state: str) -> str:
    if state == "timeout":
        return "timeout"
    if not answer.strip():
        return "plan_error" if state == "plan_error" else ("error" if state else "no_answer")
    ok = sum(c["ok"] for c in checks)
    if ok == len(checks):
        return ""
    return "partial" if ok else "wrong"


def finish_record(rec: RunRecord, task: EvalTask, trace_root: Path) -> RunRecord:
    """Fill in cost, writes and citations from the traces, then score the run."""
    paths = find_traces(trace_root, rec.task_ids)
    st = trace_stats(paths)
    rec.traces = [str(p) for p in paths]
    rec.llm_calls = st["llm_calls"]
    rec.prompt_tokens, rec.completion_tokens = st["prompt_tokens"], st["completion_tokens"]
    rec.tool_calls, rec.tool_errors, rec.tools = st["tool_calls"], st["tool_errors"], st["tools"]
    rec.sent_back = st["sent_back"]
    rec.fetched = st["fetched"]
    rec.writes = st["writes"]
    if rec.cancelled_at:
        remote = find_traces(trace_root, [t["task_id"] for t in rec.remote])
        rec.calls_after_cancel = calls_after(remote, rec.cancelled_at)
    if task.cites:
        rec.cited = urls_in(rec.answer)
        rec.unfetched_citations = [u for u in rec.cited if u not in set(rec.fetched)]
    rec.checks = score(task, rec.answer, rec)
    if task.cancel is not None:
        # A cancelled run has no answer to show; only its checks count.
        rec.passed = all(c["ok"] for c in rec.checks)
        rec.failure = "" if rec.passed else classify(rec.checks, "(cancelled)", "")
        return rec
    rec.passed = bool(rec.answer.strip()) and all(c["ok"] for c in rec.checks)
    rec.failure = "" if rec.passed else classify(rec.checks, rec.answer, rec.state) or "wrong"
    return rec


async def run_single(
    task: EvalTask,
    card: AgentCard,
    http: httpx.AsyncClient,
    *,
    skill: str,
    timeout: float,
) -> RunRecord:
    rec = RunRecord(
        system="single", task_id=task.id, category=task.category, repeat=0, passed=False
    )
    seen: list[str] = []

    def on_event(event) -> None:
        if event.WhichOneof("payload") == "task" and event.task.id and not seen:
            seen.append(event.task.id)

    t0 = time.perf_counter()
    run: TaskRun | None = None
    job = asyncio.ensure_future(
        ask(card, task.prompt, http, on_event=on_event, metadata={"skill": skill})
    )
    try:
        if task.cancel is not None and task.cancel.after is not None:
            await asyncio.wait({job}, timeout=task.cancel.after)
            if not job.done():
                rec.cancelled_at = time.time()
                if seen:
                    said = await _cancel_within(card, seen[0], http)
                    rec.message = f"cancelled after {task.cancel.after:g} s: {said}"
                else:
                    rec.message = f"no task id after {task.cancel.after:g} s, nothing to cancel"
                    job.cancel()
        run = await asyncio.wait_for(job, max(1.0, timeout - (time.perf_counter() - t0)))
    except TimeoutError:
        rec.state, rec.message = "timeout", f"no answer after {timeout:g} s"
        if seen:
            rec.message += f"; remote task {await _cancel_within(card, seen[0], http)}"
    except asyncio.CancelledError:
        if not rec.cancelled_at:
            raise
        rec.state = "canceled"
    except Exception as exc:
        rec.state, rec.message = "error", f"{type(exc).__name__}: {exc}"[:500]
    rec.seconds = round(time.perf_counter() - t0, 1)
    if run is not None:
        note = rec.message
        rec.state, rec.message, rec.answer = run.state, run.final_message, run.answer
        if rec.cancelled_at:
            rec.message = f"{note}; {run.final_message}".strip("; ")
        seen = [run.task_id or (seen[0] if seen else "")]
    rec.task_ids = [t for t in seen if t]
    if rec.cancelled_at and rec.task_ids:
        await asyncio.sleep(task.cancel.grace if task.cancel else 0)
        tid = rec.task_ids[0]
        rec.remote = [
            {"agent": card.name, "task_id": tid, "state": await task_state(card, tid, http)}
        ]
    return rec


async def _cancel_within(card: AgentCard, task_id: str, http: httpx.AsyncClient) -> str:
    try:
        return await asyncio.wait_for(cancel(card, task_id, http), 10)
    except TimeoutError:
        return "cancel failed: no reply in 10 s"


# The orchestrator's journal lines (Recorder.state -> on_line) that a cancel task reads.
_SENDING = re.compile(r"^\[working\] (\S+): sending ")
_TOOK = re.compile(r"^\[working\] (\S+): (.+) took it as task (\S+)$")
_ENDED = re.compile(
    r"^\[working\] (\S+): .+ (?:completed|failed|timeout|unreachable|rejected|canceled) after "
)


async def _state_by_name(
    cards: list[AgentCard], agent: str, task_id: str, http: httpx.AsyncClient
) -> str:
    """Ask the agent named ``agent`` for the task's state (two cards can share a name, so
    try each until one knows the task)."""
    said = f"error: no card named {agent!r}"
    for card in (c for c in cards if c.name == agent):
        said = await task_state(card, task_id, http)
        if not said.startswith("error:"):
            break
    return said


async def run_multi(
    task: EvalTask,
    cards: list[AgentCard],
    chat: Chat,
    http: httpx.AsyncClient,
    *,
    trace_root: Path,
    policy: RunPolicy,
    timeout: float,
    model_name: str,
) -> RunRecord:
    """``task_ids`` holds the orchestration's run id (its own trace: plan and combine
    tokens) and every A2A task it sent, retries and fallbacks included. For a cancel task,
    the remote tasks come from the orchestrator's journal lines, since a cancelled
    orchestration returns nothing."""
    rec = RunRecord(system="multi", task_id=task.id, category=task.category, repeat=0, passed=False)
    when = task.cancel
    remote: dict[str, dict[str, Any]] = {}
    journal: list[str] = []
    orch: asyncio.Future | None = None

    def on_line(line: str) -> None:
        journal.append(line)
        if m := _TOOK.match(line):
            sid, agent, tid = m.groups()
            remote[tid] = {"agent": agent, "task_id": tid, "step": sid}
        elif (m := _ENDED.match(line)) and not rec.cancelled_at:
            for t in remote.values():
                if t["step"] == m.group(1):
                    t["finished_before_cancel"] = True
        if (
            when is not None
            and when.at == "dispatch"
            and not rec.cancelled_at
            and _SENDING.match(line)
            and orch is not None
        ):
            # Called inside the orchestration: it stops at its next await, which is the
            # wait for the A2A call it has just started, before any task id has come back.
            rec.cancelled_at = time.time()
            orch.cancel()

    t0 = time.perf_counter()
    out: Orchestration | None = None
    orch = asyncio.ensure_future(
        orchestrate(
            task.prompt,
            cards,
            chat,
            http,
            trace_root=trace_root,
            policy=policy,
            model_name=model_name,
            on_line=on_line,
            allow_writes=task.allow_writes,
        )
    )
    try:
        if when is not None and when.after is not None:
            await asyncio.wait({orch}, timeout=when.after)
            if not orch.done():
                rec.cancelled_at = time.time()
                orch.cancel()
        await asyncio.wait({orch}, timeout=max(1.0, timeout - (time.perf_counter() - t0)))
    except asyncio.CancelledError:  # the eval itself was stopped: stop the run too
        orch.cancel()
        await asyncio.shield(asyncio.wait({orch}))
        raise
    if not orch.done():
        # orchestrate() cancels every remote task before re-raising.
        orch.cancel()
        await asyncio.wait({orch})
        rec.state, rec.message = "timeout", f"no answer after {timeout:g} s"
    elif orch.cancelled():
        rec.state = "canceled"
        said = [ln.removeprefix("[working] ") for ln in journal if "cancel" in ln.lower()]
        rec.message = "; ".join(said)[:500] or "cancelled"
    elif (exc := orch.exception()) is not None:
        rec.state, rec.message = "error", f"{type(exc).__name__}: {exc}"[:500]
    else:
        out = orch.result()
    rec.seconds = round(time.perf_counter() - t0, 1)
    if rec.cancelled_at:
        rec.task_ids = [*remote]
        if remote:
            await asyncio.sleep(when.grace if when else 0)
        for t in remote.values():
            t.setdefault("finished_before_cancel", False)
            t["state"] = await _state_by_name(cards, t["agent"], t["task_id"], http)
        rec.remote = list(remote.values())
    if out is None:
        return rec
    rec.answer, rec.message = out.answer, out.message
    rec.state = "plan_error" if not out.plan else out.state
    rec.plan = [s.skill for s in out.plan]
    attempts = [a for r in out.results.values() for a in r.attempts]
    rec.agent_tasks = len(attempts)
    rec.task_ids = [out.run_id, *(a.task_id for a in attempts if a.task_id)]
    rec.steps_failed = sum(not r.ok for r in out.results.values())
    return rec


# --- the whole evaluation ----------------------------------------------------------------


def _git_sha(cwd: Path | None = None) -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (f"{sha}-dirty" if dirty else sha) or None


def load_results(path: Path) -> list[RunRecord]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(RunRecord.model_validate_json(line))
    return out


def schedule(
    tasks: list[EvalTask], systems: list[str], repeats: int
) -> list[tuple[EvalTask, str, int]]:
    """Task by task, both systems back to back, alternating which goes first, so neither
    system always meets a cold or a warm model."""
    order = []
    for r in range(repeats):
        for i, t in enumerate(tasks):
            pair = systems if (i + r) % 2 == 0 else list(reversed(systems))
            order += [(t, s, r) for s in pair if t.runs_on(s)]
    return order


RunOne = Callable[[EvalTask, str], Awaitable[RunRecord]]


async def run_eval(
    tasks: list[EvalTask],
    systems: list[str],
    *,
    repeats: int,
    out_dir: Path,
    run_one: RunOne,
    trace_root: Path,
    meta: dict[str, Any] | None = None,
    progress: Callable[[RunRecord, int, int], None] | None = None,
) -> list[RunRecord]:
    """Run the schedule, skipping (system, task, repeat) already in ``results.jsonl``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sink = out_dir / "results.jsonl"
    done = {r.key for r in load_results(sink)}
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "started": datetime.now().isoformat(timespec="seconds"),
                    "git_sha": _git_sha(),
                    "systems": systems,
                    "tasks": [t.id for t in tasks],
                    "repeats": repeats,
                    **(meta or {}),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    plan = schedule(tasks, systems, repeats)
    new: list[RunRecord] = []
    for n, (task, system, r) in enumerate(plan, 1):
        if (system, task.id, r) in done:
            continue
        started = datetime.now().isoformat(timespec="seconds")
        rec = await run_one(task, system)
        rec.repeat, rec.started = r, started
        finish_record(rec, task, trace_root)
        with sink.open("a", encoding="utf-8") as f:
            f.write(rec.model_dump_json() + "\n")
        new.append(rec)
        if progress:
            progress(rec, n, len(plan))
    return new


# --- the report --------------------------------------------------------------------------


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def report(results: list[RunRecord], tasks: list[EvalTask] | None = None) -> str:
    """Markdown tables: overall, per category, per task, and every failure."""
    if not results:
        return "No results.\n"
    systems = [s for s in SYSTEMS if any(r.system == s for r in results)]
    by: dict[str, list[RunRecord]] = defaultdict(list)
    for r in results:
        by[r.system].append(r)
    order = [t.id for t in tasks] if tasks else list(dict.fromkeys(r.task_id for r in results))
    cats = list(dict.fromkeys(r.category for r in results))
    lines = ["## Overall", ""]
    lines += [
        "| | " + " | ".join(systems) + " |",
        "|---|" + "---|" * len(systems),
    ]

    def row(label: str, fn: Callable[[list[RunRecord]], str]) -> None:
        lines.append(f"| {label} | " + " | ".join(fn(by[s]) for s in systems) + " |")

    row("Runs passed", lambda rs: _pct(sum(r.passed for r in rs), len(rs)))
    row(
        "Passed, every cited page fetched",
        lambda rs: _pct(sum(r.passed and not r.unfetched_citations for r in rs), len(rs)),
    )
    row(
        "Checks passed",
        lambda rs: _pct(sum(c["ok"] for r in rs for c in r.checks), sum(len(r.checks) for r in rs)),
    )
    row("Median time (s)", lambda rs: f"{_median([r.seconds for r in rs]):.0f}")
    row("Mean time (s)", lambda rs: f"{_mean([r.seconds for r in rs]):.0f}")
    row("Mean tokens", lambda rs: f"{_mean([r.tokens for r in rs]):,.0f}")
    row("Mean model calls", lambda rs: f"{_mean([r.llm_calls for r in rs]):.1f}")
    row("Mean tool calls", lambda rs: f"{_mean([r.tool_calls for r in rs]):.1f}")
    row("Mean A2A tasks", lambda rs: f"{_mean([r.agent_tasks or 1 for r in rs]):.1f}")

    def cites(rs: list[RunRecord]) -> str:
        cited = [r for r in rs if r.cited]
        bad = [r for r in cited if r.unfetched_citations]
        return f"{len(bad)} of {len(cited)}"

    row("Runs citing an unfetched page", cites)

    def sent_back(rs: list[RunRecord]) -> str:
        by_check: Counter[str] = Counter()
        for r in rs:
            by_check.update(r.sent_back)
        runs = sum(bool(r.sent_back) for r in rs)
        if not runs:
            return "none"
        detail = ", ".join(f"{k} {v}" for k, v in sorted(by_check.items()))
        return f"{runs} runs ({detail})"

    row("Answers sent back by a check", sent_back)
    row(
        "Failures",
        lambda rs: (
            ", ".join(
                f"{k} {v}" for k, v in sorted(Counter(r.failure for r in rs if r.failure).items())
            )
            or "none"
        ),
    )

    lines += ["", "## By category", ""]
    lines += [
        "| Category | "
        + " | ".join(f"{s} passed | {s} median s | {s} mean tokens" for s in systems)
        + " |",
        "|---|" + "---|---|---|" * len(systems),
    ]
    for c in cats:
        cells = []
        for s in systems:
            rs = [r for r in by[s] if r.category == c]
            if not rs:
                cells += ["-", "-", "-"]
                continue
            cells += [
                _pct(sum(r.passed for r in rs), len(rs)),
                f"{_median([r.seconds for r in rs]):.0f}",
                f"{_mean([r.tokens for r in rs]):,.0f}",
            ]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += ["", "## By task", ""]
    lines += [
        "| Task | " + " | ".join(f"{s} passed | {s} median s" for s in systems) + " | multi plan |",
        "|---|" + "---|---|" * len(systems) + "---|",
    ]
    for tid in order:
        cells = []
        for s in systems:
            rs = [r for r in by[s] if r.task_id == tid]
            if not rs:
                cells += ["-", "-"]
                continue
            cells += [
                f"{sum(r.passed for r in rs)}/{len(rs)}",
                f"{_median([r.seconds for r in rs]):.0f}",
            ]
        plans = Counter(
            " > ".join(r.plan) for r in by.get("multi", []) if r.task_id == tid and r.plan
        )
        plan = "; ".join(f"{p} (x{n})" if n > 1 else p for p, n in plans.most_common(2))
        lines.append(f"| {tid} | " + " | ".join(cells) + f" | {plan} |")

    if "single" in by and "multi" in by:
        better = worse = same = 0
        for tid in order:
            if not all(any(r.task_id == tid for r in by[s]) for s in ("single", "multi")):
                continue
            a = sum(r.passed for r in by["single"] if r.task_id == tid)
            b = sum(r.passed for r in by["multi"] if r.task_id == tid)
            better += b > a
            worse += b < a
            same += a == b
        lines += [
            "",
            f"Per task, multi passed more often on {better}, less often on {worse} and "
            f"equally on {same}.",
        ]

    fails = [r for r in results if not r.passed or r.unfetched_citations]
    if fails:
        lines += ["", "## Failures and citation problems", ""]
        lines += ["| System | Task | Rep | Failure | Detail |", "|---|---|---|---|---|"]
        for r in sorted(
            fails,
            key=lambda r: (order.index(r.task_id) if r.task_id in order else 0, r.system, r.repeat),
        ):
            bad = [c["detail"] for c in r.checks if not c["ok"]]
            detail = "; ".join(bad) or r.message
            if r.unfetched_citations:
                detail += f"; cites unfetched {', '.join(r.unfetched_citations[:3])}"
            if r.steps_failed:
                detail += f"; {r.steps_failed} step(s) failed"
            detail = detail.replace("|", "\\|").replace("\n", " ")[:300]
            lines.append(
                f"| {r.system} | {r.task_id} | {r.repeat} | {r.failure or 'cites'} | {detail} |"
            )
    return "\n".join(lines) + "\n"
