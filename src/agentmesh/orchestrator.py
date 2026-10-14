"""M3: the orchestrator. It splits a task across agents it finds by their Agent Cards.

1. **Discover:** read each agent's card. Every skill on every card goes into one catalogue.
2. **Plan:** a local model writes a JSON plan. Each step names one catalogue skill, the full
   message that agent will receive, and the steps it depends on. The ``skill`` field is
   constrained to the catalogue's ids, so agents are chosen by what their cards say, never by
   name in code. A plan that still fails validation is sent back with the error.
3. **Run:** each step goes to the agent whose card lists its skill, as an ordinary A2A task.
   ``{{s1}}`` in a step's input is replaced by step s1's answer. Steps whose inputs are
   ready run in parallel; a step whose dependency failed is skipped.
4. **Combine:** the model writes the final reply from the step results only.

Every delegated message carries ``{"orchestration": <run id>, "step": "s1"}`` metadata, which
the agents record in their A2A journals. The orchestrator writes its own journal and trace
in the same formats as the agents (``<trace root>/orchestrator/``), so ``agentmesh watch``
shows it as one more lane and links it to the tasks it started.

Not here yet (M4): retries, fallback to another agent with the same skill, and cancelling a
remote task when its step times out (the remote agent keeps working).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from a2a.types import AgentCard, StreamResponse

from agentmesh.client import ask, discover
from agentmesh.journal import JOURNAL_NAME, JsonlJournal

MAX_STEPS = 6
PLAN_ATTEMPTS = 3
# Longest step result passed to the combining model.
MAX_RESULT = 8000

_STEP_ID = re.compile(r"^s[0-9]{1,2}$")
_REF = re.compile(r"\{\{\s*(s[0-9]{1,2})\s*\}\}")

PLAN_PROMPT = """\
You are the orchestrator of a team of AI agents. You never do the work yourself: you split \
the user's task into steps and give each step to one skill from the catalogue.

Rules:
- Use only skills from the catalogue, by id. Choose each one by what its description says \
it does, and respect what each agent's description says it cannot do.
- A step's "input" is the entire message that agent receives. The agent sees nothing else: \
not the user's task and not the other steps. Make it complete and self-contained, and say \
what the agent should return.
- To give a step an earlier step's result, write {{<step id>}} in its input, for example \
"Check each of these claims against its cited source: {{s1}}". It is replaced by that \
step's full answer, and the step then runs after it.
- Steps that don't depend on each other run at the same time.
- Use as few steps as the task needs, at most 6. Include every skill the task actually \
needs and no others.
- Anything written to a knowledge base must first go through a verification skill, and \
only the claims it marks SUPPORTED may be stored, each with its source.

Reply with JSON only: {"steps": [{"id": "s1", "skill": "<skill id>", "input": "...", \
"depends_on": []}, ...]}, with ids s1, s2, ... in the order the steps should start."""

COMBINE_PROMPT = """\
You write the final reply to a user's task from the results of a team of AI agents.
- Use only what the results say. Do not add facts of your own.
- Lead with the answer to the task. Keep sources (URLs, lecture timestamps) next to the \
claims they support.
- Say which steps failed or were skipped, and what that leaves undone.
- Say that something was stored in the knowledge base only if a result says it was \
stored, and list what was stored."""


# --- the model -------------------------------------------------------------------------


@dataclass
class ChatReply:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


# (messages, JSON schema or None) -> reply
Chat = Callable[[list[dict[str, str]], dict[str, Any] | None], Awaitable[ChatReply]]


class OllamaChat:
    """One Ollama ``/api/chat`` call per request; ``schema`` turns on constrained JSON."""

    def __init__(
        self,
        model: str,
        url: str = "http://127.0.0.1:11434",
        *,
        think: bool = False,
        num_ctx: int = 16384,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.think = think
        self.num_ctx = num_ctx
        self.timeout = timeout

    async def __call__(
        self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None
    ) -> ChatReply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {"num_ctx": self.num_ctx, "temperature": 0},
        }
        if schema is not None:
            body["format"] = schema
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            r = await http.post(f"{self.url}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
        return ChatReply(
            content=data["message"]["content"],
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )


# --- the catalogue ---------------------------------------------------------------------


@dataclass
class Skill:
    id: str
    name: str
    description: str
    agent: str
    card: AgentCard
    examples: list[str] = field(default_factory=list)


Catalogue = dict[str, list[Skill]]


async def discover_all(
    urls: list[str], http: httpx.AsyncClient
) -> tuple[list[AgentCard], list[str]]:
    """Every card that answers, plus one error line per agent that doesn't."""

    async def one(url: str) -> AgentCard | str:
        try:
            return await discover(url, http)
        except Exception as exc:
            return f"{url}: {type(exc).__name__}: {exc}"

    found = await asyncio.gather(*(one(u) for u in urls))
    cards = [c for c in found if isinstance(c, AgentCard)]
    errors = [c for c in found if isinstance(c, str)]
    return cards, errors


def build_catalogue(cards: list[AgentCard]) -> Catalogue:
    """Skill id -> the skills with that id, in card order (the first one is used)."""
    cat: Catalogue = {}
    for card in cards:
        for s in card.skills:
            cat.setdefault(s.id, []).append(
                Skill(s.id, s.name, s.description, card.name, card, list(s.examples))
            )
    return cat


def describe_catalogue(cards: list[AgentCard]) -> str:
    """The catalogue as the planner reads it: each agent's description, then its skills."""
    lines = []
    for card in cards:
        lines.append(f"Agent: {card.name}. {card.description}")
        for s in card.skills:
            ex = f' Example: "{s.examples[0]}"' if s.examples else ""
            lines.append(f"  - skill `{s.id}`: {s.description}{ex}")
    return "\n".join(lines)


def agent_url(card: AgentCard) -> str:
    return card.supported_interfaces[0].url if card.supported_interfaces else ""


# --- the plan --------------------------------------------------------------------------


class PlanError(ValueError):
    """The model's plan can't be run; the message says why, for the model to fix it."""


@dataclass
class Step:
    id: str
    skill: str
    input: str
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "skill": self.skill,
            "input": self.input,
            "depends_on": self.depends_on,
        }


def plan_schema(skill_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_STEPS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "skill": {"type": "string", "enum": skill_ids},
                        "input": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "skill", "input", "depends_on"],
                },
            }
        },
        "required": ["steps"],
    }


def parse_plan(content: str, catalogue: Catalogue) -> list[Step]:
    """Check a plan completely: ids, skills, inputs, dependencies, no cycles."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlanError(f"the reply is not valid JSON ({exc})") from exc
    raw = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise PlanError('the plan needs a non-empty "steps" list')
    if len(raw) > MAX_STEPS:
        raise PlanError(f"the plan has {len(raw)} steps; the limit is {MAX_STEPS}")

    steps: list[Step] = []
    for n, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise PlanError(f"step {n} is not an object")
        sid = str(item.get("id", "")).strip()
        if not _STEP_ID.match(sid):
            raise PlanError(f"step {n}: id must be s1, s2, ...; got {sid!r}")
        if any(s.id == sid for s in steps):
            raise PlanError(f"two steps have the id {sid}")
        skill = str(item.get("skill", "")).strip()
        if skill not in catalogue:
            known = ", ".join(sorted(catalogue))
            raise PlanError(f"step {sid}: unknown skill {skill!r}; the skills are: {known}")
        text = str(item.get("input", "")).strip()
        if not text:
            raise PlanError(f"step {sid} has an empty input")
        deps = item.get("depends_on") or []
        if not isinstance(deps, list):
            raise PlanError(f"step {sid}: depends_on must be a list of step ids")
        merged = list(dict.fromkeys([str(d).strip() for d in deps] + _REF.findall(text)))
        steps.append(Step(sid, skill, text, merged))

    ids = {s.id for s in steps}
    for s in steps:
        for d in s.depends_on:
            if d == s.id:
                raise PlanError(f"step {s.id} depends on itself")
            if d not in ids:
                raise PlanError(f"step {s.id} uses {d}, which is not a step in the plan")
    _check_acyclic(steps)
    return steps


def _check_acyclic(steps: list[Step]) -> None:
    remaining = {s.id: set(s.depends_on) for s in steps}
    while remaining:
        ready = [sid for sid, deps in remaining.items() if not deps]
        if not ready:
            raise PlanError(f"the steps {sorted(remaining)} depend on each other in a cycle")
        for sid in ready:
            del remaining[sid]
        for deps in remaining.values():
            deps.difference_update(ready)


async def make_plan(
    task: str, cards: list[AgentCard], catalogue: Catalogue, chat: Chat
) -> tuple[list[Step], list[dict[str, Any]]]:
    """A valid plan plus one record per model attempt. Raises PlanError after the last try."""
    messages = [
        {"role": "system", "content": PLAN_PROMPT},
        {
            "role": "user",
            "content": f"Catalogue:\n{describe_catalogue(cards)}\n\nUser's task:\n{task}",
        },
    ]
    schema = plan_schema(sorted(catalogue))
    attempts: list[dict[str, Any]] = []
    for _ in range(PLAN_ATTEMPTS):
        reply = await chat(messages, schema)
        record = {
            "content": reply.content,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
        }
        attempts.append(record)
        try:
            return parse_plan(reply.content, catalogue), attempts
        except PlanError as exc:
            record["error"] = str(exc)
            messages += [
                {"role": "assistant", "content": reply.content},
                {
                    "role": "user",
                    "content": f"That plan can't run: {exc}. Reply with a corrected plan.",
                },
            ]
    raise PlanError(f"no valid plan after {PLAN_ATTEMPTS} attempts: {attempts[-1]['error']}")


# --- recording -------------------------------------------------------------------------


class Recorder:
    """Writes the orchestration's A2A-style journal and AgentOS-style trace (or nothing).

    The trace uses the AgentOS event names ``agentmesh watch`` already renders: the plan is a
    model step whose "tool calls" are the delegations, and each step's result is a tool result.
    """

    AGENT = "Orchestrator"

    def __init__(
        self, folder: Path | None, run_id: str, on_line: Callable[[str], None] | None = None
    ) -> None:
        self.run_id = run_id
        self.on_line = on_line
        self.journal = JsonlJournal(folder / JOURNAL_NAME, self.AGENT) if folder else None
        self.trace: Path | None = None
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            self.trace = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id}.jsonl"

    def state(self, state: str, **data: Any) -> None:
        if self.journal is not None:
            self.journal({"task": self.run_id, "context": self.run_id, "state": state, **data})
        if self.on_line is not None:
            self.on_line(f"[{state}] {data.get('message', '')}".rstrip())

    def event(self, event: str, **data: Any) -> None:
        if self.trace is None:
            return
        line = json.dumps(
            {"ts": round(time.time(), 3), "event": event, **data}, ensure_ascii=False, default=str
        )
        with self.trace.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# --- running ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step: Step
    agent: str = ""
    url: str = ""
    input: str = ""
    task_id: str = ""
    state: str = "pending"  # pending | working | completed | failed | skipped | ...
    answer: str = ""
    message: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    started: float | None = None
    ended: float | None = None

    @property
    def ok(self) -> bool:
        return self.state == "completed"

    @property
    def label(self) -> str:
        return f"{self.step.id} {self.step.skill} -> {self.agent or '?'}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.step.to_dict(),
            "agent": self.agent,
            "url": self.url,
            "sent": self.input,
            "task_id": self.task_id,
            "state": self.state,
            "answer": self.answer,
            "message": self.message,
            "stats": self.stats,
            "seconds": round(self.ended - self.started, 1) if self.started and self.ended else None,
        }


def fill(text: str, results: dict[str, StepResult]) -> str:
    """Replace every ``{{sN}}`` with step sN's answer."""
    return _REF.sub(lambda m: results[m.group(1)].answer.strip(), text)


def step_input(step: Step, results: dict[str, StepResult]) -> str:
    """The message a step's agent receives.

    The planner often lists a dependency in ``depends_on`` but never writes ``{{sN}}`` into
    the input ("Verify the claims from the research"), which would send the agent a request
    about data it never sees. Any dependency the input doesn't reference is appended.
    """
    text = fill(step.input, results)
    used = set(_REF.findall(step.input))
    for dep in step.depends_on:
        if dep not in used:
            text += f"\n\nResult of step {dep}:\n{results[dep].answer.strip()}"
    return text


async def _run_one(
    step: Step,
    results: dict[str, StepResult],
    catalogue: Catalogue,
    http: httpx.AsyncClient,
    rec: Recorder,
    timeout: float,
) -> None:
    """Runs one step to a terminal state. Never raises: failures go into the result."""
    r = results[step.id]
    skill = catalogue[step.skill][0]
    r.agent, r.url = skill.agent, agent_url(skill.card)
    r.input = step_input(step, results)
    r.state, r.started = "working", time.time()
    rec.state("working", message=f"{step.id}: sending {step.skill} to {r.agent}")

    def seen(event: StreamResponse) -> None:
        if event.WhichOneof("payload") == "task" and not r.task_id:
            r.task_id = event.task.id
            rec.state("working", message=f"{step.id}: {r.agent} took it as task {r.task_id}")

    try:
        run = await asyncio.wait_for(
            ask(
                skill.card,
                r.input,
                http,
                on_event=seen,
                metadata={"orchestration": rec.run_id, "step": step.id},
            ),
            timeout,
        )
        r.task_id = run.task_id or r.task_id
        r.state, r.answer, r.message, r.stats = (
            run.state or "failed",
            run.answer,
            run.final_message,
            run.stats,
        )
        if r.ok and not r.answer.strip():
            r.state, r.message = "failed", "the agent completed with an empty answer"
    except TimeoutError:
        r.state, r.message = "failed", f"no result after {timeout:g} s"
    except Exception as exc:
        r.state, r.message = "failed", f"{type(exc).__name__}: {exc}"
    r.ended = time.time()
    took = f"{r.ended - r.started:.0f} s"
    detail = "" if r.ok else f": {r.message[:300]}"
    rec.state("working", message=f"{step.id}: {r.agent} {r.state} after {took}{detail}")
    rec.event(
        "tool_result",
        step=1,
        call={"name": r.label, "arguments": {"input": r.input, "task": r.task_id}},
        ok=r.ok,
        output=r.answer if r.ok else f"{r.state}: {r.message}",
    )


async def run_steps(
    plan: list[Step],
    catalogue: Catalogue,
    http: httpx.AsyncClient,
    rec: Recorder,
    step_timeout: float,
) -> dict[str, StepResult]:
    results = {s.id: StepResult(s) for s in plan}
    waiting = {s.id: s for s in plan}
    running: set[asyncio.Task[None]] = set()
    while waiting or running:
        changed = True
        while changed:  # a skip can make further steps skippable
            changed = False
            for sid, s in list(waiting.items()):
                deps = [results[d] for d in s.depends_on]
                bad = [d for d in deps if d.state not in ("pending", "working", "completed")]
                if bad:
                    del waiting[sid]
                    r = results[sid]
                    r.state = "skipped"
                    r.message = "needs " + ", ".join(f"{d.step.id} ({d.state})" for d in bad)
                    rec.state("working", message=f"{sid}: skipped, {r.message}")
                    rec.event(
                        "tool_result",
                        step=1,
                        call={"name": f"{sid} {s.skill}", "arguments": {"input": s.input}},
                        ok=False,
                        output=f"skipped: {r.message}",
                    )
                    changed = True
                elif all(d.ok for d in deps):
                    del waiting[sid]
                    running.add(
                        asyncio.ensure_future(
                            _run_one(s, results, catalogue, http, rec, step_timeout)
                        )
                    )
        if not running:
            break
        done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            t.result()  # _run_one doesn't raise; surface a bug if it does
    return results


async def combine(
    task: str, plan: list[Step], results: dict[str, StepResult], chat: Chat
) -> ChatReply:
    parts = []
    for s in plan:
        r = results[s.id]
        body = r.answer if r.ok else f"({r.state}) {r.message}"
        if len(body) > MAX_RESULT:
            body = body[:MAX_RESULT] + " [...]"
        parts.append(
            f"### {s.id}: {s.skill} by {r.agent or 'no agent'} - {r.state}\n"
            f"Asked: {r.input or s.input}\n\nResult:\n{body}"
        )
    messages = [
        {"role": "system", "content": COMBINE_PROMPT},
        {
            "role": "user",
            "content": f"User's task:\n{task}\n\nStep results, in plan order:\n\n"
            + "\n\n".join(parts),
        },
    ]
    return await chat(messages, None)


@dataclass
class Orchestration:
    run_id: str
    task: str
    state: str = "working"  # completed | failed
    plan: list[Step] = field(default_factory=list)
    results: dict[str, StepResult] = field(default_factory=dict)
    answer: str = ""
    message: str = ""
    trace: Path | None = None
    tokens: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "state": self.state,
            "message": self.message,
            "steps": [self.results[s.id].to_dict() for s in self.plan if s.id in self.results]
            or [s.to_dict() for s in self.plan],
            "answer": self.answer,
            "orchestrator_tokens": self.tokens,
            "trace": str(self.trace) if self.trace else None,
        }


async def orchestrate(
    task: str,
    cards: list[AgentCard],
    chat: Chat,
    http: httpx.AsyncClient,
    *,
    trace_root: Path | None = None,
    step_timeout: float = 900.0,
    plan_only: bool = False,
    on_line: Callable[[str], None] | None = None,
    model_name: str = "",
) -> Orchestration:
    run_id = str(uuid.uuid4())
    rec = Recorder(trace_root / "orchestrator" if trace_root else None, run_id, on_line)
    out = Orchestration(run_id, task, trace=rec.trace)
    catalogue = build_catalogue(cards)

    def finish(state: str, message: str = "", stop: str = "final_answer") -> Orchestration:
        out.state, out.message = state, message
        rec.event("run_end", stop_reason=stop, answer=out.answer)
        if state == "completed":
            rec.state("completed", answer=out.answer, stats=_stats(out))
        else:
            rec.state("failed", message=message, stats=_stats(out))
        return out

    rec.state("submitted", text=task)
    rec.event(
        "run_start",
        model=model_name,
        tools=[
            f"{sid} ({', '.join(s.agent for s in skills)})" for sid, skills in catalogue.items()
        ],
        task=task,
    )
    names = ", ".join(c.name for c in cards)
    rec.state("working", message=f"planning with {len(catalogue)} skills from {names}")
    if not catalogue:
        return finish("failed", "no agents with skills were found", stop="no_agents")

    try:
        out.plan, attempts = await make_plan(task, cards, catalogue, chat)
    except PlanError as exc:
        return finish("failed", str(exc), stop="plan_error")
    except Exception as exc:
        return finish("failed", f"planner error: {type(exc).__name__}: {exc}", stop="error")
    for a in attempts:
        out.tokens["prompt"] += a["prompt_tokens"]
        out.tokens["completion"] += a["completion_tokens"]
    rec.event(
        "llm_response",
        step=1,
        content=json.dumps({"steps": [s.to_dict() for s in out.plan]}, indent=2)
        + (
            f"\n\n({len(attempts) - 1} rejected plan(s) before this one)"
            if len(attempts) > 1
            else ""
        ),
        tool_calls=[
            {
                "name": f"{s.id} {s.skill} -> {catalogue[s.skill][0].agent}",
                "arguments": {"input": s.input, "depends_on": s.depends_on},
            }
            for s in out.plan
        ],
        prompt_tokens=sum(a["prompt_tokens"] for a in attempts),
        completion_tokens=sum(a["completion_tokens"] for a in attempts),
    )
    summary = "; ".join(
        f"{s.id} {s.skill} -> {catalogue[s.skill][0].agent}"
        + (f" after {', '.join(s.depends_on)}" if s.depends_on else "")
        for s in out.plan
    )
    rec.state("working", message=f"plan: {summary}")
    if plan_only:
        out.answer = json.dumps({"steps": [s.to_dict() for s in out.plan]}, indent=2)
        return finish("completed")

    out.results = await run_steps(out.plan, catalogue, http, rec, step_timeout)
    rec.state("working", message="combining the results")
    try:
        reply = await combine(task, out.plan, out.results, chat)
    except Exception as exc:
        return finish("failed", f"combining failed: {type(exc).__name__}: {exc}", stop="error")
    out.answer = reply.content.strip()
    out.tokens["prompt"] += reply.prompt_tokens
    out.tokens["completion"] += reply.completion_tokens
    rec.event(
        "llm_response",
        step=2,
        content=out.answer,
        tool_calls=[],
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
    )
    bad = [r for r in out.results.values() if not r.ok]
    if bad:
        listed = ", ".join(f"{r.step.id} {r.state}" for r in bad)
        return finish("failed", f"not every step completed: {listed}", stop="step_failed")
    return finish("completed")


def _stats(out: Orchestration) -> dict[str, Any]:
    done = [r for r in out.results.values() if r.ok]
    return {
        "steps": len(out.plan),
        "steps_completed": len(done),
        "agent_prompt_tokens": sum(int(r.stats.get("prompt_tokens", 0)) for r in done),
        "agent_completion_tokens": sum(int(r.stats.get("completion_tokens", 0)) for r in done),
        "orchestrator_prompt_tokens": out.tokens["prompt"],
        "orchestrator_completion_tokens": out.tokens["completion"],
    }
