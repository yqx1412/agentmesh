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

Reliability (M4, see ``RunPolicy``): a failed attempt is retried, then handed to the next
agent whose card lists the same skill (in discovery order). A timed-out attempt's remote
task is cancelled, and cancelling the whole orchestration cancels every remote task still
running. With a ``PushReceiver`` the steps are followed by push notification instead of
a held-open stream.

Security (M5): every delegated message names the skill it wants, so the agent can check the
orchestrator may use it, and carries the signed results of the steps it depends on
(``attestations``). The orchestrator logs in with its own token (``orchestrate --as``).

Chained tasks (after M6): every step also receives the user's task (``STEP_CONTEXT``), an
agent that replies ``CANNOT: <reason>`` has failed its step (so retries and fallback apply,
and then the planner is asked once for steps that get the missing input; ``replan_brief``),
and skills a card tags ``write`` are planned only when the run allows writes.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from a2a.types import AgentCard, StreamResponse

from agentmesh.access import WRITE_RECORD_HEADER
from agentmesh.client import PushReceiver, TaskRun, ask, ask_push, cancel, discover
from agentmesh.journal import JOURNAL_NAME, JsonlJournal
from agentmesh.numbers import TRIVIAL as _TRIVIAL
from agentmesh.numbers import numbers as _numbers
from agentmesh.urls import strip_citations, urls_in

MAX_STEPS = 6
PLAN_ATTEMPTS = 3
# Longest step result passed to the combining model.
MAX_RESULT = 8000

_STEP_ID = re.compile(r"^s[0-9]{1,2}$")
# A step input that spells out arithmetic: the word "formula", a product or power
# ("layers * kv_heads", "1024^3"). Live, the planner wrote the KV-cache formula without the
# x2 for keys and values, the Analyst followed it, and every run got half the size or less.
_FORMULA = re.compile(r"\bformula\b|[\w)\]]\s*[*\u00d7]\s*[\w(\[]|\w\s*\^\s*\d|\*\*\s*\d", re.I)
_REF = re.compile(r"\{\{\s*(s[0-9]{1,2})\s*\}\}")

PLAN_PROMPT = """\
You are the orchestrator of a team of AI agents. You never do the work yourself: you split \
the user's task into steps and give each step to one skill from the catalogue.

Rules:
- Use only skills from the catalogue, by id. Choose each one by what its description says \
it does, and respect what each agent's description says it cannot do.
- A step's "input" is the message that agent receives. It also gets the user's task, for \
context only, but never the other steps. Make the input complete and self-contained, and say \
what the agent should return.
- When a later step computes something, the step that gathers its inputs must ask for every \
value that computation needs.
- Before the steps, list in "inputs" every value the final answer depends on, and its \
"source": "given in the task", or the id of the step that will look it up. Include values \
the task names only as a condition, for example "at its full native context length" \
(needs that context length), "over a 100 Mbit/s link" (given) or "in fp16" (2 bytes per \
value, given). Then make the step named as each value's source ask for that value by name.
- Never write a formula, method or worked number into a step's input. Name what to \
compute and the values to use; the agent that computes chooses the method.
- Ask each step for exactly what the user's task asks, in the same direction: "A per B" \
stays "A per B", never "B per A". The step that computes the answer asks the task's \
question word for word ("How many A fit in B?", never "the ratio of A to B").
- Keep every qualifier the user's task puts on a value, word for word ("native context \
length", "KV heads", "fp16"). Never swap one for a looser or different word ("maximum \
context length", "attention heads"): to the agent it names a different value.
- To give a step an earlier step's result, write {{<step id>}} in its input, for example \
"Check each of these claims against its cited source: {{s1}}". It is replaced by that \
step's full answer, and the step then runs after it.
- Steps that don't depend on each other run at the same time.
- Use as few steps as the task needs, at most 6. Include every skill the task actually \
needs and no others.
- Give each skill one step that asks for everything the task needs from it, for example \
"Find the license and the native context length of ...". A second step with the same skill \
is allowed only when it needs an earlier step's result.
- Add a verification step only when the user's task asks to check, verify or fact-check \
something, or when the plan writes to a knowledge base. A task that asks for sources or \
citations does not need one: the step that finds the facts cites its sources.
- Anything written to a knowledge base must first go through a verification skill, and \
only the claims it marks SUPPORTED may be stored, each with its source.

- After the steps, in "format", copy word for word the part of the user's task that says \
how the final reply must look (a line to end with, a template, a length), for example \
'End with one line "TOTAL: <number>".', or "" when the task says nothing about it. Only \
the final reply writes that line.

Reply with JSON only: {"inputs": [{"value": "...", "source": "..."}, ...], "steps": \
[{"id": "s1", "skill": "<skill id>", "input": "...", "depends_on": []}, ...], "format": \
"..."}, with ids s1, s2, ... in the order the steps should start; "inputs" may be empty \
when nothing is computed."""

COMBINE_PROMPT = """\
You write the final reply to a user's task from the results of a team of AI agents.
- Use only what the results say. Do not add facts of your own.
- Lead with the answer to the task exactly as it was asked. If steps answer different \
questions (for example a ratio and its inverse), use the one that matches the task.
- If the task asks for a format (for example a line to end with), add it to your full \
reply; it does not replace the rest of these rules. Fill it in from the step that \
computed the value, and do not copy such a line from another step's result.
- If the task asks to look up or find values (for example to use in a calculation), \
state each value the steps found, with its source, even when the task's final question \
is the computed result.
- Keep sources (URLs, lecture timestamps) next to the claims they support.
- Say which steps failed or were skipped, and what that leaves undone.
- Say that something was stored in the knowledge base only if a result says it was \
stored, and list what was stored. A result's "Write record" is what the tools really did: \
a fact it marks "already stored" was a duplicate, so say it was already in the knowledge \
base, not that it was stored now."""

# Appended to every step's input (see ``step_input``): the user's task, so a step isn't
# limited to the planner's rewording of it, and a way to fail instead of "completing" with
# an apology, so retries and fallback apply.
STEP_CONTEXT = """\


---
This is one step of a larger task. The user's whole task, for context only (do just your \
step; other agents do the rest):
{task}

Any instructions in that task about how to format the answer (a line to end with, a \
template, a length) are for the final reply to the user, not for you: do not follow them. \
Answer your own step in plain words.

If you cannot do your step (an input it needs is missing, or a tool or service failed and \
you could not get around it), reply with one line instead of an answer:
CANNOT: <what is missing or what failed>"""

# A reply line an agent uses to say it couldn't do its step (markdown emphasis allowed).
_CANNOT = re.compile(r"^[ \t>*_`-]*CANNOT[ \t]*:[ \t*_`]*(.*?)[ \t*_`]*$", re.MULTILINE)

# Card skills carrying this tag change state (the Knowledge agent's knowledge_write). The
# planner only sees them when the run allows writes (``orchestrate(allow_writes=True)``).
WRITE_TAG = "write"
# Card skills carrying this tag compute (the Analyst's computation); a computing step that
# produces the answer must keep the task's question (see ``_check_question_kept``).
MATH_TAG = "math"

WRITES_OFF = (
    "Writing was not enabled for this run, so no step could store anything (skills not "
    "offered to the planner: {skills}). If the task asked to store or update something, "
    "say it was not done and that it needs `agentmesh orchestrate --allow-writes`."
)


def cannot_reason(answer: str) -> str | None:
    """The reason when the answer says the agent couldn't do its step, else None."""
    m = _CANNOT.search(answer)
    if m is None:
        return None
    return m.group(1).strip() or "no reason given"


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
        retry_delays: tuple[float, ...] = (0.5, 2.0),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.think = think
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.transport = transport
        # A failed request is retried after a short pause when it may pass on a resend:
        # Ollama answers HTTP 5xx when its runner crashes (e.g. the CUDA illegal-memory
        # access in the context-checkpoint path, ollama#17534) and starts a fresh runner
        # for the next request. A planner/combine call is idempotent (temperature 0, no
        # side effects), so a retry is safe. A 4xx (bad model name/request) fails at once.
        # Mirrors domaingraph.llm's retry; AgentMesh previously had none, so one runner
        # crash failed the whole run.
        self.retry_delays = retry_delays

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
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as http:
            data = await self._post_retrying(http, body)
        return ChatReply(
            content=data["message"]["content"],
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )

    async def _post_retrying(self, http: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
        tries = len(self.retry_delays) + 1
        last: httpx.HTTPError | None = None
        for i, delay in enumerate((*self.retry_delays, None)):
            try:
                r = await http.post(f"{self.url}/api/chat", json=body)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                # 5xx (a crashed runner) may pass on resend; 4xx will not.
                if exc.response.status_code < 500 or delay is None:
                    if delay is None and i:
                        raise httpx.HTTPError(f"{exc} (gave up after {tries} tries)") from exc
                    raise
                last = exc
            except httpx.TransportError as exc:  # no connection yet / dropped: may pass
                if delay is None:
                    raise httpx.HTTPError(f"{exc} (gave up after {tries} tries)") from exc
                last = exc
            await asyncio.sleep(delay)  # type: ignore[arg-type]  # delay is not None here
        raise last or AssertionError("unreachable")


# --- the catalogue ---------------------------------------------------------------------


@dataclass
class Skill:
    id: str
    name: str
    description: str
    agent: str
    card: AgentCard
    examples: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def writes(self) -> bool:
        return WRITE_TAG in self.tags


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
                Skill(s.id, s.name, s.description, card.name, card, list(s.examples), list(s.tags))
            )
    return cat


def without_writes(catalogue: Catalogue) -> tuple[Catalogue, list[str]]:
    """The catalogue minus every skill some card tags ``write``, and those skills' ids."""
    removed = sorted(sid for sid, skills in catalogue.items() if any(s.writes for s in skills))
    return {k: v for k, v in catalogue.items() if k not in removed}, removed


def describe_catalogue(cards: list[AgentCard], only: Catalogue | None = None) -> str:
    """The catalogue as the planner reads it: each agent's description, then its skills
    (only those in ``only``, when given; an agent left with none is not listed)."""
    lines = []
    for card in cards:
        skills = [s for s in card.skills if only is None or s.id in only]
        if not skills:
            continue
        lines.append(f"Agent: {card.name}. {card.description}")
        for s in skills:
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
    # Constrained decoding writes properties in schema order: the planner names every value
    # the answer depends on ("inputs") before it writes the steps, and only then copies the
    # task's reply format ("format"). Format first made the planner write "Compute the
    # ratio of A to B" instead of the task's "How many A fit in B" (0/3 on the ratio task).
    return {
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "required": ["value", "source"],
                },
            },
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
            },
            "format": {"type": "string"},
        },
        "required": ["inputs", "steps", "format"],
    }


def _span(text: str) -> re.Pattern[str] | None:
    """A pattern matching ``text`` word for word, ignoring case and spacing (the task may
    be wrapped differently from the planner's copy)."""
    words = text.split()
    return re.compile(r"\s+".join(map(re.escape, words)), re.I) if words else None


def plan_format(content: str) -> str:
    """The planner's "format": the part of the task that says how the final reply must
    look, or "" (also for a plan without one)."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ""
    raw = data.get("format") if isinstance(data, dict) else None
    return raw.strip() if isinstance(raw, str) else ""


# The quoted template in a format instruction: End with one line "RATIO: <number>".
_TEMPLATE = re.compile(r"[\"\u201c'`]([^\"\u201d'`]+)[\"\u201d'`]")


def format_lead(fmt: str) -> str:
    """The fixed label a reply-format line starts with ("RATIO:" for 'End with one line
    "RATIO: <number>".', or for a bare 'RATIO: <number>'), or "" when the format has no
    "LABEL: <...>" template to recognise (for example "Cite the pages you used.")."""
    m = _TEMPLATE.search(fmt)
    template = m.group(1) if m else fmt
    lead = re.split(r"[<{\[]", template, maxsplit=1)[0].strip()
    return lead if re.fullmatch(r"[A-Za-z][\w ()/-]{0,39}:", lead) else ""


def drop_format_lines(answer: str, fmt: str) -> tuple[str, list[str]]:
    """A step's answer without the lines that fill in the task's reply format, and the
    lines dropped.

    Steps see the whole task, format included: the format can say what is being asked
    ("RATIO: <number>" after "How many ... fit" names the 8, not 0.125; without it the final
    reply wrote "RATIO: 0.125" in 3 of 3 runs). But a prompt rule telling steps not to
    follow it didn't hold: Research kept ending its own answer with a "RATIO:" line (once
    "RATIO: 0.25", which the final reply kept). Only the final reply may write that line, so
    it is taken out of every step's answer before anything else reads it."""
    lead = format_lead(fmt)
    if not lead:
        return answer, []
    # Case-sensitive: live, a case-insensitive match also took "- Ratio: 4096 / 32768 =
    # 0.125" out of the Analyst's list of numbers, which is content, not the format line.
    line = re.compile(r"^[ \t>*_`#-]*" + re.escape(lead) + r".*$", re.MULTILINE)
    dropped = [m.group(0).strip() for m in line.finditer(answer)]
    if not dropped:
        return answer, []
    return re.sub(r"\n{3,}", "\n\n", line.sub("", answer)).strip(), dropped


def plan_inputs(content: str) -> list[dict[str, str]]:
    """The planner's "inputs" list (values the answer depends on, and where each comes from).
    It only guides the planner and goes into the trace; a plan without one is still valid."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    raw = data.get("inputs") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    return [
        {"value": str(i.get("value", "")), "source": str(i.get("source", ""))}
        for i in raw
        if isinstance(i, dict)
    ]


def parse_plan(
    content: str,
    catalogue: Catalogue,
    done: frozenset[str] = frozenset(),
    taken: frozenset[str] = frozenset(),
    redo: frozenset[str] = frozenset(),
    task: str = "",
) -> list[Step]:
    """Check a plan completely: ids, skills, inputs, dependencies, no cycles.

    For a re-plan, ``taken`` holds every earlier step id (none may be reused), ``done``
    the ones that completed, which new steps may depend on, and ``redo`` the skills of the
    steps that replied CANNOT, each of which the new plan must use again. A step input may
    not spell out a formula unless the user's ``task`` itself contains it."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlanError(f"the reply is not valid JSON ({exc})") from exc
    raw = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise PlanError('the plan needs a non-empty "steps" list')
    if len(raw) > MAX_STEPS:
        raise PlanError(f"the plan has {len(raw)} steps; the limit is {MAX_STEPS}")
    fmt = _span(plan_format(content))
    if fmt is not None and task and not fmt.search(task):
        raise PlanError(
            '"format" must be copied word for word from the user\'s task, or be "" when '
            "the task says nothing about how the reply must look"
        )

    steps: list[Step] = []
    for n, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise PlanError(f"step {n} is not an object")
        sid = str(item.get("id", "")).strip()
        if not _STEP_ID.match(sid):
            raise PlanError(f"step {n}: id must be s1, s2, ...; got {sid!r}")
        if any(s.id == sid for s in steps):
            raise PlanError(f"two steps have the id {sid}")
        if sid in done or sid in taken:
            raise PlanError(f"{sid} is an earlier step; new steps need new ids")
        skill = str(item.get("skill", "")).strip()
        if skill not in catalogue:
            known = ", ".join(sorted(catalogue))
            raise PlanError(f"step {sid}: unknown skill {skill!r}; the skills are: {known}")
        text = str(item.get("input", "")).strip()
        if not text:
            raise PlanError(f"step {sid} has an empty input")
        formula = next(
            (m for m in _FORMULA.finditer(text) if m.group(0).lower() not in task.lower()), None
        )
        if formula:
            raise PlanError(
                f"step {sid}'s input spells out a formula or method ({formula.group(0)!r}); "
                "name only what to compute and which values to use, and let the agent that "
                "computes choose the method"
            )
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
            if d not in ids and d not in done:
                why = "did not complete" if d in taken else "is not a step in the plan"
                raise PlanError(f"step {s.id} uses {d}, which {why}")
    _check_acyclic(steps, done)
    _check_one_step_per_skill(steps)
    _check_question_kept(steps, task, catalogue)
    missing = sorted(redo - {s.skill for s in steps})
    if missing:
        raise PlanError(
            "the new steps must redo the work that failed; add a step using "
            + ", ".join(missing)
            + " that gets the missing input"
        )
    return steps


def _words(text: str) -> list[str]:
    """Lowercase words, with {{sN}} references and any brackets holding them removed."""
    text = re.sub(r"\(\s*\{\{\s*s\d+\s*\}\}\s*\)|\{\{\s*s\d+\s*\}\}", " ", text)
    return re.findall(r"[\w']+", text.lower().replace("\u2019", "'"))


def _in_order(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(w in it for w in needle)


def _check_question_kept(steps: list[Step], task: str, catalogue: Catalogue) -> None:
    """A computing step (a skill its card tags ``math``) that turns earlier steps' results
    into the answer must ask the task's question in the task's words (other words may be
    added between them, and {{sN}} references).

    Live, the planner turned "How many of A fit in B?" into "Compute the ratio of A to B"
    in most runs; the Analyst then correctly gave A/B = 0.125 instead of 8. Only a task
    with exactly one question ("?") is checked, and only computing steps that use another
    step's result and that no step uses (the ones that produce the answer)."""
    one = " ".join(task.split())  # a question may be wrapped over lines
    questions = [q.strip() for q in re.findall(r"[^.?!]*\?", one) if q.strip()]
    if len(questions) != 1:
        return
    want = _words(questions[0])
    used = {d for s in steps for d in s.depends_on}

    def computes(s: Step) -> bool:
        return any(MATH_TAG in (getattr(k, "tags", None) or []) for k in catalogue[s.skill])

    finals = [s for s in steps if s.depends_on and s.id not in used and computes(s)]
    if finals and not any(_in_order(want, _words(s.input)) for s in finals):
        ids = ", ".join(s.id for s in finals)
        raise PlanError(
            f"step {ids} computes the answer, so its input must ask the user's question in "
            f'the user\'s own words: "{questions[0]}" (add the values to use around it, or '
            "{{sN}} references, but do not reword the question)"
        )


def _check_one_step_per_skill(steps: list[Step]) -> None:
    """Two steps may share a skill only when one of them needs the other's result.

    Independent steps on one skill each cost a full agent run (the M6 rerun's planner split
    one Qwen3-8B lookup into two parallel Research steps); one step can ask for both."""
    by_id = {s.id: s for s in steps}
    upstream: dict[str, set[str]] = {}

    def ancestors(sid: str) -> set[str]:
        if sid not in upstream:
            upstream[sid] = set()
            for d in by_id[sid].depends_on:
                if d in by_id:
                    upstream[sid] |= {d} | ancestors(d)
        return upstream[sid]

    for i, a in enumerate(steps):
        for b in steps[i + 1 :]:
            if a.skill == b.skill and a.id not in ancestors(b.id) and b.id not in ancestors(a.id):
                raise PlanError(
                    f"{a.id} and {b.id} both use {a.skill} and neither needs the other's "
                    f"result; ask for everything the task needs from {a.skill} in one step"
                )


def _check_acyclic(steps: list[Step], done: frozenset[str] = frozenset()) -> None:
    remaining = {s.id: set(s.depends_on) - done for s in steps}
    while remaining:
        ready = [sid for sid, deps in remaining.items() if not deps]
        if not ready:
            raise PlanError(f"the steps {sorted(remaining)} depend on each other in a cycle")
        for sid in ready:
            del remaining[sid]
        for deps in remaining.values():
            deps.difference_update(ready)


async def make_plan(
    task: str,
    cards: list[AgentCard],
    catalogue: Catalogue,
    chat: Chat,
    *,
    earlier: str = "",
    done: frozenset[str] = frozenset(),
    taken: frozenset[str] = frozenset(),
    redo: frozenset[str] = frozenset(),
    tokens: dict[str, int] | None = None,
) -> tuple[list[Step], list[dict[str, Any]]]:
    """A valid plan plus one record per model attempt. Raises PlanError after the last try.

    For a re-plan, ``earlier`` describes the steps so far (``replan_brief``), ``taken``
    holds their ids and ``done`` the completed ones, which the new steps may use.

    ``tokens`` (the run's ``out.tokens``) is credited the instant each reply returns, before
    the next ``await``. A cancel landing mid-plan then still records the tokens already
    spent -- the caller's post-return fold never runs on the cancel path, so without this
    a cancelled run reported zero planner tokens."""
    messages = [
        {"role": "system", "content": PLAN_PROMPT},
        {
            "role": "user",
            "content": f"Catalogue:\n{describe_catalogue(cards, catalogue)}\n\n"
            f"User's task:\n{task}" + (f"\n\n{earlier}" if earlier else ""),
        },
    ]
    schema = plan_schema(sorted(catalogue))
    attempts: list[dict[str, Any]] = []
    for _ in range(PLAN_ATTEMPTS):
        reply = await chat(messages, schema)
        if tokens is not None:
            tokens["prompt"] += reply.prompt_tokens
            tokens["completion"] += reply.completion_tokens
        record = {
            "content": reply.content,
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
        }
        attempts.append(record)
        try:
            return parse_plan(reply.content, catalogue, done, taken, redo, task), attempts
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
    attempts: list[Attempt] = field(default_factory=list)
    attestation: str = ""  # this step's signed result, when its agent signs (M5)
    attestations: list[str] = field(default_factory=list)  # forwarded from dependencies
    cannot: str = ""  # the agent's reason, when an attempt replied CANNOT

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
            "attempts": [a.to_dict() for a in self.attempts],
            "signed": bool(self.attestation),
            "forwarded_signatures": len(self.attestations),
            "cannot": self.cannot,
        }


def fill(text: str, results: dict[str, StepResult]) -> str:
    """Replace every ``{{sN}}`` with step sN's answer."""
    return _REF.sub(lambda m: results[m.group(1)].answer.strip(), text)


def step_input(step: Step, results: dict[str, StepResult], task: str = "") -> str:
    """The message a step's agent receives.

    The planner often lists a dependency in ``depends_on`` but never writes ``{{sN}}`` into
    the input ("Verify the claims from the research"), which would send the agent a request
    about data it never sees. Any dependency the input doesn't reference is appended.

    With ``task``, the user's own task is appended too (``STEP_CONTEXT``): in M6 every
    chained-task failure came from a step seeing only the planner's rewording, which had
    dropped an input or flipped a ratio. It also tells the agent how to fail (``CANNOT:``).
    """
    text = fill(step.input, results)
    used = set(_REF.findall(step.input))
    for dep in step.depends_on:
        if dep not in used:
            text += f"\n\nResult of step {dep}:\n{results[dep].answer.strip()}"
    if task.strip():
        text += STEP_CONTEXT.format(task=task.strip())
    return text


# Longest earlier result shown to the planner in a re-plan.
MAX_BRIEF_RESULT = 2000


def replan_brief(plan: list[Step], results: dict[str, StepResult]) -> str:
    """What the planner reads on a re-plan: every step so far with its outcome, and what to
    write. Completed results stay usable as ``{{sN}}``, so finished work isn't repeated."""
    context_start = STEP_CONTEXT.split("{task}")[0]
    lines = []
    for s in plan:
        r = results.get(s.id)
        if r is None:
            continue
        asked = (r.input or s.input).split(context_start)[0].strip()
        if len(asked) > MAX_BRIEF_RESULT:
            asked = asked[:MAX_BRIEF_RESULT] + " [...]"
        if r.ok:
            body = r.answer.strip()
            if len(body) > MAX_BRIEF_RESULT:
                body = body[:MAX_BRIEF_RESULT] + " [...]"
            outcome = f"completed. Result:\n{body}"
        elif r.cannot:
            outcome = f"FAILED, the agent replied: CANNOT: {r.cannot}"
        else:
            outcome = f"{r.state}: {r.message[:500]}"
        lines.append(f"### {s.id} ({s.skill}) - {outcome}\nAsked: {asked}")
    nxt = max((int(sid[1:]) for sid in results), default=0) + 1
    return (
        "An earlier plan for this task has already run. At least one step said it could not "
        "do its part, usually because an input it needs was never asked for.\n\n"
        + "\n\n".join(lines)
        + "\n\nWrite a plan of ONLY the new steps that finish the task: first get what the "
        "failed steps said was missing, then redo those steps and any step skipped because "
        "of them. Use a completed step's result by writing {{" + "<its id>}} in an input "
        f"(and listing it in depends_on); do not repeat completed steps. New step ids must "
        f"start at s{nxt} and continue from there."
    )


@dataclass
class RunPolicy:
    """How hard a step tries before it fails.

    A step is tried on the first agent whose card lists its skill. A failure is retried on
    the same agent up to ``retries`` times (after ``backoff`` seconds, doubling), except a
    timeout, which is never retried on the same agent. Then, with ``fallback``, the next
    agent with the skill gets the same treatment. A timed-out attempt's remote task is
    cancelled, so the agent stops working on it.
    """

    step_timeout: float = 900.0  # per attempt
    retries: int = 1
    backoff: float = 2.0
    fallback: bool = True
    push: PushReceiver | None = None  # follow tasks by push notification instead of a stream
    poll_every: float = 15.0  # push mode: ask the agent if nothing arrived for this long
    # On cancel, how long a step waits for the agent to report its task id so the cancel can
    # reach the remote task. The id is normally the agent's first reply (within ms), but a
    # slow agent could still be mid-first-call when the cancel lands; a too-short wait gives
    # up before the id arrives and the remote task keeps running uncancelled. 10s matches the
    # fleet/eval cancel grace (fleet.stop, CancelAt.grace).
    cancel_id_wait: float = 10.0
    # After the run, a step that failed with CANNOT (a missing input) sends the planner back
    # once per re-plan to add the steps that get what was missing (``replan_brief``).
    replans: int = 1


@dataclass
class Attempt:
    agent: str
    url: str
    task_id: str = ""
    state: str = "working"
    message: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "url": self.url,
            "task_id": self.task_id,
            "state": self.state,
            "message": self.message,
            "seconds": round(self.seconds, 1),
        }


async def _attempt(
    step: Step,
    skill: Skill,
    r: StepResult,
    a: Attempt,
    http: httpx.AsyncClient,
    rec: Recorder,
    policy: RunPolicy,
) -> TaskRun | None:
    """One try on one agent. Returns the run, or None when it timed out or the agent was
    unreachable (``a`` says which). Cancelling this coroutine cancels the remote task."""

    def seen(event: StreamResponse) -> None:
        kind = event.WhichOneof("payload")
        tid = event.task.id if kind == "task" else ""
        if tid and not a.task_id:
            a.task_id = r.task_id = tid
            rec.state("working", message=f"{step.id}: {a.agent} took it as task {tid}")

    meta: dict[str, Any] = {"orchestration": rec.run_id, "step": step.id, "skill": step.skill}
    if r.attestations:
        meta["attestations"] = list(r.attestations)
    if policy.push is not None:
        call = ask_push(
            skill.card,
            r.input,
            http,
            policy.push,
            on_event=seen,
            metadata=meta,
            poll_every=policy.poll_every,
        )
    else:
        call = ask(skill.card, r.input, http, on_event=seen, metadata=meta)
    # The call runs as its own task so that a cancel of this coroutine doesn't cut it off
    # before the agent has told us the task id: without the id there is nothing to cancel.
    job = asyncio.ensure_future(call)
    try:
        finished, _ = await asyncio.wait({job}, timeout=policy.step_timeout)
    except asyncio.CancelledError:  # the orchestration was cancelled: stop the agent too
        await asyncio.shield(_stop(job, step, skill, a, http, rec, policy))
        raise
    if finished:
        try:
            return job.result()
        except Exception as exc:
            a.state, a.message = "unreachable", f"{type(exc).__name__}: {exc}"[:500]
            return None
    await _drop(job)
    a.state, a.message = "timeout", f"no result after {policy.step_timeout:g} s"
    if a.task_id:
        await _cancel_remote(step, skill, a, http, rec)
    return None


# How long a cancelled step waits for the agent to report its task id, so the cancel can
# reach the remote task. The id is normally the agent's first reply, within milliseconds.
# This is the default; a run can override it with RunPolicy.cancel_id_wait.
ID_WAIT = 10.0


async def _stop(
    job: asyncio.Future,
    step: Step,
    skill: Skill,
    a: Attempt,
    http: httpx.AsyncClient,
    rec: Recorder,
    policy: RunPolicy | None = None,
) -> None:
    wait = policy.cancel_id_wait if policy is not None else ID_WAIT
    deadline = time.monotonic() + wait
    while not a.task_id and not job.done() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    if a.task_id and not job.done():
        await _cancel_remote(step, skill, a, http, rec)
    elif not a.task_id and not job.done():
        rec.state(
            "working",
            message=f"{step.id}: {a.agent} gave no task id in {wait:g} s; nothing to cancel",
        )
    await _drop(job)


async def _drop(job: asyncio.Future) -> None:
    """Cancel the call and wait for it to unwind, ignoring how it ends."""
    job.cancel()
    await asyncio.wait({job})  # never raises the call's own error
    if not job.cancelled():
        job.exception()  # mark it retrieved: the call's outcome no longer matters


async def _cancel_remote(
    step: Step, skill: Skill, a: Attempt, http: httpx.AsyncClient, rec: Recorder
) -> None:
    try:
        said = await asyncio.wait_for(cancel(skill.card, a.task_id, http), 10)
    except TimeoutError:
        said = "cancel failed: no reply in 10 s"
    a.message += f"; remote task {said}" if a.message else f"remote task {said}"
    rec.state("working", message=f"{step.id}: asked {a.agent} to cancel {a.task_id}: {said}")


async def _run_one(
    step: Step,
    results: dict[str, StepResult],
    catalogue: Catalogue,
    http: httpx.AsyncClient,
    rec: Recorder,
    policy: RunPolicy,
    task: str = "",
    fmt: str = "",
) -> None:
    """Runs one step to a terminal state. Never raises (except when cancelled): failures,
    retries and fallbacks go into the result."""
    r = results[step.id]
    r.input = step_input(step, results, task)
    # M5: hand on every signed result this step depends on (the Knowledge agent only
    # writes what the Verifier signed). The orchestrator can relay them but not make one.
    r.attestations = [results[d].attestation for d in step.depends_on if results[d].attestation]
    r.state, r.started = "working", time.time()
    candidates = catalogue[step.skill] if policy.fallback else catalogue[step.skill][:1]
    for n, skill in enumerate(candidates):
        r.agent, r.url = skill.agent, agent_url(skill.card)
        tries = 0
        while True:
            tries += 1
            a = Attempt(skill.agent, r.url)
            r.attempts.append(a)
            r.task_id = ""
            how = "" if tries == 1 and n == 0 else f" (attempt {len(r.attempts)})"
            rec.state("working", message=f"{step.id}: sending {step.skill} to {a.agent}{how}")
            t0 = time.time()
            run = await _attempt(step, skill, r, a, http, rec, policy)
            a.seconds = time.time() - t0
            if run is not None:
                a.task_id = run.task_id or a.task_id
                a.state, a.message = run.state or "failed", run.final_message
                answer, dropped = drop_format_lines(run.answer, fmt)
                if dropped:
                    rec.state(
                        "working",
                        message=f"{step.id}: dropped the reply-format line(s) "
                        f"{dropped} from {a.agent}'s answer; only the final reply writes them",
                    )
                r.answer, r.stats = answer, run.stats
                r.attestation = run.attestation
                if a.state == "completed" and not answer.strip():
                    a.state, a.message = "failed", "the agent completed with an empty answer"
                elif a.state == "completed" and (why := cannot_reason(answer)):
                    # An agent that says it couldn't do its part has failed the step, even
                    # though its A2A task completed; without this, M4's retries and
                    # fallback never fire and the next step gets an apology as its input.
                    a.state, a.message = "failed", f"the agent could not do the step: {why}"
                    r.cannot = why
            r.task_id, r.state, r.message = a.task_id, a.state, a.message
            if a.state == "completed":
                break
            rec.state(
                "working",
                message=f"{step.id}: {a.agent} {a.state} after {a.seconds:.0f} s: "
                f"{a.message[:300]}",
            )
            # A timeout or a refusal won't go better on the same agent; anything else might.
            if a.state in ("timeout", "canceled", "rejected") or tries > policy.retries:
                break
            await asyncio.sleep(policy.backoff * 2 ** (tries - 1))
        if r.ok:
            break
        if n + 1 < len(candidates):
            rec.state(
                "working",
                # Name + URL: two agents can share a card name (two Research agents).
                message=f"{step.id}: falling back from {skill.agent} ({agent_url(skill.card)}) "
                f"to {candidates[n + 1].agent} ({agent_url(candidates[n + 1].card)})",
            )
    if not r.ok:
        r.state = "failed"
        r.answer = ""
        tried = "; ".join(f"{a.agent}: {a.state}, {a.message[:200]}" for a in r.attempts)
        r.message = f"every attempt failed ({tried})"
    r.ended = time.time()
    took = f"{r.ended - r.started:.0f} s"
    detail = "" if r.ok else f": {r.message[:300]}"
    rec.state("working", message=f"{step.id}: {r.agent} {r.state} after {took}{detail}")
    rec.event(
        "tool_result",
        step=1,
        call={
            "name": r.label,
            "arguments": {
                "input": r.input,
                "task": r.task_id,
                "attempts": [a.to_dict() for a in r.attempts],
            },
        },
        ok=r.ok,
        output=r.answer if r.ok else f"{r.state}: {r.message}",
    )


async def run_steps(
    plan: list[Step],
    catalogue: Catalogue,
    http: httpx.AsyncClient,
    rec: Recorder,
    policy: RunPolicy | float,
    task: str = "",
    done: dict[str, StepResult] | None = None,
    fmt: str = "",
) -> dict[str, StepResult]:
    """Runs the plan. With ``task`` (the user's task), every step also receives it; with
    ``fmt`` (the task's reply format), lines filling it in are dropped from step answers.

    ``done`` holds completed steps from an earlier plan (a re-plan): the new steps can
    depend on them, and they are returned with the new results."""
    if not isinstance(policy, RunPolicy):  # the M3 signature passed a step timeout
        policy = RunPolicy(step_timeout=float(policy))
    results = {**(done or {}), **{s.id: StepResult(s) for s in plan}}
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
                            _run_one(s, results, catalogue, http, rec, policy, task, fmt)
                        )
                    )
        if not running:
            break
        try:
            done, running = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # The orchestration was cancelled: cancel every running step, which cancels its
            # remote task, and wait for that cleanup before giving up.
            for t in running:
                t.cancel()
            await asyncio.gather(*running, return_exceptions=True)
            for r in results.values():
                if r.state in ("pending", "working"):
                    r.state, r.message = "canceled", "the orchestration was cancelled"
            raise
        for t in done:
            t.result()  # _run_one doesn't raise; surface a bug if it does
    return results


async def combine(
    task: str,
    plan: list[Step],
    results: dict[str, StepResult],
    chat: Chat,
    writes_off: list[str] | None = None,
    left_out: list[str] | None = None,
    tokens: dict[str, int] | None = None,
) -> ChatReply:
    context_start = STEP_CONTEXT.split("{task}")[0]
    parts = []
    for s in plan:
        r = results[s.id]
        body = r.answer if r.ok else f"({r.state}) {r.message}"
        if len(body) > MAX_RESULT:
            body = body[:MAX_RESULT] + " [...]"
        asked = (r.input or s.input).split(context_start)[0]  # the task is given once below
        parts.append(
            f"### {s.id}: {s.skill} by {r.agent or 'no agent'} - {r.state}\n"
            f"Asked: {asked}\n\nResult:\n{body}"
        )
    note = f"\n\n{WRITES_OFF.format(skills=', '.join(writes_off))}" if writes_off else ""
    if left_out:
        note += "\n\n" + LEFT_OUT.format(values=", ".join(left_out))
    messages = [
        {"role": "system", "content": COMBINE_PROMPT},
        {
            "role": "user",
            "content": f"User's task:\n{task}{note}\n\nStep results, in plan order:\n\n"
            + "\n\n".join(parts),
        },
    ]
    reply = await chat(messages, None)
    if tokens is not None:
        tokens["prompt"] += reply.prompt_tokens
        tokens["completion"] += reply.completion_tokens
    return reply


def ungrounded_citations(task: str, answer: str, results: dict[str, StepResult]) -> list[str]:
    """URLs the combined answer cites that no completed step's answer carries and the task
    did not give -- citations the combiner invented or mangled.

    The orchestrator never fetches; the pages live in the Research sub-run (a separate
    process and trace), so the combiner has no fetched set of its own to check against. But
    a Research sub-answer has already passed Research's own citation guard, so every URL it
    carries is a page Research fetched. The combiner's job is to relay, not to source, so a
    citation in the final answer that appears in no step result (and isn't one the user's
    task supplied) is unsupported -- the combiner either hallucinated it or corrupted a real
    one. Those are the citations to drop. This is the orchestrator analogue of Research's
    ``AnswerGuardLLM`` citation check, which only governs Research's own replies; before
    this the combined answer passed through no guard at all. A 2026-10-26 M6 run showed the
    gap: Research fetched docs.ollama.com correctly, but the eval flagged the final answer's
    citation because nothing re-checked the orchestrator's synthesis."""
    grounded = set(urls_in(task))
    for r in results.values():
        if r.ok:
            grounded.update(urls_in(r.answer))
    return [u for u in urls_in(answer) if u not in grounded]


def write_records(plan: list[Step], results: dict[str, StepResult]) -> str:
    """Every completed step's write record (see ``access.Grant.write_report``), in plan
    order, as one block for the end of the final answer. Empty when no step wrote."""
    blocks = []
    for s in plan:
        r = results.get(s.id)
        if r is None or not r.ok or WRITE_RECORD_HEADER not in r.answer:
            continue
        lines = r.answer.split(WRITE_RECORD_HEADER, 1)[1].strip().splitlines()
        items = [ln for ln in lines if ln.startswith("- ")]
        if items:
            blocks.append(f"{s.id} ({r.agent or s.skill}):\n" + "\n".join(items))
    return f"{WRITE_RECORD_HEADER}\n" + "\n".join(blocks) if blocks else ""


# Said to the combiner when its first reply left out values the steps looked up.
LEFT_OUT = (
    "Your previous reply left out these values that the steps looked up and the "
    "computation used: {values}. Write the reply again, and state each of them with what "
    "it is and its source."
)
# A task that asks for values to be found, so the reply must state them (COMBINE_PROMPT).
_ASKS_LOOKUP = re.compile(r"\b(find|look(?:s|ing)? (?:it |them )?up|search)\b", re.I)


def values_left_out(
    task: str,
    plan: list[Step],
    results: dict[str, StepResult],
    catalogue: Catalogue,
    answer: str,
) -> list[str]:
    """Values the steps looked up and a computing step used, that the reply doesn't state.

    Live, the summary sometimes replied with only "RATIO: 8.0" although the task asked to
    find Ollama's default window and Qwen3-8B's context length (2 of 10 ratio runs failed
    on it). A looked-up value is a number in the answer of a computing step (a skill its
    card tags ``math``) that also appears in the answer of a step it depends on, directly
    or not, and is not in the task itself. Only checked when the task asks to find, look
    up or search for something."""
    if not _ASKS_LOOKUP.search(task):
        return []
    by_id = {s.id: s for s in plan}

    def computes(s: Step) -> bool:
        return any(MATH_TAG in (getattr(k, "tags", None) or []) for k in catalogue.get(s.skill, []))

    def upstream(sid: str, seen: set[str]) -> set[str]:
        for d in by_id[sid].depends_on:
            if d in by_id and d not in seen:
                seen.add(d)
                upstream(d, seen)
        return seen

    given = _numbers(task)
    used: dict[Decimal, str] = {}
    for s in plan:
        r = results.get(s.id)
        if r is None or not r.ok or not computes(s):
            continue
        computed = _numbers(r.answer)
        for d in upstream(s.id, set()):
            dr = results.get(d)
            if dr is None or not dr.ok or computes(by_id[d]):
                continue
            for value, text in _numbers(dr.answer).items():
                # 0, 1 and 2 are counts and factors ("1 window", "x2 for K and V", "2
                # bytes") far more often than looked-up values.
                if value in computed and value not in given and value not in _TRIVIAL:
                    used.setdefault(value, text)
    stated = _numbers(answer)
    return [text for value, text in used.items() if value not in stated]


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
    replans: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "state": self.state,
            "message": self.message,
            "replans": self.replans,
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
    policy: RunPolicy | None = None,
    plan_only: bool = False,
    on_line: Callable[[str], None] | None = None,
    model_name: str = "",
    allow_writes: bool = False,
) -> Orchestration:
    """Plan, run and combine. ``policy`` sets retries, fallback and push; without one, each
    step gets ``step_timeout`` and the default policy. Cancelling this coroutine cancels
    every remote task still running and records the run as canceled.

    Skills a card tags ``write`` are offered to the planner only with ``allow_writes``: in
    M6 the planner added knowledge-base writes to tasks that only asked a question. The
    M5 signature check stops unverified writes, this stops unrequested ones."""
    policy = policy or RunPolicy(step_timeout=step_timeout)
    run_id = str(uuid.uuid4())
    rec = Recorder(trace_root / "orchestrator" if trace_root else None, run_id, on_line)
    out = Orchestration(run_id, task, trace=rec.trace)
    try:
        return await _orchestrate(
            task, cards, chat, http, rec, out, policy, plan_only, model_name, allow_writes
        )
    except asyncio.CancelledError:
        out.state, out.message = "canceled", "the orchestration was cancelled"
        rec.event("run_end", stop_reason="cancelled", answer="")
        rec.state("canceled", message=out.message)
        raise


async def _orchestrate(
    task: str,
    cards: list[AgentCard],
    chat: Chat,
    http: httpx.AsyncClient,
    rec: Recorder,
    out: Orchestration,
    policy: RunPolicy,
    plan_only: bool,
    model_name: str,
    allow_writes: bool = False,
) -> Orchestration:
    catalogue = build_catalogue(cards)
    writes_off: list[str] = []
    if not allow_writes:
        catalogue, writes_off = without_writes(catalogue)

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
    if writes_off:
        rec.state(
            "working",
            message=f"writes are off, not offering: {', '.join(writes_off)} (--allow-writes)",
        )
    if not catalogue:
        return finish("failed", "no agents with skills were found", stop="no_agents")

    try:
        out.plan, attempts = await make_plan(task, cards, catalogue, chat, tokens=out.tokens)
    except PlanError as exc:
        return finish("failed", str(exc), stop="plan_error")
    except Exception as exc:
        return finish("failed", f"planner error: {type(exc).__name__}: {exc}", stop="error")
    rec.event(
        "llm_response",
        step=1,
        content=json.dumps(
            {
                "format": plan_format(attempts[-1]["content"]),
                "inputs": plan_inputs(attempts[-1]["content"]),
                "steps": [s.to_dict() for s in out.plan],
            },
            indent=2,
        )
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
    fmt = plan_format(attempts[-1]["content"])
    if format_lead(fmt):
        rec.state("working", message=f"reply format, kept out of step answers: {fmt}")
    if plan_only:
        out.answer = json.dumps({"format": fmt, "steps": [s.to_dict() for s in out.plan]}, indent=2)
        return finish("completed")

    out.results = await run_steps(out.plan, catalogue, http, rec, policy, task, fmt=fmt)
    model_step = 1
    for _ in range(max(0, policy.replans)):
        stuck = [r for r in out.results.values() if r.state == "failed" and r.cannot]
        if not stuck:
            break
        model_step += 1
        why = "; ".join(f"{r.step.id}: {r.cannot[:200]}" for r in stuck)
        rec.state("working", message=f"re-planning, because {why}")
        done = frozenset(sid for sid, r in out.results.items() if r.ok)
        try:
            new, attempts = await make_plan(
                task,
                cards,
                catalogue,
                chat,
                earlier=replan_brief(out.plan, out.results),
                done=done,
                taken=frozenset(out.results),
                redo=frozenset(r.step.skill for r in stuck),
                tokens=out.tokens,
            )
        except PlanError as exc:
            rec.state("working", message=f"re-plan failed, keeping the first plan: {exc}")
            break
        rec.event(
            "llm_response",
            step=model_step,
            content="Re-plan:\n" + json.dumps({"steps": [s.to_dict() for s in new]}, indent=2),
            tool_calls=[
                {
                    "name": f"{s.id} {s.skill} -> {catalogue[s.skill][0].agent}",
                    "arguments": {"input": s.input, "depends_on": s.depends_on},
                }
                for s in new
            ],
            prompt_tokens=sum(a["prompt_tokens"] for a in attempts),
            completion_tokens=sum(a["completion_tokens"] for a in attempts),
        )
        # The new steps take over from every step that didn't complete.
        for r in out.results.values():
            if not r.ok:
                r.message = f"{r.state}, replaced by a re-plan: {r.message}"
                r.state = "replaced"
        out.replans += 1
        rec.state("working", message="re-plan: " + "; ".join(f"{s.id} {s.skill}" for s in new))
        out.plan = [*out.plan, *new]
        kept = {sid: r for sid, r in out.results.items() if r.ok}
        out.results.update(
            await run_steps(new, catalogue, http, rec, policy, task, done=kept, fmt=fmt)
        )

    rec.state("working", message="combining the results")
    try:
        reply = await combine(task, out.plan, out.results, chat, writes_off, tokens=out.tokens)
    except Exception as exc:
        return finish("failed", f"combining failed: {type(exc).__name__}: {exc}", stop="error")
    out.answer = reply.content.strip()
    if missing := values_left_out(task, out.plan, out.results, catalogue, out.answer):
        # Asked once more with the values named; the second reply is kept only if it
        # leaves out fewer of them.
        rec.state(
            "working",
            message=f"the reply left out looked-up values ({', '.join(missing)}), asking again",
        )
        try:
            again = await combine(
                task, out.plan, out.results, chat, writes_off, missing, tokens=out.tokens
            )
        except Exception as exc:
            rec.state("working", message=f"asking again failed, keeping the first reply: {exc}")
        else:
            still = values_left_out(task, out.plan, out.results, catalogue, again.content)
            if len(still) < len(missing):
                out.answer, reply = again.content.strip(), again
            if still:
                rec.state("working", message=f"the reply still leaves out: {', '.join(still)}")
    if ungrounded := ungrounded_citations(task, out.answer, out.results):
        # Last line of defence on the combiner's synthesis: drop a citation no step result
        # carries (so no agent fetched it) and the task did not give. Research guards its
        # own replies, but the combined answer was unguarded; see ``ungrounded_citations``.
        out.answer = strip_citations(out.answer, ungrounded)
        rec.event("citations_stripped", flagged=ungrounded, content=reply.content)
        rec.state("working", message=f"dropped unsupported citations: {', '.join(ungrounded)}")
    if records := write_records(out.plan, out.results):
        # Built by the Knowledge agent's code from add_fact's results, so the user sees
        # what was really stored even when the summary calls a duplicate "stored".
        out.answer = f"{out.answer}\n\n{records}"
    rec.event(
        "llm_response",
        step=model_step + 1,
        content=out.answer,
        tool_calls=[],
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
    )
    bad = [r for r in out.results.values() if not r.ok and r.state != "replaced"]
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
