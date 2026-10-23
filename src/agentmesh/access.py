"""M5: what each caller may ask an agent to do.

An authenticated request names the skill it wants (``"skill"`` in the message metadata;
an agent with one skill assumes it). The agent then checks, from its own spec:

1. **Who may call it:** the skill's ``callers``. Anyone else gets ``rejected``.
2. **Signed input, where the skill needs it:** with ``require_attestation = "verifier"``
   the request must carry at least one result the Verifier signed (``"attestations"`` in
   the metadata). A missing, forged, altered or stale one is ``rejected``.
3. **What the task may touch:** the skill's ``tools`` is the only toolset the task gets,
   and every call of an ``attested_tools`` tool must cite a source URL that a signed
   verdict marked SUPPORTED, at most once per supporting claim.

So the Knowledge agent's write skill only stores facts whose source the Verifier signed
off on, whoever relays the request, and its lookup skill can't write at all. The result is
a ``Grant`` that the executor hands the runner (through ``GRANT``) for the third check.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from agentmesh.auth import (
    AttestationError,
    AuthConfig,
    norm_url,
    sign_attestation,
    supported_claims,
    supported_sources,
    verify_attestation,
)
from agentmesh.numbers import TRIVIAL, numbers
from agentmesh.spec import AgentSpec, SkillSpec

# Most signed results one request may carry.
MAX_ATTESTATIONS = 8

# Heads the code-built list of a task's writes at the end of its answer.
WRITE_RECORD_HEADER = "Write record (from the tool results):"


class Refused(Exception):
    """The caller may not do this; the message says why, and is shown to the caller."""


def tool_matches(tool_name: str, wanted: str) -> bool:
    """MCP tools are registered as ``<server>__<tool>``; accept either form."""
    return tool_name == wanted or tool_name.endswith(f"__{wanted}")


def source_of(args: Any) -> str:
    """The source URL a write call cites: ``details.source`` or ``source``."""
    if hasattr(args, "model_dump"):
        args = args.model_dump()
    if not isinstance(args, dict):
        return ""
    details = args.get("details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = None
    if isinstance(details, dict) and isinstance(details.get("source"), str):
        return details["source"]
    src = args.get("source")
    return src if isinstance(src, str) else ""


_WORDS = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
_STOP = frozenset(
    {
        *("the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has"),
        *("have", "had", "its", "it's", "into", "onto", "than", "then", "can", "could"),
        *("will", "would", "should", "may", "might", "also", "which", "who", "what", "when"),
        *("where", "how", "does", "did", "not", "but", "all", "any", "each", "per", "via"),
        *("about", "over", "under", "more", "most", "such", "only", "own", "same", "other"),
        *("their", "they", "them", "his", "her", "our", "your", "you", "one"),
    }
)
# Share of a statement's content words that must be in what the Verifier checked.
MATCH_SHARE = 0.6


def statement_matches(statement: str, evidence: str) -> bool:
    """Whether ``statement`` says what ``evidence`` (a claim the Verifier supported, with
    its quote) says: every number in it is in the evidence, and at least ``MATCH_SHARE`` of
    its content words are too (a shared 5-letter prefix counts, so "supports" matches
    "support"). Rewording passes; a different claim under the same URL doesn't."""
    want = {v for v in numbers(statement) if v not in TRIVIAL}
    if not want <= set(numbers(evidence)):
        return False
    have = set(_WORDS.findall(evidence.lower()))
    stems = {w[:5] for w in have if len(w) >= 5}
    content = [
        w
        for w in _WORDS.findall(statement.lower())
        if len(w) >= 3 and w not in _STOP and not w.replace(".", "").replace(",", "").isdigit()
    ]
    if not content:
        return False
    hits = sum(w in have or (len(w) >= 5 and w[:5] in stems) for w in content)
    return hits >= MATCH_SHARE * len(content)


@dataclass
class WriteRecord:
    """One attested write: the statement, and the id and ``new`` flag the tool returned
    (``new`` is False for a duplicate, None when the result couldn't be read)."""

    statement: str
    id: int | None = None
    new: bool | None = None


def read_write_result(args: Any, result: Any) -> WriteRecord:
    """Build a ``WriteRecord`` from an add_fact-style call and its result (a dict, or the
    JSON text an MCP tool returns)."""
    if hasattr(args, "model_dump"):
        args = args.model_dump()
    statement = ""
    if isinstance(args, dict):
        statement = str(args.get("statement") or args.get("text") or "")
    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            data = None
    rec = WriteRecord(" ".join(statement.split()))
    if isinstance(data, dict):
        if isinstance(data.get("id"), int):
            rec.id = data["id"]
        if isinstance(data.get("new"), bool):
            rec.new = data["new"]
    return rec


@dataclass
class Grant:
    """What one admitted task may do."""

    caller: str
    skill: str
    tools: list[str] | None = None
    attested_tools: list[str] = field(default_factory=list)
    # Remaining uses per supported source URL (normalised).
    budget: Counter[str] = field(default_factory=Counter)
    # Per supported URL, what the Verifier checked with it (claim + quote), one entry per
    # claim not yet stored. A write must state one of them, not just cite the URL.
    claims: dict[str, list[str]] = field(default_factory=dict)
    _spent: dict[str, list[str]] = field(default_factory=dict, repr=False)
    signed: list[str] = field(default_factory=list)  # the signed tasks' ids
    # One entry per attested write that went through: what the tool said it did.
    writes: list[WriteRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, entry: WriteRecord) -> None:
        with self._lock:
            self.writes.append(entry)

    def write_report(self) -> str:
        """The task's writes as the tools reported them, for the end of the answer. The
        model's own wording can call a duplicate "stored"; this list is built from the
        tool results, so it can't."""
        if not self.writes:
            return ""
        lines = [WRITE_RECORD_HEADER]
        for w in self.writes:
            ident = f"id {w.id}" if w.id is not None else "no id"
            what = "stored as new" if w.new else "already stored, not added again"
            if w.new is None:
                what = "result unclear"
            lines.append(f"- {ident}: {what}: {w.statement[:200]}")
        return "\n".join(lines)

    def allows_tool(self, name: str) -> bool:
        return self.tools is None or any(tool_matches(name, t) for t in self.tools)

    def is_attested(self, name: str) -> bool:
        return any(tool_matches(name, t) for t in self.attested_tools)

    def spend(self, tool: str, args: Any) -> tuple[str, str | None]:
        """Take one use of the cited source. Returns (url, None) or (url, why not)."""
        src = source_of(args)
        url = norm_url(src) if src else ""
        listed = ", ".join(sorted(self.budget)) or "none"
        with self._lock:
            if not url:
                return url, (
                    f"refused: {tool} must cite its source as details.source, one of the "
                    f"URLs the Verifier marked SUPPORTED: {listed}"
                )
            if self.budget[url] <= 0:
                why = (
                    "every claim the Verifier supported with it is already stored"
                    if url in self.budget
                    else "the Verifier did not mark any claim with this source SUPPORTED"
                )
                return url, f"refused: {src}: {why}. Sources still allowed: {listed}"
            if url in self.claims:
                statement = read_write_result(args, None).statement
                left = self.claims[url]
                match = next((e for e in left if statement_matches(statement, e)), None)
                if match is None:
                    checked = "; ".join(f'"{" ".join(e.split())[:160]}"' for e in left)
                    return url, (
                        f"refused: the Verifier did not check this statement with {src}. "
                        f"What it supported with that source: {checked}. Store one of those "
                        "claims as it was verified."
                    )
                left.remove(match)
                self._spent.setdefault(url, []).append(match)
            self.budget[url] -= 1
            return url, None

    def refund(self, url: str) -> None:
        with self._lock:
            self.budget[url] += 1
            if self._spent.get(url):
                self.claims[url].append(self._spent[url].pop())


# The current task's grant (None when the agent runs without auth). Set by the executor;
# asyncio.to_thread copies it into the worker thread, like executor.TASK_ID.
GRANT: ContextVar[Grant | None] = ContextVar("agentmesh_grant", default=None)


class Access:
    """One agent's access rules, built from its spec and the auth folder."""

    def __init__(self, spec: AgentSpec, auth: AuthConfig) -> None:
        self.spec = spec
        self.skills: dict[str, SkillSpec] = {s.id: s for s in spec.skills}
        issuers = {s.require_attestation for s in spec.skills if s.require_attestation}
        # Loaded now, so a missing key fails at startup rather than on the first request.
        self.public_keys = {name: auth.public_key(name) for name in sorted(issuers)}
        self.signer = spec.attest_as
        self.signing_key = auth.signing_key(spec.attest_as) if spec.attest_as else None

    def admit(self, caller: str, skill_id: str, attestations: list[str]) -> Grant:
        if not caller:
            raise Refused("the request is not logged in")
        if not skill_id:
            if len(self.skills) != 1:
                ids = ", ".join(self.skills)
                raise Refused(f"say which skill you want (metadata 'skill'): one of {ids}")
            skill_id = next(iter(self.skills))
        skill = self.skills.get(skill_id)
        if skill is None:
            raise Refused(f"{self.spec.name} has no skill {skill_id!r}")
        if caller not in skill.callers:
            raise Refused(f"{caller!r} may not use {skill_id}; allowed: {', '.join(skill.callers)}")
        grant = Grant(caller, skill_id, skill.tools, list(skill.attested_tools))
        if skill.require_attestation:
            self._check_signed(skill, attestations, grant)
        return grant

    def _check_signed(self, skill: SkillSpec, attestations: list[str], grant: Grant) -> None:
        issuer = skill.require_attestation
        assert issuer is not None
        if not attestations:
            raise Refused(
                f"{skill.id} only accepts work the {issuer} signed, and this request carries "
                f"no signed {issuer} result"
            )
        problems = []
        for token in attestations[:MAX_ATTESTATIONS]:
            try:
                payload = verify_attestation(token, self.public_keys[issuer], issuer)
            except AttestationError as exc:
                problems.append(str(exc))
                continue
            grant.signed.append(str(payload.get("task", "")))
            grant.budget.update(supported_sources(str(payload.get("answer", ""))))
            for url, evidence in supported_claims(
                str(payload.get("input", "")), str(payload.get("answer", ""))
            ):
                grant.claims.setdefault(url, []).append(evidence)
        if not grant.signed:
            raise Refused(f"{skill.id}: no valid {issuer} signature ({'; '.join(problems)})")
        if skill.attested_tools and not grant.budget:
            raise Refused(f"{skill.id}: the {issuer} marked no claim with a source SUPPORTED")

    def sign(self, *, task: str, caller: str, input: str, answer: str) -> str | None:
        if self.signing_key is None or self.signer is None:
            return None
        return sign_attestation(
            self.signing_key,
            self.signer,
            agent=self.spec.name,
            task=task,
            caller=caller,
            input=input,
            answer=answer,
        )
