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
    supported_sources,
    verify_attestation,
)
from agentmesh.spec import AgentSpec, SkillSpec

# Most signed results one request may carry.
MAX_ATTESTATIONS = 8


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


@dataclass
class Grant:
    """What one admitted task may do."""

    caller: str
    skill: str
    tools: list[str] | None = None
    attested_tools: list[str] = field(default_factory=list)
    # Remaining uses per supported source URL (normalised).
    budget: Counter[str] = field(default_factory=Counter)
    signed: list[str] = field(default_factory=list)  # the signed tasks' ids
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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
            self.budget[url] -= 1
            return url, None

    def refund(self, url: str) -> None:
        with self._lock:
            self.budget[url] += 1


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
