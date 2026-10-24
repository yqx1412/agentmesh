"""Agent spec files: one TOML per agent, naming its identity, skills, model and tools.

Example (``agents/general.toml``)::

    name = "General Agent"
    description = "..."
    port = 9101
    model = "qwen3:8b"
    kind = "plain"            # an AgentOS agent kind
    max_steps = 12
    workspace = "workspace/general"
    agentos_config = "agents/general.agentos.toml"   # optional: MCP servers

    [[skills]]
    id = "file_tasks"
    name = "Workspace file tasks"
    description = "..."
    tags = ["files"]
    examples = ["..."]

Relative paths resolve against the spec file's folder.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

BuiltinTool = Literal["read_file", "write_file", "calculator", "run_python", "run_command"]
SANDBOX_TOOLS = ("run_python", "run_command")


class SpecError(Exception):
    """The agent spec is missing or invalid."""


class SkillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    # M5 access rules (enforced only when the agent runs with an auth folder).
    # Principals allowed to ask for this skill.
    callers: list[str] = Field(default_factory=lambda: ["orchestrator", "user"])
    # Tools a task under this skill may use (built-in or MCP names); None = all of them.
    tools: list[str] | None = None
    # A signer (e.g. "verifier") whose signed result must come with the request.
    require_attestation: str | None = None
    # Tools whose calls must cite a source that signed result marked SUPPORTED, at most
    # once per supporting claim (e.g. ["add_fact"]).
    attested_tools: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _attested_needs_signer(self) -> SkillSpec:
        if self.attested_tools and not self.require_attestation:
            raise ValueError(f"skill {self.id}: attested_tools needs require_attestation")
        if not self.callers:
            raise ValueError(f"skill {self.id}: callers is empty, so nobody could use it")
        return self


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    version: str = "0.1.0"
    # Loopback by default. M5 adds logins, but tokens travel over plain HTTP, so the mesh
    # still stays on one machine.
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    model: str = "qwen3:8b"
    ollama_url: str = "http://127.0.0.1:11434"
    num_ctx: int = 16384
    kind: str = "plain"
    max_steps: int = Field(default=12, ge=1)
    # Highest AgentOS tool permission that runs. Nobody can approve a call over A2A, so
    # anything above this is refused, never asked.
    allow: Literal["read", "write", "dangerous"] = "write"
    # AgentOS built-in tools this agent gets. run_python / run_command are the A7 sandbox
    # (limits from the agentos_config's [sandbox] table) and need allow = "dangerous".
    builtin_tools: list[BuiltinTool] = Field(
        default_factory=lambda: ["read_file", "write_file", "calculator"]
    )
    # Appended to AgentOS's system prompt: the agent's role and output format.
    instructions: str | None = None
    # Tool name -> successful calls needed before a final answer is accepted, e.g.
    # {fetch_url = 2}. Small models ignore this when it's only in the prompt, so the
    # runtime enforces it: an early answer is sent back with a reminder (a few times at most).
    require_before_answer: dict[str, int] = Field(default_factory=dict)
    # Every "double-quoted" string of 20+ characters in the final answer must appear in some
    # tool result; an answer with an invented quote is sent back (a few times at most).
    check_quotes: bool = False
    # A tool name (Research: "fetch_url"): every number in the final answer must appear in
    # one of that tool's successful results or in the task, else the answer is sent back.
    check_numbers: str | None = None
    # A tool name (Research: "fetch_url"): every URL the final answer cites must be one that
    # tool fetched successfully (or one the task gives), else the answer is sent back.
    check_citations: str | None = None
    # qwen3's reasoning mode: slower, but needed for multi-step arithmetic.
    think: bool = False
    workspace: Path
    agentos_config: Path | None = None
    # M5: sign every completed result with this signer's key (keys/<name>.key).
    attest_as: str | None = None
    skills: list[SkillSpec] = Field(min_length=1)

    @field_validator("builtin_tools")
    @classmethod
    def _unique_tools(cls, tools: list[str]) -> list[str]:
        if len(tools) != len(set(tools)):
            raise ValueError(f"duplicate builtin tools: {tools}")
        return tools

    @model_validator(mode="after")
    def _sandbox_needs_dangerous(self) -> AgentSpec:
        sandboxed = [t for t in self.builtin_tools if t in SANDBOX_TOOLS]
        if sandboxed and self.allow != "dangerous":
            raise ValueError(f"{sandboxed} are 'dangerous' tools; set allow = \"dangerous\"")
        if self.instructions is not None and self.kind != "plain":
            raise ValueError('instructions are only supported for kind = "plain"')
        if any(n < 1 for n in self.require_before_answer.values()):
            raise ValueError("require_before_answer counts must be at least 1")
        return self

    @field_validator("skills")
    @classmethod
    def _unique_skill_ids(cls, skills: list[SkillSpec]) -> list[SkillSpec]:
        ids = [s.id for s in skills]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate skill ids: {ids}")
        return skills

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_spec(path: Path) -> AgentSpec:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"spec not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"{path}: invalid TOML: {exc}") from exc
    try:
        spec = AgentSpec.model_validate(data)
    except ValidationError as exc:
        raise SpecError(f"{path}: {exc}") from exc
    base = path.resolve().parent
    update = {"workspace": (base / spec.workspace).resolve()}
    if spec.agentos_config is not None:
        update["agentos_config"] = (base / spec.agentos_config).resolve()
    return spec.model_copy(update=update)
