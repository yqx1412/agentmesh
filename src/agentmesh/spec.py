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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class SpecError(Exception):
    """The agent spec is missing or invalid."""


class SkillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    version: str = "0.1.0"
    # Loopback by default: M1 has no authentication (that's M5), so nothing else should reach it.
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
    workspace: Path
    agentos_config: Path | None = None
    skills: list[SkillSpec] = Field(min_length=1)

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
