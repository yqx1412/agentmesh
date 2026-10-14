"""M2: the four specialist specs, per-agent tools, the answer guard, webtools and the fleet."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from agentos.config import AgentOSConfig, load_config
from agentos.llm import ChatResponse, Message, ToolCall
from mcp.server.mcpserver.exceptions import ToolError

from agentmesh import fleet
from agentmesh.runtime import (
    AnswerGuardLLM,
    RequireToolsLLM,
    invented_quotes,
    missing_calls,
    select_tools,
    system_prompt,
)
from agentmesh.spec import AgentSpec, SkillSpec, SpecError, load_spec
from agentmesh.webtools import check_url, fetch_page, page_window

AGENTS = Path(__file__).parents[1] / "agents"
SKILL = SkillSpec(id="x", name="X", description="x")


# --- specs -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", fleet.DEFAULT_FLEET)
def test_specialist_specs_load(name):
    spec = load_spec(AGENTS / f"{name}.toml")
    assert spec.host == "127.0.0.1"
    assert spec.instructions
    assert spec.agentos_config is not None and spec.agentos_config.exists()
    load_config(spec.agentos_config)  # the MCP config parses too


def test_fleet_ports_are_distinct():
    ports = [load_spec(p).port for p in fleet.default_specs(AGENTS)]
    assert len(set(ports)) == len(ports)


def test_research_requires_fetches():
    assert load_spec(AGENTS / "research.toml").require_before_answer == {"fetch_url": 2}


def test_sandbox_tools_need_dangerous(tmp_path):
    with pytest.raises(ValueError, match="dangerous"):
        AgentSpec(
            name="a",
            description="a",
            port=1,
            workspace=tmp_path,
            skills=[SKILL],
            builtin_tools=["run_python"],
            allow="write",
        )


def test_require_counts_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        AgentSpec(
            name="a",
            description="a",
            port=1,
            workspace=tmp_path,
            skills=[SKILL],
            require_before_answer={"fetch_url": 0},
        )


def test_select_tools_keeps_spec_order(tmp_path):
    spec = AgentSpec(
        name="a",
        description="a",
        port=1,
        workspace=tmp_path,
        skills=[SKILL],
        builtin_tools=["calculator", "read_file"],
    )
    assert [t.name for t in select_tools(spec, AgentOSConfig())] == ["calculator", "read_file"]


def test_system_prompt_appends_instructions(tmp_path):
    spec = AgentSpec(
        name="a",
        description="a",
        port=1,
        workspace=tmp_path,
        skills=[SKILL],
        instructions="You are the Analyst.",
    )
    assert system_prompt(spec).endswith("You are the Analyst.")


# --- the answer guard --------------------------------------------------------------------


def tool_msg(name: str, content: str = "ok") -> Message:
    return Message(role="tool", content=content, tool_name=name)


def test_missing_calls_counts_only_successes():
    history = [tool_msg("web__fetch_url"), tool_msg("web__fetch_url", "ERROR: HTTP 404")]
    assert missing_calls(history, {"fetch_url": 2}) == {"fetch_url": 1}
    assert missing_calls([*history, tool_msg("fetch_url")], {"fetch_url": 2}) == {}
    # A name that merely ends the same way is not the tool.
    assert missing_calls([tool_msg("web__prefetch_url")], {"fetch_url": 1}) == {"fetch_url": 1}


class ScriptedLLM:
    model = "scripted"

    def __init__(self, replies: list[Message]) -> None:
        self.replies = replies
        self.seen: list[list[Message]] = []

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        return ChatResponse(message=self.replies.pop(0))


ANSWER = Message(role="assistant", content="the answer")
FETCH = Message(role="assistant", tool_calls=[ToolCall(name="web__fetch_url", arguments={})])


def test_guard_sends_early_answer_back():
    inner = ScriptedLLM([ANSWER, FETCH])
    resp = RequireToolsLLM(inner, {"fetch_url": 2}).chat([Message(role="user", content="q")], [])
    assert resp.message.tool_calls  # the model was asked again and now fetches
    nudge = inner.seen[1][-1]
    assert nudge.role == "user" and "2 more successful fetch_url" in nudge.content


def test_guard_passes_answer_once_satisfied():
    inner = ScriptedLLM([ANSWER])
    history = [tool_msg("web__fetch_url"), tool_msg("web__fetch_url")]
    assert RequireToolsLLM(inner, {"fetch_url": 2}).chat(history, []).message is ANSWER
    assert len(inner.seen) == 1


def test_guard_gives_up_after_max_nudges():
    inner = ScriptedLLM([ANSWER] * 4)
    guard = RequireToolsLLM(inner, {"fetch_url": 1}, max_nudges=3)
    assert guard.chat([], []).message.content == "the answer"
    assert len(inner.seen) == 4
    # The budget is per task: later calls don't nudge again.
    inner.replies = [ANSWER]
    guard.chat([], [])
    assert len(inner.seen) == 5


PAGE = tool_msg(
    "web__fetch_url", "# Flags\n`--spec-draft-model FNAME`: draft model for\n speculative decoding."
)


def test_invented_quotes_tolerates_formatting_but_not_paraphrase():
    good = 'SUPPORTED - "--spec-draft-model FNAME: draft model for speculative decoding" [u]'
    assert invented_quotes(good, [PAGE]) == []
    bad = 'SUPPORTED - "--spec-draft-model gives the path to the draft model" [u]'
    assert invented_quotes(bad, [PAGE]) == ["--spec-draft-model gives the path to the draft model"]
    # Short quotes (a flag name) aren't checked, and failed fetches are not evidence.
    assert invented_quotes('the "--spec-type" flag', []) == []
    failed = tool_msg("web__fetch_url", "ERROR: draft model for speculative decoding")
    assert invented_quotes('"draft model for speculative decoding"', [failed]) != []


def test_quote_guard_sends_invented_quote_back():
    fake = Message(role="assistant", content='1. SUPPORTED - "the page says something else"')
    real = Message(
        role="assistant", content='1. SUPPORTED - "draft model for speculative decoding"'
    )
    inner = ScriptedLLM([fake, real])
    resp = AnswerGuardLLM(inner, check_quotes=True).chat([PAGE], [])
    assert resp.message is real
    assert "not in any page you fetched" in inner.seen[1][-1].content


def test_quote_guard_off_by_default():
    fake = Message(role="assistant", content='"the page says something else entirely"')
    assert AnswerGuardLLM(ScriptedLLM([fake])).chat([PAGE], []).message is fake


# --- webtools ----------------------------------------------------------------------------


def resolver(addr: str):
    return lambda host, port: [addr]


@pytest.mark.parametrize(
    "addr",
    ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1", "::1", "fe80::1", "::ffff:127.0.0.1"],
)
def test_check_url_refuses_non_public(addr):
    with pytest.raises(ToolError, match="non-public"):
        check_url("http://example.test/", resolver(addr))


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://example.com/", "http://user:pw@example.com/", "http:///x"]
)
def test_check_url_refuses_bad_urls(url):
    with pytest.raises(ToolError):
        check_url(url, resolver("93.184.215.14"))


def test_fetch_refuses_redirect_to_local():
    def handler(request):
        if request.url.host == "public.test":
            return httpx.Response(302, headers={"location": "http://local.test/admin"})
        return httpx.Response(200, text="secret")

    addrs = {"public.test": "93.184.215.14", "local.test": "127.0.0.1"}
    with pytest.raises(ToolError, match="non-public"):
        fetch_page(
            "http://public.test/",
            resolve=lambda host, port: [addrs[host]],
            transport=httpx.MockTransport(handler),
        )


def test_fetch_extracts_text_and_windows():
    def handler(request):
        return httpx.Response(200, text="hello " * 1000, headers={"content-type": "text/plain"})

    page = fetch_page(
        "http://public.test/a.txt",
        resolve=resolver("93.184.215.14"),
        transport=httpx.MockTransport(handler),
    )
    w = page_window(page, 0, 500)
    assert len(w["text"]) == 500 and w["next_offset"] == 500
    assert page_window(page, w["total_chars"] - 10, 500)["next_offset"] is None


# --- fleet -------------------------------------------------------------------------------


def test_plan_rejects_duplicate_ports(tmp_path):
    a = (AGENTS / "analyst.toml").read_text(encoding="utf-8")
    for name in ("a", "b"):
        (tmp_path / f"{name}.toml").write_text(
            # A port nothing listens on, so the duplicate check is what fails (not "in use").
            a.replace('agentos_config = "analyst.agentos.toml"', "").replace(
                "port = 9112", "port = 9189"
            ),
            encoding="utf-8",
        )
    with pytest.raises(SpecError, match="both use port"):
        fleet.plan([tmp_path / "a.toml", tmp_path / "b.toml"], tmp_path / "logs")


def test_guard_sends_empty_answer_back_even_with_no_rules():
    # qwen3:8b once ended a Knowledge task with an empty reply and no tool call (M3 demo).
    inner = ScriptedLLM([Message(role="assistant", content="  "), ANSWER])
    resp = AnswerGuardLLM(inner).chat([Message(role="user", content="q")], [])
    assert resp.message is ANSWER
    assert "reply was empty" in inner.seen[1][-1].content
