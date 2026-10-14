"""M3: the orchestrator, end to end in-process: real A2A servers, a scripted planner model."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest

from agentmesh.executor import RunOutcome
from agentmesh.journal import JOURNAL_NAME, JsonlJournal
from agentmesh.orchestrator import (
    ChatReply,
    PlanError,
    build_catalogue,
    describe_catalogue,
    discover_all,
    fill,
    make_plan,
    orchestrate,
    parse_plan,
    plan_schema,
    step_input,
)
from agentmesh.server import build_app
from agentmesh.spec import AgentSpec, SkillSpec
from agentmesh.watch import AgentInfo, Watcher

# --- fixtures --------------------------------------------------------------------------


def spec(name: str, port: int, skill: str, tmp_path: Path) -> AgentSpec:
    return AgentSpec(
        name=name,
        description=f"{name} for tests",
        port=port,
        workspace=tmp_path / name,
        skills=[SkillSpec(id=skill, name=skill, description=f"does {skill}")],
    )


class Fleet:
    """Agents served in-process; ``http`` routes each agent's URL to its app."""

    def __init__(self, tmp_path: Path, agents: dict[str, tuple[int, str, object]]) -> None:
        self.received: dict[str, list[str]] = {}
        self.specs: dict[str, AgentSpec] = {}
        mounts = {}
        for name, (port, skill, runner) in agents.items():
            s = spec(name, port, skill, tmp_path)
            self.specs[name] = s
            journal = JsonlJournal(tmp_path / "traces" / name.lower() / JOURNAL_NAME, name)
            app = build_app(s, self._recording(name, runner), journal)
            mounts[s.url] = httpx.ASGITransport(app=app)
        self.http = httpx.AsyncClient(mounts=mounts, timeout=20)
        self.urls = [s.url for s in self.specs.values()]

    def _recording(self, name, runner):
        def run(task, progress, flag):
            self.received.setdefault(name, []).append(task)
            return runner(task, progress, flag)

        return run


def answers(text: str, stop: str = "final_answer"):
    def run(task, progress, flag):
        progress("step 1: working")
        return RunOutcome(answer=text, stop_reason=stop, steps=1, prompt_tokens=10)

    return run


class ScriptedChat:
    """Replies in order; records every request."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[dict[str, str]], dict | None]] = []

    async def __call__(self, messages, schema=None):
        self.calls.append((messages, schema))
        return ChatReply(self.replies.pop(0), prompt_tokens=100, completion_tokens=10)


def plan(*steps: dict) -> str:
    return json.dumps({"steps": list(steps)})


def step(sid: str, skill: str, text: str, deps=()) -> dict:
    return {"id": sid, "skill": skill, "input": text, "depends_on": list(deps)}


CAT_SKILLS = ["web_research", "claim_verification", "knowledge_write"]


@pytest.fixture
def three(tmp_path):
    return Fleet(
        tmp_path,
        {
            "Research": (9001, "web_research", answers("1. Claim A [https://a.example]")),
            "Verifier": (9002, "claim_verification", answers("1. SUPPORTED - claim A")),
            "Knowledge": (9003, "knowledge_write", answers("stored fact f1")),
        },
    )


async def cards_of(fleet: Fleet):
    cards, errors = await discover_all(fleet.urls, fleet.http)
    assert errors == []
    return cards


# --- plans -----------------------------------------------------------------------------


async def test_catalogue_comes_from_the_cards(three):
    cards = await cards_of(three)
    cat = build_catalogue(cards)
    assert sorted(cat) == sorted(CAT_SKILLS)
    assert cat["claim_verification"][0].agent == "Verifier"
    text = describe_catalogue(cards)
    assert "Agent: Research." in text and "skill `knowledge_write`" in text
    assert plan_schema(sorted(cat))["properties"]["steps"]["items"]["properties"]["skill"][
        "enum"
    ] == sorted(CAT_SKILLS)


async def test_unreachable_agent_is_reported_not_fatal(three):
    cards, errors = await discover_all([*three.urls, "http://127.0.0.1:9"], three.http)
    assert len(cards) == 3
    assert len(errors) == 1 and errors[0].startswith("http://127.0.0.1:9")


def _cat(*skills):
    return {s: [object()] for s in skills}


def test_references_become_dependencies():
    steps = parse_plan(
        plan(step("s1", "a", "go"), step("s2", "b", "check {{s1}} and {{ s1 }}")), _cat("a", "b")
    )
    assert steps[1].depends_on == ["s1"]


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("not json", "not valid JSON"),
        ('{"steps": []}', "non-empty"),
        (plan(step("s1", "zzz", "x")), "unknown skill 'zzz'"),
        (plan(step("step1", "a", "x")), "id must be"),
        (plan(step("s1", "a", "x"), step("s1", "a", "y")), "two steps have the id s1"),
        (plan(step("s1", "a", "")), "empty input"),
        (plan(step("s1", "a", "use {{s2}}")), "s2, which is not a step"),
        (plan(step("s1", "a", "x", ["s1"])), "depends on itself"),
        (plan(step("s1", "a", "{{s2}}"), step("s2", "a", "{{s1}}")), "cycle"),
        (plan(*[step(f"s{i}", "a", "x") for i in range(1, 8)]), "limit is 6"),
    ],
)
def test_bad_plans_are_refused(content, error):
    with pytest.raises(PlanError, match=error):
        parse_plan(content, _cat("a"))


async def test_bad_plan_is_sent_back_with_the_error(three):
    cards = await cards_of(three)
    chat = ScriptedChat(
        plan(step("s1", "web_search", "x")),  # not a catalogue skill
        plan(step("s1", "web_research", "Research X. Return claims with URLs.")),
    )
    steps, attempts = await make_plan("task", cards, build_catalogue(cards), chat)
    assert [s.skill for s in steps] == ["web_research"]
    assert "unknown skill" in attempts[0]["error"]
    retry = chat.calls[1][0][-1]["content"]
    assert retry.startswith("That plan can't run: step s1: unknown skill 'web_search'")


async def test_gives_up_after_three_bad_plans(three):
    cards = await cards_of(three)
    chat = ScriptedChat("nope", "nope", "nope")
    with pytest.raises(PlanError, match="no valid plan after 3"):
        await make_plan("task", cards, build_catalogue(cards), chat)


# --- running ---------------------------------------------------------------------------


CHAIN = plan(
    step("s1", "web_research", "Research local LLM inference. Return claims with URLs."),
    step("s2", "claim_verification", "Check each claim against its source: {{s1}}"),
    step("s3", "knowledge_write", "Store only the SUPPORTED claims: {{s2}}"),
)


async def test_chain_routes_by_card_and_passes_results_on(three, tmp_path):
    cards = await cards_of(three)
    chat = ScriptedChat(CHAIN, "Final: claim A is supported and stored as f1.")
    out = await orchestrate(
        "Research local LLM inference and update my knowledge base",
        cards,
        chat,
        three.http,
        trace_root=tmp_path / "traces",
    )
    assert out.state == "completed", out.message
    assert out.answer == "Final: claim A is supported and stored as f1."
    assert [out.results[s].agent for s in ("s1", "s2", "s3")] == [
        "Research",
        "Verifier",
        "Knowledge",
    ]
    # Each agent got its step's text, with the earlier answer filled in.
    assert three.received["Verifier"] == [
        "Check each claim against its source: 1. Claim A [https://a.example]"
    ]
    assert three.received["Knowledge"] == [
        "Store only the SUPPORTED claims: 1. SUPPORTED - claim A"
    ]
    # The combining model saw every step's result.
    combined = chat.calls[1][0][-1]["content"]
    assert "s3: knowledge_write by Knowledge - completed" in combined
    assert "stored fact f1" in combined
    d = out.to_dict()
    assert [s["state"] for s in d["steps"]] == ["completed"] * 3
    assert all(s["task_id"] for s in d["steps"])


async def test_tasks_are_tagged_and_watch_links_them(three, tmp_path):
    cards = await cards_of(three)
    root = tmp_path / "traces"
    out = await orchestrate("t", cards, ScriptedChat(CHAIN, "done"), three.http, trace_root=root)
    # Every agent's journal records which orchestration step sent the task.
    for name, sid in (("research", "s1"), ("verifier", "s2"), ("knowledge", "s3")):
        recs = [json.loads(x) for x in (root / name / JOURNAL_NAME).read_text().splitlines()]
        sub = next(r for r in recs if r["state"] == "submitted")
        assert (sub["orchestration"], sub["step"]) == (out.run_id, sid)
        assert sub["task"] == out.results[sid].task_id

    state = Watcher(root, [AgentInfo("research", "Research")]).state()
    assert state["agents"][0]["key"] == "orchestrator"  # its lane comes first
    orch = state["agents"][0]["tasks"][0]
    assert orch["id"] == out.run_id and orch["state"] == "completed"
    kinds = [s["kind"] for s in orch["steps"]]
    assert kinds == ["start", "model", "tool", "tool", "tool", "model", "end"]
    assert [c["name"] for c in orch["steps"][1]["tool_calls"]] == [
        "s1 web_research -> Research",
        "s2 claim_verification -> Verifier",
        "s3 knowledge_write -> Knowledge",
    ]
    child = next(a for a in state["agents"] if a["key"] == "verifier")["tasks"][0]
    assert (child["orchestration"], child["step"]) == (out.run_id, "s2")


async def test_failed_step_skips_its_dependents(tmp_path):
    fleet = Fleet(
        tmp_path,
        {
            "Research": (9001, "web_research", answers("half", stop="max_steps")),
            "Verifier": (9002, "claim_verification", answers("unused")),
            "Knowledge": (9003, "knowledge_write", answers("unused")),
        },
    )
    cards = await cards_of(fleet)
    chat = ScriptedChat(CHAIN, "Research failed, nothing was stored.")
    out = await orchestrate("t", cards, chat, fleet.http)
    assert out.state == "failed"
    assert [out.results[s].state for s in ("s1", "s2", "s3")] == ["failed", "skipped", "skipped"]
    assert out.results["s2"].message == "needs s1 (failed)"
    assert out.results["s3"].message == "needs s2 (skipped)"
    assert "Verifier" not in fleet.received and "Knowledge" not in fleet.received
    assert "s1 failed, s2 skipped, s3 skipped" in out.message
    assert out.answer == "Research failed, nothing was stored."  # still reported


async def test_independent_steps_run_at_the_same_time(tmp_path):
    both = threading.Barrier(2, timeout=10)  # breaks unless both runs are in flight at once

    def meet(text):
        def run(task, progress, flag):
            both.wait()
            return RunOutcome(answer=text, stop_reason="final_answer")

        return run

    fleet = Fleet(
        tmp_path,
        {
            "Research": (9001, "web_research", meet("r")),
            "Analyst": (9002, "computation", meet("a")),
        },
    )
    cards = await cards_of(fleet)
    chat = ScriptedChat(
        plan(step("s1", "web_research", "find X"), step("s2", "computation", "compute Y")), "ok"
    )
    out = await orchestrate("t", cards, chat, fleet.http)
    assert out.state == "completed", out.message


async def test_plan_only_sends_nothing(three):
    cards = await cards_of(three)
    out = await orchestrate("t", cards, ScriptedChat(CHAIN), three.http, plan_only=True)
    assert out.state == "completed"
    assert [s["skill"] for s in json.loads(out.answer)["steps"]] == CAT_SKILLS
    assert three.received == {}


async def test_no_agents_fails_cleanly():
    async with httpx.AsyncClient() as http:
        out = await orchestrate("t", [], ScriptedChat(), http)
    assert (out.state, out.message) == ("failed", "no agents with skills were found")


def test_fill_replaces_every_reference():
    from agentmesh.orchestrator import Step, StepResult

    r = StepResult(Step("s1", "a", "x"), answer="  ANSWER \n")
    assert fill("a {{s1}} b {{ s1 }}", {"s1": r}) == "a ANSWER b ANSWER"


def test_step_input_appends_dependencies_the_input_never_references():
    from agentmesh.orchestrator import Step, StepResult

    s1 = StepResult(Step("s1", "a", "go"), state="completed", answer=" claim A [1] ")
    s2 = StepResult(Step("s2", "a", "go"), state="completed", answer="claim B")
    both = {"s1": s1, "s2": s2}

    # Listed in depends_on but not written into the input: appended, labelled.
    out = step_input(Step("s3", "b", "Verify the claims.", ["s1"]), both)
    assert out == "Verify the claims.\n\nResult of step s1:\nclaim A [1]"

    # Referenced inline: substituted once, not appended again.
    out = step_input(Step("s3", "b", "Check {{s1}}; also see", ["s1", "s2"]), both)
    assert out == "Check claim A [1]; also see\n\nResult of step s2:\nclaim B"
