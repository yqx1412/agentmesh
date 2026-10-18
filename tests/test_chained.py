"""The three fleet fixes from M6's chained-task failures (benchmarks/results/m6-*.md):

1. every step also receives the user's task, not only the planner's rewording of it;
2. an agent that replies ``CANNOT: <reason>`` has failed its step, so retries, fallback and
   dependency skipping apply instead of passing an apology on as data;
3. skills a card tags ``write`` are planned only when the run allows writes.
"""

from __future__ import annotations

import pytest
from test_m3 import Fleet, ScriptedChat, answers, plan, step

from agentmesh.orchestrator import (
    STEP_CONTEXT,
    RunPolicy,
    build_catalogue,
    cannot_reason,
    describe_catalogue,
    discover_all,
    orchestrate,
    without_writes,
)
from agentmesh.spec import SkillSpec

TASK = "How many Ollama default windows fit in Qwen3-8B's native context?"
NO_WAIT = RunPolicy(retries=1, backoff=0)


async def cards_of(fleet: Fleet):
    cards, errors = await discover_all(fleet.urls, fleet.http)
    assert errors == []
    return cards


# --- 1. the user's task reaches every step ---------------------------------------------


async def test_every_step_receives_the_users_task(tmp_path):
    fleet = Fleet(
        tmp_path,
        {
            "Research": (9001, "web_research", answers("native context 32768, window 4096")),
            "Analyst": (9002, "calculate", answers("8")),
        },
    )
    chat = ScriptedChat(
        plan(
            step("s1", "web_research", "Find both numbers."),
            step("s2", "calculate", "Divide: {{s1}}", ["s1"]),
        ),
        "8",
    )
    out = await orchestrate(TASK, await cards_of(fleet), chat, fleet.http, policy=NO_WAIT)
    assert out.state == "completed", out.message
    context = STEP_CONTEXT.format(task=TASK)
    assert fleet.received["Research"] == ["Find both numbers." + context]
    assert fleet.received["Analyst"] == ["Divide: native context 32768, window 4096" + context]
    # The planner is told the agents see the task; the combiner gets it once, not per step.
    assert "It also gets the user's task" in chat.calls[0][0][0]["content"]
    combined = chat.calls[1][0][-1]["content"]
    assert combined.count(TASK) == 1
    assert "Asked: Divide: native context 32768, window 4096\n" in combined


# --- 2. "I couldn't" fails the step ----------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("CANNOT: the context length is missing", "the context length is missing"),
        ("**CANNOT:** search failed (0xc0000409)", "search failed (0xc0000409)"),
        ("I tried.\n> CANNOT: no head dimension given", "no head dimension given"),
        ("CANNOT:", "no reason given"),
        ("The answer is 8. It cannot: be less.", None),
        ("You cannot store it twice.", None),
        ("8", None),
    ],
)
def test_cannot_reason(answer, reason):
    assert cannot_reason(answer) == reason


async def test_cannot_is_retried_then_falls_back(tmp_path):
    fleet = Fleet(
        tmp_path,
        {
            "Knowledge": (9001, "knowledge_lookup", answers("CANNOT: embedding model crashed")),
            "Knowledge B": (9002, "knowledge_lookup", answers("lecture 1: 4.2x speed-up")),
        },
    )
    chat = ScriptedChat(plan(step("s1", "knowledge_lookup", "Look it up.")), "4.2x")
    out = await orchestrate("t", await cards_of(fleet), chat, fleet.http, policy=NO_WAIT)
    assert out.state == "completed", out.message
    r = out.results["s1"]
    assert [(a.agent, a.state) for a in r.attempts] == [
        ("Knowledge", "failed"),
        ("Knowledge", "failed"),  # the retry
        ("Knowledge B", "completed"),  # the fallback
    ]
    assert r.attempts[0].message == "the agent could not do the step: embedding model crashed"
    assert r.answer == "lecture 1: 4.2x speed-up"


async def test_cannot_skips_the_steps_that_need_it(tmp_path):
    fleet = Fleet(
        tmp_path,
        {
            "Research": (9001, "web_research", answers("layers 36, KV heads 8")),
            "Analyst": (9002, "calculate", answers("CANNOT: the context length is missing")),
            "Writer": (9003, "summarize", answers("unused")),
        },
    )
    chat = ScriptedChat(
        plan(
            step("s1", "web_research", "Find the config."),
            step("s2", "calculate", "KV cache size from {{s1}}", ["s1"]),
            step("s3", "summarize", "Summarize {{s2}}", ["s2"]),
        ),
        "Could not compute: the context length is missing.",
    )
    out = await orchestrate("t", await cards_of(fleet), chat, fleet.http, policy=NO_WAIT)
    assert out.state == "failed"
    assert out.results["s2"].state == "failed" and out.results["s2"].answer == ""
    assert "the context length is missing" in out.results["s2"].message
    assert out.results["s3"].state == "skipped"
    assert "Writer" not in fleet.received  # never sent an apology as its input
    combined = chat.calls[1][0][-1]["content"]
    assert "s2: calculate by Analyst - failed" in combined


# --- 3. writes only when asked for ----------------------------------------------------


WRITE = SkillSpec(id="knowledge_write", name="w", description="stores facts", tags=["write"])


@pytest.fixture
def with_writer(tmp_path):
    return Fleet(
        tmp_path,
        {
            "Research": (9001, "web_research", answers("1. Claim A [https://a.example]")),
            "Knowledge": (9002, WRITE, answers("stored fact f1")),
        },
    )


async def test_write_skills_come_from_the_card_tag(with_writer):
    cat = build_catalogue(await cards_of(with_writer))
    assert cat["knowledge_write"][0].writes and not cat["web_research"][0].writes
    kept, removed = without_writes(cat)
    assert sorted(kept) == ["web_research"] and removed == ["knowledge_write"]
    text = describe_catalogue(await cards_of(with_writer), kept)
    assert "knowledge_write" not in text and "Agent: Knowledge." not in text


async def test_writes_are_off_by_default(with_writer):
    chat = ScriptedChat(
        plan(  # the planner tries a write anyway: refused, sent back
            step("s1", "web_research", "Research A."),
            step("s2", "knowledge_write", "Store {{s1}}", ["s1"]),
        ),
        plan(step("s1", "web_research", "Research A.")),
        "A, with its source. Nothing was stored.",
    )
    out = await orchestrate("Who made A?", await cards_of(with_writer), chat, with_writer.http)
    assert out.state == "completed", out.message
    assert [s.skill for s in out.plan] == ["web_research"]
    assert "Knowledge" not in with_writer.received
    messages, schema = chat.calls[0]
    assert "knowledge_write" not in messages[1]["content"]
    assert schema["properties"]["steps"]["items"]["properties"]["skill"]["enum"] == ["web_research"]
    assert "unknown skill 'knowledge_write'" in chat.calls[1][0][-1]["content"]
    # The combiner is told writing was off, so it can't claim something was stored.
    assert "Writing was not enabled for this run" in chat.calls[2][0][-1]["content"]


async def test_allow_writes_offers_them(with_writer):
    chat = ScriptedChat(
        plan(
            step("s1", "web_research", "Research A."),
            step("s2", "knowledge_write", "Store {{s1}}", ["s1"]),
        ),
        "Stored as f1.",
    )
    out = await orchestrate(
        "Research A and update my knowledge base",
        await cards_of(with_writer),
        chat,
        with_writer.http,
        allow_writes=True,
    )
    assert out.state == "completed", out.message
    assert with_writer.received["Knowledge"][0].startswith("Store 1. Claim A")
    assert "Writing was not enabled" not in chat.calls[1][0][-1]["content"]
