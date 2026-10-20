"""The three fleet fixes from M6's chained-task failures (benchmarks/results/m6-*.md):

1. every step also receives the user's task, not only the planner's rewording of it;
2. an agent that replies ``CANNOT: <reason>`` has failed its step, so retries, fallback and
   dependency skipping apply instead of passing an apology on as data;
3. skills a card tags ``write`` are planned only when the run allows writes.
"""

from __future__ import annotations

import pytest
from test_m3 import Fleet, ScriptedChat, answers, plan, step

from agentmesh.executor import RunOutcome
from agentmesh.orchestrator import (
    STEP_CONTEXT,
    PlanError,
    RunPolicy,
    build_catalogue,
    cannot_reason,
    describe_catalogue,
    discover_all,
    orchestrate,
    parse_plan,
    plan_inputs,
    plan_schema,
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
    # The task's format instructions are for the final reply only (a Research step once
    # ended with the task's "RATIO:" line and the final answer kept it).
    assert "are for the final reply to the user, not for you" in context
    assert "it does not replace the rest of these rules" in chat.calls[1][0][0]["content"]
    # A task that only asks for citations gets no verification step (the M6 rerun's
    # planner added one to every r-qwen-license lookup).
    assert "does not need one: the step that finds" in chat.calls[0][0][0]["content"]


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
    no_replan = RunPolicy(retries=1, backoff=0, replans=0)
    out = await orchestrate("t", await cards_of(fleet), chat, fleet.http, policy=no_replan)
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


# --- 4. re-plan when a step says an input is missing -----------------------------------


def research(task, progress, flag):
    progress("step 1: searching")
    text = "native context 32768 tokens" if "context length" in task else "layers 36, KV heads 8"
    return RunOutcome(answer=text, stop_reason="final_answer", steps=1)


def analyst(task, progress, flag):
    progress("step 1: computing")
    if "32768" not in task.split("\n---\n")[0]:
        return RunOutcome(answer="CANNOT: missing context length", stop_reason="final_answer")
    return RunOutcome(answer="4.5 GiB", stop_reason="final_answer", steps=1)


KV_PLAN = plan(
    step("s1", "web_research", "Find the layers and KV heads."),
    step("s2", "calculate", "KV cache from {{s1}}", ["s1"]),
)
KV_REPLAN = plan(
    step("s3", "web_research", "Find the native context length."),
    step("s4", "calculate", "KV cache from {{s1}} at {{s3}}", ["s1", "s3"]),
)


@pytest.fixture
def kv(tmp_path):
    return Fleet(
        tmp_path,
        {"Research": (9001, "web_research", research), "Analyst": (9002, "calculate", analyst)},
    )


async def test_cannot_triggers_a_replan_that_reuses_finished_work(kv):
    chat = ScriptedChat(KV_PLAN, KV_REPLAN, "4.5 GiB")
    out = await orchestrate("KV cache?", await cards_of(kv), chat, kv.http, policy=NO_WAIT)
    assert out.state == "completed", out.message
    assert out.replans == 1
    assert [s.id for s in out.plan] == ["s1", "s2", "s3", "s4"]
    assert out.results["s2"].state == "replaced"
    assert out.results["s4"].answer == "4.5 GiB"
    # s1 ran once; the re-plan used its result instead of repeating it.
    assert [t.split("\n---\n")[0].strip() for t in kv.received["Research"]] == [
        "Find the layers and KV heads.",
        "Find the native context length.",
    ]
    assert kv.received["Analyst"][-1].startswith(
        "KV cache from layers 36, KV heads 8 at native context 32768 tokens"
    )
    brief = chat.calls[1][0][1]["content"]
    assert "s2 (calculate) - FAILED, the agent replied: CANNOT: missing context length" in brief
    assert "s1 (web_research) - completed. Result:\nlayers 36, KV heads 8" in brief
    assert "start at s3" in brief
    assert out.to_dict()["replans"] == 1


async def test_a_replan_may_not_reuse_or_depend_on_unfinished_steps(kv):
    bad = plan(step("s2", "calculate", "again {{s1}}", ["s1"]))
    worse = plan(step("s3", "calculate", "with {{s2}}", ["s2"]))
    chat = ScriptedChat(KV_PLAN, bad, worse, KV_REPLAN, "4.5 GiB")
    out = await orchestrate("KV cache?", await cards_of(kv), chat, kv.http, policy=NO_WAIT)
    assert out.state == "completed", out.message
    # make_plan sends one growing conversation; the re-plan's is the last one recorded.
    said = [m["content"] for m in chat.calls[3][0] if m["role"] == "user"]
    assert any("s2 is an earlier step; new steps need new ids" in m for m in said)
    assert any("uses s2, which did not complete" in m for m in said)


async def test_a_replan_must_redo_the_failed_skill(kv):
    # Live, a re-plan once only fetched the missing value; the combiner then did the
    # arithmetic itself, wrongly. Such a plan is now sent back.
    fetch_only = plan(step("s3", "web_research", "Find the native context length."))
    chat = ScriptedChat(KV_PLAN, fetch_only, KV_REPLAN, "4.5 GiB")
    out = await orchestrate("KV cache?", await cards_of(kv), chat, kv.http, policy=NO_WAIT)
    assert out.state == "completed", out.message
    said = [m["content"] for m in chat.calls[2][0] if m["role"] == "user"]
    assert any("must redo the work that failed; add a step using calculate" in m for m in said)
    assert out.results["s4"].answer == "4.5 GiB"


async def test_replans_are_bounded(kv):
    still_missing = plan(step("s3", "calculate", "KV cache from {{s1}}", ["s1"]))
    chat = ScriptedChat(KV_PLAN, still_missing, "Could not compute it.")
    out = await orchestrate("KV cache?", await cards_of(kv), chat, kv.http, policy=NO_WAIT)
    assert out.state == "failed" and out.replans == 1
    assert out.results["s3"].state == "failed" and out.results["s3"].cannot
    assert len(chat.calls) == 3  # plan, one re-plan, combine


async def test_other_failures_do_not_replan(tmp_path):
    fleet = Fleet(tmp_path, {"Research": (9001, "web_research", answers("half", stop="max_steps"))})
    chat = ScriptedChat(plan(step("s1", "web_research", "x")), "nothing")
    out = await orchestrate("t", await cards_of(fleet), chat, fleet.http, policy=NO_WAIT)
    assert out.state == "failed" and out.replans == 0
    assert len(chat.calls) == 2


# --- the planner names every input first ----------------------------------------------


def test_plan_schema_asks_for_inputs_before_steps():
    # Constrained decoding writes properties in schema order, so "inputs" must come first:
    # live, listing the values first is what made the KV-cache plan ask for the context
    # length in its first step instead of after a CANNOT and a re-plan.
    schema = plan_schema(["web_research"])
    assert list(schema["properties"]) == ["inputs", "steps"]
    assert schema["required"] == ["inputs", "steps"]
    item = schema["properties"]["inputs"]["items"]
    assert item["required"] == ["value", "source"]


def test_plan_inputs_reads_the_list_and_tolerates_its_absence():
    reply = '{"inputs": [{"value": "native context length", "source": "s1"}, "junk"], "steps": []}'
    assert plan_inputs(reply) == [{"value": "native context length", "source": "s1"}]
    assert plan_inputs('{"steps": []}') == []
    assert plan_inputs("not json") == []
    assert plan_inputs('{"inputs": "s1"}') == []


def _one_step(text: str) -> str:
    return plan(step("s1", "calculate", text))


@pytest.mark.parametrize(
    "text",
    [
        "Compute the KV cache using the formula: layers x heads.",
        "Compute (layers * kv_heads * head_dim * tokens * 2) / (1024^3).",
        "Compute layers \u00d7 kv_heads \u00d7 head_dim.",
        "Divide by 1024**3.",
    ],
)
def test_a_step_input_may_not_spell_out_a_formula(text):
    # Live, the planner wrote the KV-cache formula without the x2 for keys and values; the
    # Analyst followed it and all three runs got the size wrong.
    cat = {"calculate": [object()]}
    with pytest.raises(PlanError, match="formula or method"):
        parse_plan(_one_step(text), cat, task="KV cache of Qwen3-8B?")


def test_names_values_without_a_formula_and_a_formula_from_the_task_pass():
    cat = {"calculate": [object()]}
    ok = "Compute the fp16 KV-cache size in GiB from the layers, KV heads and head dim."
    assert parse_plan(_one_step(ok), cat, task="t")
    # The user's own formula may be passed on (a-weights says "1 GiB = 2^30 bytes").
    task = "How many GiB do 8.2B fp16 weights take (1 GiB = 2^30 bytes)?"
    assert parse_plan(
        _one_step("Compute 8.2B x 2 bytes in GiB, 1 GiB = 2^30 bytes."), cat, task=task
    )
    # Markdown bold is not multiplication.
    assert parse_plan(_one_step("Compute the **KV-cache** size."), cat, task="t")


def test_two_independent_steps_may_not_share_a_skill():
    # The M6 rerun's license lookup became two parallel Research steps, one per question,
    # each a full agent run; one step can ask for both.
    cat = {"web_research": [object()], "calculate": [object()]}
    split = plan(
        step("s1", "web_research", "Find the license."),
        step("s2", "web_research", "Find the native context length."),
    )
    with pytest.raises(PlanError, match="s1 and s2 both use web_research"):
        parse_plan(split, cat)
    # Also when a later step joins them: s2 still doesn't need s1.
    joined = plan(
        step("s1", "web_research", "Find the license."),
        step("s2", "web_research", "Find the native context length."),
        step("s3", "calculate", "Use {{s1}} and {{s2}}.", ["s1", "s2"]),
    )
    with pytest.raises(PlanError, match="neither needs the other's result"):
        parse_plan(joined, cat)


def test_a_skill_may_repeat_when_one_step_needs_the_others_result():
    cat = {"web_research": [object()], "calculate": [object()]}
    direct = plan(
        step("s1", "web_research", "Which model does the page recommend?"),
        step("s2", "web_research", "Find the license of {{s1}}.", ["s1"]),
    )
    assert [s.id for s in parse_plan(direct, cat)] == ["s1", "s2"]
    # Through another step counts too.
    chained = plan(
        step("s1", "web_research", "Find the model's size."),
        step("s2", "calculate", "Pick a GPU for {{s1}}.", ["s1"]),
        step("s3", "web_research", "Find the price of {{s2}}.", ["s2"]),
    )
    assert len(parse_plan(chained, cat)) == 3
    # A re-plan's new step may reuse a skill an earlier (done) step used.
    replan = plan(step("s3", "web_research", "Find the context length."))
    assert parse_plan(replan, cat, done=frozenset({"s1"}), taken=frozenset({"s1", "s2"}))
