"""M6: the single-vs-multi evaluation harness, in-process (real A2A servers, scripted models)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from agentos.agent import Tracer
from agentos.llm import ChatResponse, Message, ToolCall
from test_m3 import Fleet, ScriptedChat, plan, step
from test_m4 import Proc

from agentmesh.evaluate import (
    EvalTask,
    RunRecord,
    finish_record,
    load_results,
    load_tasks,
    normalize_url,
    report,
    run_eval,
    run_multi,
    run_single,
    schedule,
    score,
    select_tasks,
    timestamps_in,
    trace_stats,
    urls_in,
)
from agentmesh.executor import TASK_ID, RunOutcome
from agentmesh.orchestrator import RunPolicy, discover_all
from agentmesh.runtime import AnswerGuardLLM
from agentmesh.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "benchmarks" / "m6" / "tasks.yaml"

# --- the task set ----------------------------------------------------------------------

# A correct answer per task, written by hand from the ground truth in tasks.yaml. Each must
# pass its task's checks, so a typo in a regex can't silently fail every run.
GOLD = {
    "r-qwen-license": "Qwen3-8B is Apache 2.0 licensed; native context 32,768 tokens.",
    "r-a2a-origin": "It was contributed by Google and is now under the Linux Foundation.",
    "r-ollama-port": "Ollama binds 127.0.0.1 port 11434 by default [https://docs.ollama.com/faq].",
    "r-llama-draft": "Use `-md` (also `--model-draft`) to pass the draft model.",
    "r-mcp-definition": "MCP is the Model Context Protocol; the docs liken it to a USB-C port.",
    "a-kv-cache": "The KV cache takes 1.50 GiB.",
    "a-weights": "About 7.70 GiB of weights.",
    "a-latency-stats": "Mean: 250.14 ms. Median: 212 ms.",
    "a-throughput": "In total 59.9 s.",
    "k-radix-analogy": "He calls it the Excel spreadsheet approach (Lecture 7, 45:27-46:50).",
    "k-peak-timing": "The theta(n) one took 13 seconds and the theta(log n) one 0.001 seconds.",
    "k-table-doubling": "Constant amortized time per insert (Lecture 9, 19:28).",
    "v-three-claims": (
        "1. SUPPORTED\n2. CONTRADICTED: contributed by Google\n3. SUPPORTED\nFALSE: 2"
    ),
    "c-qwen-kv": "36 layers, 8 KV heads, head dim 128: 4.5 GiB at 32,768 tokens.",
    "c-ollama-ratio": "Ollama defaults to 4096 tokens; Qwen3-8B has 32,768.\nRATIO: 8",
    "c-peak-speedup": "13 s vs 0.001 s, so it was 13,000 times faster.",
}
# Plausible wrong answers that must fail at least one check.
WRONG = {
    "r-qwen-license": "Qwen3-8B uses the MIT license and a 128k context.",
    "a-kv-cache": "The KV cache takes 0.75 GiB.",  # the missing x2 seen in M2
    "a-latency-stats": "Mean: 250.14 ms. Median: 240 ms.",
    "k-radix-analogy": "It is like an Excel spreadsheet (Lecture 7, 12:00).",  # wrong part
    "k-table-doubling": "Constant amortized per insert.",  # no timestamp at all
    "v-three-claims": "FALSE: 1, 2",
    "c-ollama-ratio": "Ollama defaults to 4096 tokens.\nRATIO: 8 (roughly 4)",
    "c-peak-speedup": "It was 1,300 times faster.",
}


def test_task_set_loads_and_covers_every_category():
    tasks = load_tasks(TASKS)
    assert len(tasks) == 16
    assert {t.category for t in tasks} == {
        "research",
        "analysis",
        "knowledge",
        "verification",
        "composite",
    }
    assert set(GOLD) == {t.id for t in tasks}


@pytest.mark.parametrize("task", load_tasks(TASKS), ids=lambda t: t.id)
def test_gold_answer_passes_every_check(task):
    checks = score(task, GOLD[task.id])
    assert all(c["ok"] for c in checks), checks


@pytest.mark.parametrize("tid", sorted(WRONG))
def test_wrong_answer_fails(tid):
    task = next(t for t in load_tasks(TASKS) if t.id == tid)
    assert not all(c["ok"] for c in score(task, WRONG[tid]))


@pytest.mark.parametrize(
    ("answer", "ok"),
    [
        ("Google contributed it; it is now under the Linux Foundation.", True),
        ("Google contributed it; the Agentic AI Foundation hosts it now.", True),
        ("Google contributed it; the Apache Software Foundation hosts it.", False),
        ("Anthropic contributed it to the Linux Foundation.", False),
    ],
)
def test_a2a_origin_accepts_either_foundation_name(answer, ok):
    task = next(t for t in load_tasks(TASKS) if t.id == "r-a2a-origin")
    assert all(c["ok"] for c in score(task, answer)) is ok


def test_select_by_id_or_category():
    tasks = load_tasks(TASKS)
    assert {t.id for t in select_tasks(tasks, ["k-*"])} == {
        "k-radix-analogy",
        "k-peak-timing",
        "k-table-doubling",
    }
    assert len(select_tasks(tasks, ["composite"])) == 3
    assert select_tasks(tasks, None) == tasks


def test_bad_task_files_are_refused(tmp_path):
    from agentmesh.evaluate import EvalError

    p = tmp_path / "t.yaml"
    p.write_text("tasks:\n  - {id: a, category: x, prompt: p, checks: []}\n", encoding="utf-8")
    with pytest.raises(EvalError):
        load_tasks(p)
    p.write_text(
        "tasks:\n"
        "  - {id: a, category: x, prompt: p, checks: [{type: answer_contains, values: [y]}]}\n"
        "  - {id: a, category: x, prompt: p, checks: [{type: answer_contains, values: [y]}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(EvalError, match="duplicate"):
        load_tasks(p)


def test_solo_spec_has_the_union_of_read_tools_and_no_writes():
    spec = load_spec(ROOT / "agents" / "solo.toml")
    assert spec.port == 9115 and spec.model == "qwen3:14b"
    assert set(spec.builtin_tools) == {"calculator", "run_python"}
    # Guarded like Research: a cited page it never fetched is sent back / stripped.
    assert spec.check_citations == "fetch_url"
    cfg = (ROOT / "agents" / "solo.agentos.toml").read_text(encoding="utf-8")
    assert "web_search" in cfg and "fetch_url" in cfg and '"search"' in cfg
    for write in ("add_fact", "forget_fact", "clear_scope", "ingest_source"):
        assert write not in re.sub(r"#.*", "", cfg)


# --- checks and parsing ----------------------------------------------------------------


def test_timestamps():
    assert timestamps_in("at 45:27-46:50 and 1:02:03") == [2727, 2810, 3723]
    assert timestamps_in("ratio 3:4 and 12:75") == []  # not mm:ss


def test_urls_are_normalized():
    assert normalize_url("HTTPS://Docs.Ollama.com/faq/#x") == "https://docs.ollama.com/faq"
    assert normalize_url("https://a.example/") == "https://a.example"
    text = "See [https://a.example/x]. And (https://b.example/y), https://a.example/x."
    assert urls_in(text) == ["https://a.example/x", "https://b.example/y"]


def _write(path: Path, events: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return path


def test_trace_stats_counts_agents_and_skips_delegations(tmp_path):
    a = _write(
        tmp_path / "research" / "t-1.jsonl",
        [
            {"event": "llm_response", "prompt_tokens": 100, "completion_tokens": 10},
            {
                "event": "tool_result",
                "ok": True,
                "call": {"name": "web__fetch_url", "arguments": {"url": "https://a.example/"}},
            },
            {"event": "tool_result", "ok": False, "call": {"name": "web__fetch_url"}},
        ],
    )
    o = _write(
        tmp_path / "orchestrator" / "t-run.jsonl",
        [
            {"event": "llm_response", "prompt_tokens": 50, "completion_tokens": 5},
            {"event": "tool_result", "ok": True, "call": {"name": "s1 web_research -> R"}},
        ],
    )
    s = trace_stats([a, o])
    assert (s["llm_calls"], s["prompt_tokens"], s["completion_tokens"]) == (2, 150, 15)
    assert (s["tool_calls"], s["tool_errors"]) == (2, 1)  # the delegation is not a tool
    assert s["fetched"] == ["https://a.example"]


def test_trace_stats_counts_the_url_after_a_redirect(tmp_path):
    # NVIDIA's model-card PDF redirects to another host; the answer cited the final URL,
    # and the eval used to report it as an unfetched citation.
    asked = "https://developer.nvidia.com/downloads/assets/card.pdf"
    final = "https://developer.download.nvidia.com/assets/Card.pdf"
    a = _write(
        tmp_path / "research" / "t-1.jsonl",
        [
            {
                "event": "tool_result",
                "ok": True,
                "call": {"name": "web__fetch_url", "arguments": {"url": asked}},
                "output": json.dumps({"url": final, "title": None, "text": "..."}),
            },
            {
                "event": "tool_result",
                "ok": True,
                "call": {"name": "web__fetch_url", "arguments": {"url": "https://b.example/"}},
                "output": "not json",
            },
        ],
    )
    assert trace_stats([a])["fetched"] == [asked, final, "https://b.example"]


def test_trace_stats_tallies_unverified_numbers_and_quotes(tmp_path):
    # When the nudge budget is spent, the guard emits *_unverified for a number/quote it
    # still flagged. These are not model calls of their own (unlike answer_sent_back).
    a = _write(
        tmp_path / "research" / "t-1.jsonl",
        [
            {"event": "llm_response", "prompt_tokens": 100, "completion_tokens": 10},
            {"event": "numbers_unverified", "flagged": ["4,096"], "nudge": 3},
            {"event": "quotes_unverified", "flagged": ['"not on any page"'], "nudge": 3},
        ],
    )
    s = trace_stats([a])
    assert s["unverified"] == {"numbers": 1, "quotes": 1}
    assert s["llm_calls"] == 1  # the unverified flags are not extra model calls


def test_trace_stats_counts_both_parts_of_a_hugging_face_fetch(tmp_path):
    # One card fetch returns the card and the model's config.json under "from <url>"
    # headings. Research's citation guard counts both; the eval scored a citation of the
    # config.json as unfetched (r-qwen-license, 2026-10-26).
    card = "https://huggingface.co/Qwen/Qwen3-8B"
    config = "https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json"
    text = f"[This model's config.json, from {config}]\n{{}}\n\n[The model card, from {card}]\nx"
    a = _write(
        tmp_path / "research" / "t-1.jsonl",
        [
            {
                "event": "tool_result",
                "ok": True,
                "call": {"name": "web__fetch_url", "arguments": {"url": card}},
                "output": json.dumps({"url": card, "title": None, "text": text}),
            },
        ],
    )
    assert trace_stats([a])["fetched"] == [card, config]


def test_guard_reports_tokens_of_refused_replies():
    class Inner:
        model = "m"

        def __init__(self) -> None:
            self.n = 0

        def chat(self, messages, tools):
            self.n += 1
            return ChatResponse(
                message=Message(role="assistant", content="" if self.n == 1 else "done"),
                prompt_tokens=100,
                completion_tokens=7,
            )

    resp = AnswerGuardLLM(Inner()).chat([Message(role="user", content="q")], [])
    assert resp.message.content == "done"
    assert (resp.prompt_tokens, resp.completion_tokens) == (200, 14)


def test_guard_writes_each_send_back_to_the_trace(tmp_path):
    # In the 2026-10-24 rerun nothing in the traces said whether a check had refused an
    # answer: the agent's trace only has the reply it keeps.
    page = Message(
        role="tool",
        tool_name="web__fetch_url",
        tool_call_id="f1",
        content=json.dumps({"url": "https://a.example/", "text": "The port is 11434."}),
    )
    call = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="f1", name="web__fetch_url", arguments={"url": "https://a.example/"})
        ],
    )
    replies = [
        "",  # empty
        "Port 11434 [https://b.example/]",  # cites a page it didn't fetch
        "Port 11434 [https://a.example/]",
    ]

    class Inner:
        model = "m"

        def chat(self, messages, tools):
            return ChatResponse(
                message=Message(role="assistant", content=replies.pop(0)),
                prompt_tokens=100,
                completion_tokens=7,
            )

    trace = tmp_path / "research" / "t-1.jsonl"
    guard = AnswerGuardLLM(Inner(), check_citations="fetch_url", tracer=Tracer(trace))
    resp = guard.chat([Message(role="user", content="q"), call, page], [])
    assert resp.message.content == "Port 11434 [https://a.example/]"
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [(e["event"], e["check"], e["nudge"]) for e in events] == [
        ("answer_sent_back", "empty", 1),
        ("answer_sent_back", "citations", 2),
    ]
    assert events[1]["flagged"] == ["https://b.example"]
    assert events[1]["content"] == "Port 11434 [https://b.example/]"
    assert (events[1]["prompt_tokens"], events[1]["completion_tokens"]) == (100, 7)
    # The eval counts them, and each refused reply as a model call.
    s = trace_stats([trace])
    assert s["sent_back"] == {"empty": 1, "citations": 1}
    assert s["llm_calls"] == 2


def test_guard_flags_an_unverified_number_after_the_nudge_budget_is_spent(tmp_path):
    # Numbers are load-bearing, so unlike a citation the guard can't strip one. When the
    # budget is spent and the answer still states a number no fetched page gives, the answer
    # ships but the guard records numbers_unverified so the eval marks the run unverified.
    page = Message(
        role="tool",
        tool_name="web__fetch_url",
        tool_call_id="f1",
        content=json.dumps({"url": "https://a.example/", "text": "The context is 32768 tokens."}),
    )
    call = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="f1", name="web__fetch_url", arguments={"url": "https://a.example/"})
        ],
    )

    class Inner:  # keeps citing a head dimension (5120) no fetched page states
        model = "m"

        def chat(self, messages, tools):
            return ChatResponse(
                message=Message(role="assistant", content="Head dimension: 5120"),
                prompt_tokens=80,
                completion_tokens=4,
            )

    trace = tmp_path / "research" / "t-2.jsonl"
    guard = AnswerGuardLLM(Inner(), check_numbers="fetch_url", max_nudges=2, tracer=Tracer(trace))
    resp = guard.chat([Message(role="user", content="q"), call, page], [])
    assert resp.message.content == "Head dimension: 5120"  # shipped as-is
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[-1] == "numbers_unverified"  # after the sent-back nudges
    assert events[-1]["flagged"] == ["5120"] and events[-1]["nudge"] == 2
    s = trace_stats([trace])
    assert s["unverified"] == {"numbers": 1}


def test_guard_says_when_an_empty_reply_was_cut_off(tmp_path):
    # 2026-10-26 rerun: in every a-weights run qwen3:14b spent all 2,048 tokens of its reply
    # limit thinking ("7.7 or 7.70?"), so its reply was empty. Only done_reason shows that.
    seen = []

    class Inner:
        model = "m"

        def chat(self, messages, tools):
            seen.append(messages)
            first = len(seen) == 1
            return ChatResponse(
                message=Message(role="assistant", content="" if first else "7.70 GiB"),
                prompt_tokens=1270,
                completion_tokens=2048 if first else 500,
                done_reason="length" if first else "stop",
            )

    trace = tmp_path / "analyst" / "t-1.jsonl"
    resp = AnswerGuardLLM(Inner(), tracer=Tracer(trace)).chat(
        [Message(role="user", content="q")], []
    )
    assert resp.message.content == "7.70 GiB"
    assert "cut off at the length limit" in seen[1][-1].content
    [event] = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert (event["check"], event["done_reason"]) == ("empty", "length")
    assert event["flagged"] == ["cut off at the reply limit"]
    # A kept reply that was cut off counts too (an over-long answer the guard let through).
    with trace.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "llm_response", "done_reason": "length"}) + "\n")
        f.write(json.dumps({"event": "llm_response", "done_reason": "stop"}) + "\n")
    s = trace_stats([trace])
    assert (s["cut_off"], s["sent_back"]) == (2, {"empty": 1})


def test_guard_keeps_the_plain_empty_reminder_without_a_done_reason():
    # Older AgentOS replies, and other backends, have no done_reason.
    seen = []

    class Inner:
        model = "m"

        def chat(self, messages, tools):
            seen.append(messages)
            content = "" if len(seen) == 1 else "done"
            return ChatResponse(message=Message(role="assistant", content=content))

    AnswerGuardLLM(Inner()).chat([Message(role="user", content="q")], [])
    assert seen[1][-1].content.startswith("Your reply was empty. Do the task now")


def test_guard_without_a_tracer_or_a_refusal_writes_nothing(tmp_path):
    trace = tmp_path / "t.jsonl"

    class Inner:
        model = "m"

        def chat(self, messages, tools):
            return ChatResponse(message=Message(role="assistant", content="fine"))

    AnswerGuardLLM(Inner(), tracer=Tracer(trace)).chat([], [])
    assert not trace.exists()
    AnswerGuardLLM(Inner()).chat([], [])  # no tracer: nothing to write to


# --- the two systems, in-process ---------------------------------------------------------


def tracing(root: Path, folder: str, answer: str, fetched=(), tokens: int = 10):
    """A fake agent that writes an AgentOS-style trace named after its A2A task."""

    def run(task, progress, flag):
        events = [{"event": "llm_response", "prompt_tokens": tokens, "completion_tokens": 1}]
        events += [
            {
                "event": "tool_result",
                "ok": True,
                "call": {"name": "web__fetch_url", "arguments": {"url": u}},
            }
            for u in fetched
        ]
        _write(root / folder / f"20261017-000000-{TASK_ID.get()}.jsonl", events)
        return RunOutcome(answer=answer, stop_reason="final_answer", steps=1)

    return run


TASK = EvalTask.model_validate(
    {
        "id": "r-x",
        "category": "research",
        "cites": True,
        "prompt": "Which port does Ollama use?",
        "checks": [{"type": "answer_number", "value": 11434}],
    }
)


async def test_single_run_is_scored_and_costed_from_its_trace(tmp_path):
    root = tmp_path / "traces"
    fleet = Fleet(
        tmp_path,
        {
            "Solo": (
                9015,
                "general",
                tracing(
                    root,
                    "solo",
                    "Port 11434 [https://docs.ollama.com/faq] [https://never.example/x]",
                    fetched=["https://docs.ollama.com/faq"],
                    tokens=500,
                ),
            )
        },
    )
    cards, _ = await discover_all(fleet.urls, fleet.http)
    rec = await run_single(TASK, cards[0], fleet.http, skill="general", timeout=20)
    finish_record(rec, TASK, root)
    assert rec.passed and rec.state == "completed"
    assert rec.prompt_tokens == 500 and rec.tool_calls == 1
    assert rec.unfetched_citations == ["https://never.example/x"]
    assert rec.citations_unscoreable is False


async def test_citations_are_unscoreable_when_no_fetch_trace_is_found(tmp_path):
    # The run cites a page but its located trace did no fetch_url at all -- the fetching
    # agent's trace was not under trace_root (the eval's --trace-root didn't match the
    # fleet's --trace-dir). cited - fetched would flag everything; the flag says so instead.
    root = tmp_path / "traces"
    fleet = Fleet(
        tmp_path,
        {
            "Solo": (
                9015,
                "general",
                tracing(
                    root,
                    "solo",
                    "Port 11434 [https://docs.ollama.com/faq].",
                    fetched=(),  # no fetch_url recorded in the trace the scorer finds
                    tokens=500,
                ),
            )
        },
    )
    cards, _ = await discover_all(fleet.urls, fleet.http)
    rec = await run_single(TASK, cards[0], fleet.http, skill="general", timeout=20)
    finish_record(rec, TASK, root)
    assert rec.cited == ["https://docs.ollama.com/faq"]
    assert rec.citations_unscoreable is True
    # The report counts it apart, not as a run that cited an unfetched page.
    text = report([rec])
    assert "Citations unscoreable | 1 (fetch traces not found)" in text
    assert "Runs citing an unfetched page | 0 of 0" in text


async def test_multi_run_sums_every_agent_and_the_orchestrator(tmp_path):
    root = tmp_path / "traces"
    fleet = Fleet(
        tmp_path,
        {
            "Research": (
                9011,
                "web_research",
                tracing(
                    root,
                    "research",
                    "1. port 11434 [https://docs.ollama.com/faq]",
                    fetched=["https://docs.ollama.com/faq"],
                    tokens=300,
                ),
            ),
            "Verifier": (
                9013,
                "claim_verification",
                tracing(root, "verifier", "1. SUPPORTED", tokens=200),
            ),
        },
    )
    cards, _ = await discover_all(fleet.urls, fleet.http)
    chat = ScriptedChat(
        plan(
            step("s1", "web_research", "Find Ollama's port"),
            step("s2", "claim_verification", "Check: {{s1}}"),
        ),
        "Ollama listens on 11434 [https://docs.ollama.com/faq].",
    )
    rec = await run_multi(
        TASK,
        cards,
        chat,
        fleet.http,
        trace_root=root,
        policy=RunPolicy(step_timeout=20),
        timeout=60,
        model_name="m",
    )
    finish_record(rec, TASK, root)
    assert rec.passed, rec
    assert rec.plan == ["web_research", "claim_verification"]
    assert rec.agent_tasks == 2 and rec.steps_failed == 0
    # 300 + 200 from the agents, 100 + 100 from the orchestrator's plan and combine.
    assert rec.prompt_tokens == 700
    assert rec.llm_calls == 4
    assert rec.unfetched_citations == []


async def test_multi_plan_failure_is_a_plan_error(tmp_path):
    root = tmp_path / "traces"
    fleet = Fleet(tmp_path, {"Research": (9011, "web_research", tracing(root, "r", "x"))})
    cards, _ = await discover_all(fleet.urls, fleet.http)
    chat = ScriptedChat("not json", "still not", "nope")
    rec = await run_multi(
        TASK,
        cards,
        chat,
        fleet.http,
        trace_root=root,
        policy=RunPolicy(step_timeout=20),
        timeout=60,
        model_name="m",
    )
    finish_record(rec, TASK, root)
    assert not rec.passed and rec.failure == "plan_error"


# --- the run loop and the report -------------------------------------------------------


def test_schedule_alternates_which_system_goes_first():
    tasks = load_tasks(TASKS)[:2]
    order = [(t.id, s, r) for t, s, r in schedule(tasks, ["single", "multi"], 2)]
    assert order[:4] == [
        ("r-qwen-license", "single", 0),
        ("r-qwen-license", "multi", 0),
        ("r-a2a-origin", "multi", 0),
        ("r-a2a-origin", "single", 0),
    ]
    assert order[4][1] == "multi"  # repeat 1 starts the other way round


async def test_run_eval_streams_resumes_and_reports(tmp_path):
    tasks = [TASK, TASK.model_copy(update={"id": "r-y", "category": "analysis", "cites": False})]
    calls = []

    async def run_one(task, system):
        calls.append((task.id, system))
        good = system == "multi" or task.id == "r-x"
        return RunRecord(
            system=system,
            task_id=task.id,
            category=task.category,
            repeat=0,
            passed=False,
            state="completed",
            answer="11434" if good else "8080",
            seconds=10.0 if system == "single" else 30.0,
        )

    out = tmp_path / "m6"
    await run_eval(
        tasks, ["single", "multi"], repeats=1, out_dir=out, run_one=run_one, trace_root=tmp_path
    )
    assert len(calls) == 4
    # Re-running the same folder runs nothing new.
    await run_eval(
        tasks, ["single", "multi"], repeats=1, out_dir=out, run_one=run_one, trace_root=tmp_path
    )
    assert len(calls) == 4
    results = load_results(out / "results.jsonl")
    assert sorted((r.system, r.task_id, r.passed) for r in results) == [
        ("multi", "r-x", True),
        ("multi", "r-y", True),
        ("single", "r-x", True),
        ("single", "r-y", False),
    ]
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["repeats"] == 1
    text = report(results, tasks)
    assert "| Runs passed | 1/2 (50%) | 2/2 (100%) |" in text
    assert "| Passed, every cited page fetched | 1/2 (50%) | 2/2 (100%) |" in text
    assert "multi passed more often on 1, less often on 0 and equally on 1" in text
    assert "| single | r-y | 0 | wrong |" in text
    assert "| Answers sent back by a check | none | none |" in text
    next(r for r in results if r.system == "multi").sent_back = {"citations": 2, "numbers": 1}
    assert "| Answers sent back by a check | none | 1 runs (citations 2, numbers 1) |" in (
        report(results, tasks)
    )
    assert "| Replies cut off at the reply limit | none | none |" in text
    next(r for r in results if r.system == "single").cut_off = 2
    assert "| Replies cut off at the reply limit | 1 runs (2 replies) | none |" in (
        report(results, tasks)
    )


# --- side effects: writes and cancels ------------------------------------------------------

SIDE = ROOT / "benchmarks" / "m6" / "tasks-side-effects.yaml"


def test_side_effect_tasks_load_and_stay_out_of_the_m6_set():
    tasks = {t.id: t for t in load_tasks(SIDE)}
    assert set(tasks) == {
        "w-verified-write",
        "w-unverified-detail",
        "x-cancel-dispatch",
        "x-cancel-midrun",
    }
    assert tasks["w-verified-write"].allow_writes and tasks["w-verified-write"].systems == ["multi"]
    assert tasks["x-cancel-dispatch"].cancel.at == "dispatch"
    assert tasks["x-cancel-midrun"].cancel.after == 30
    assert not ({t.id for t in load_tasks(TASKS)} & set(tasks))
    # The write task never goes to the single agent; the mid-run cancel goes to both.
    order = [(t.id, s) for t, s, _ in schedule(list(tasks.values()), ["single", "multi"], 1)]
    assert ("w-verified-write", "single") not in order
    assert {s for t, s in order if t == "x-cancel-midrun"} == {"single", "multi"}


@pytest.mark.parametrize(
    "cancel",
    [{}, {"at": "dispatch", "after": 3}],
)
def test_cancel_needs_exactly_one_moment(cancel):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvalTask.model_validate(
            {
                "id": "x",
                "category": "cancel",
                "systems": ["multi"],
                "cancel": cancel,
                "prompt": "p",
                "checks": [{"type": "remote_stopped"}],
            }
        )


def test_cancel_at_dispatch_is_multi_only():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="systems"):
        EvalTask.model_validate(
            {
                "id": "x",
                "category": "cancel",
                "cancel": {"at": "dispatch"},
                "prompt": "p",
                "checks": [{"type": "remote_stopped"}],
            }
        )


def _add_fact(statement: str, ok: bool, output: str) -> dict:
    return {
        "event": "tool_result",
        "ok": ok,
        "call": {
            "name": "domaingraph__add_fact",
            "arguments": {"statement": statement, "details": {"source": "https://x.example"}},
        },
        "output": output,
    }


def test_trace_stats_reads_new_duplicate_and_refused_writes(tmp_path):
    p = _write(
        tmp_path / "knowledge" / "t-1.jsonl",
        [
            _add_fact("Ollama binds port 11434", True, '{"id": 7, "new": true}'),
            _add_fact(
                "Ollama binds port 11434",
                True,
                "ALREADY STORED: this fact was already in the knowledge base (id 7); "
                'nothing new was added.\n{"id": 7, "new": false}',
            ),
            _add_fact("Context is 32,768", False, "refused: the Verifier did not check this"),
        ],
    )
    writes = trace_stats([p])["writes"]
    assert [(w["ok"], w["id"], w["new"]) for w in writes] == [
        (True, 7, True),
        (True, 7, False),
        (False, None, None),
    ]
    assert writes[2]["error"].startswith("refused")


def test_stored_check_reads_writes_not_the_answer():
    task = next(t for t in load_tasks(SIDE) if t.id == "w-verified-write")
    rec = RunRecord(system="multi", task_id=task.id, category="write", repeat=0, passed=False)
    claim = "Ollama binds 127.0.0.1 port 11434 by default"
    # The answer claims both were stored; the trace says only claim 1 went through.
    answer = "Both claims were stored."
    rec.writes = [
        {"statement": claim, "ok": True, "id": 7, "new": False},
        {"statement": "default context window is 32,768 tokens", "ok": False, "id": None},
    ]
    assert all(c["ok"] for c in score(task, answer, rec))
    rec.writes[1]["ok"] = True  # the PR #21 bug: the unsupported claim is stored
    assert [c["ok"] for c in score(task, answer, rec)] == [True, False]
    rec.writes = []  # nothing stored at all
    assert [c["ok"] for c in score(task, answer, rec)] == [False, True]


def _cancel_task(cancel: dict, systems=None) -> EvalTask:
    return EvalTask.model_validate(
        {
            "id": "x-c",
            "category": "cancel",
            "systems": systems,
            "cancel": {**cancel, "grace": 0.3},
            "prompt": "Which port does Ollama use?",
            "checks": [{"type": "remote_stopped", "min_tasks": 1}],
        }
    )


# The cancel tests need agents in their own processes: the in-process ASGI transport
# hands a streamed reply over only when it is complete, so no task id would arrive
# before the cancel. tests/agent_proc.py runs 20 s per task and stops when cancelled.


@pytest.fixture
def procs(tmp_path):
    started: list[Proc] = []

    def start(name: str, skill: str, seconds: float) -> Proc:
        p = Proc(tmp_path, name, skill, seconds)
        started.append(p)
        return p

    yield start
    for p in started:
        if p.p.poll() is None:
            p.kill()


async def _multi_cancel(
    tmp_path, procs, cancel: dict, policy: RunPolicy | None = None
) -> tuple[RunRecord, object]:
    agent = procs("Research", "web_research", 20)
    agent.wait_up()
    task = _cancel_task(cancel, ["multi"])
    root = tmp_path / "traces"
    async with httpx.AsyncClient(timeout=30) as http:
        cards, _ = await discover_all([agent.url], http)
        rec = await run_multi(
            task,
            cards,
            ScriptedChat(plan(step("s1", "web_research", "Find Ollama's port"))),
            http,
            trace_root=root,
            policy=policy or RunPolicy(step_timeout=30),
            timeout=30,
            model_name="m",
        )
    return finish_record(rec, task, root), agent


async def test_cancel_at_dispatch_stops_the_remote_task(tmp_path, procs):
    rec, agent = await _multi_cancel(tmp_path, procs, {"at": "dispatch"})
    assert rec.state == "canceled" and rec.cancelled_at
    assert [t["state"] for t in rec.remote] == ["canceled"]
    assert "canceled" in agent.states() and "completed" not in agent.states()
    assert rec.passed, rec.checks


async def test_cancel_at_dispatch_fails_when_the_cancel_misses_the_task(tmp_path, procs):
    # PR #19's bug, put back: the step stops waiting for the task id at once, so the
    # remote task is never cancelled. The check must catch it. cancel_id_wait=0 reproduces
    # the too-short wait that gave up before the id arrived.
    rec, _ = await _multi_cancel(
        tmp_path, procs, {"at": "dispatch"}, RunPolicy(step_timeout=30, cancel_id_wait=0.0)
    )
    assert not rec.passed
    assert "remote_stopped" in {c["type"] for c in rec.checks if not c["ok"]}


async def test_cancel_after_seconds_stops_a_multi_run(tmp_path, procs):
    rec, agent = await _multi_cancel(tmp_path, procs, {"after": 1.0})
    assert [t["state"] for t in rec.remote] == ["canceled"]
    assert "canceled" in agent.states()
    assert rec.passed, rec.checks


async def test_cancel_after_seconds_stops_the_single_agent(tmp_path, procs):
    agent = procs("Solo", "general", 20)
    agent.wait_up()
    task = _cancel_task({"after": 1.0})
    async with httpx.AsyncClient(timeout=30) as http:
        cards, _ = await discover_all([agent.url], http)
        rec = await run_single(task, cards[0], http, skill="general", timeout=30)
    finish_record(rec, task, tmp_path / "traces")
    assert rec.cancelled_at and [t["state"] for t in rec.remote] == ["canceled"]
    assert "canceled" in agent.states()
    assert rec.passed, rec.checks


def test_calls_after_counts_model_calls_past_the_cancel(tmp_path):
    from agentmesh.evaluate import calls_after

    a = _write(
        tmp_path / "a.jsonl",
        [{"ts": 1.0, "event": "llm_response"}, {"ts": 5.0, "event": "llm_response"}],
    )
    b = _write(
        tmp_path / "b.jsonl",
        [{"ts": t, "event": "llm_response"} for t in (6.0, 7.0)]
        + [{"ts": 8.0, "event": "answer_sent_back"}, {"ts": 9.0, "event": "tool_result"}],
    )
    assert calls_after([a, b], 4.0) == 3
    assert calls_after([a], 9.0) == 0


def test_remote_stopped_fails_when_a_task_kept_working():
    task = _cancel_task({"after": 1})
    rec = RunRecord(system="multi", task_id="x-c", category="cancel", repeat=0, passed=False)
    assert not score(task, "", rec)[0]["ok"]  # never cancelled: nothing was tested
    rec.cancelled_at = 1.0
    rec.remote = [{"agent": "R", "task_id": "t", "state": "working"}]
    assert "not canceled" in score(task, "", rec)[0]["detail"]
    rec.remote[0]["state"] = "canceled"
    rec.calls_after_cancel = 3
    assert "model calls after" in score(task, "", rec)[0]["detail"]
    rec.calls_after_cancel = 1
    # A step that had already finished before the cancel may be completed.
    rec.remote.append(
        {"agent": "V", "task_id": "u", "state": "completed", "finished_before_cancel": True}
    )
    assert score(task, "", rec)[0]["ok"]


def test_report_shows_a_dash_for_a_system_that_skipped_a_task():
    rs = [
        RunRecord(system="single", task_id="a", category="c", repeat=0, passed=True),
        RunRecord(system="multi", task_id="a", category="c", repeat=0, passed=True),
        RunRecord(system="multi", task_id="w", category="write", repeat=0, passed=True),
    ]
    text = report(rs)
    assert re.search(r"\| w \| - \| - \| 1/1 \|", text)
    assert re.search(r"\| write \| - \| - \| - \| 1/1 \(100%\) \|", text)
    assert "multi passed more often on 0, less often on 0 and equally on 1" in text
