"""M6: the single-vs-multi evaluation harness, in-process (real A2A servers, scripted models)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from agentos.llm import ChatResponse, Message
from test_m3 import Fleet, ScriptedChat, plan, step

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
