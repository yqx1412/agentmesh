"""``agentmesh watch``: the lifecycle journal, trace joining and the localhost page."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest
from test_m1 import fake_runner, http_for, make_spec

from agentmesh.client import ask, discover
from agentmesh.executor import TASK_ID, RunOutcome
from agentmesh.journal import JOURNAL_NAME, JsonlJournal
from agentmesh.server import build_app
from agentmesh.watch import (
    AgentInfo,
    TraceReader,
    Watcher,
    _host_ok,
    collect_agent,
    serve,
)


async def _run(tmp_path: Path, runner):
    records: list[dict] = []
    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, runner, records.append)) as http:
        run = await ask(await discover(spec.url, http), "hello", http)
    return run, records


async def test_journal_records_completed_lifecycle(tmp_path):
    run, records = await _run(tmp_path, fake_runner())
    assert [r["state"] for r in records] == ["submitted", "working", "working", "completed"]
    assert {r["task"] for r in records} == {run.task_id}
    assert records[0]["text"] == "hello"
    assert records[2]["message"] == "step 1: calling read_file"
    assert records[-1]["answer"] == "done: hello"
    assert records[-1]["stats"]["tool_calls"] == 1


async def test_journal_records_failure(tmp_path):
    _, records = await _run(tmp_path, fake_runner(answer="half", stop="max_steps"))
    assert records[-1]["state"] == "failed"
    assert "max_steps" in records[-1]["message"]


async def test_broken_journal_does_not_fail_the_task(tmp_path):
    def bad(record):
        raise OSError("disk full")

    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, fake_runner(), bad)) as http:
        run = await ask(await discover(spec.url, http), "hello", http)
    assert run.state == "completed"


async def test_runner_sees_the_a2a_task_id(tmp_path):
    def runner(task, progress, flag):
        return RunOutcome(answer=TASK_ID.get() or "none", stop_reason="final_answer")

    run, _ = await _run(tmp_path, runner)
    assert run.answer == run.task_id


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _trace(task: str = "add 2 and 3") -> list[dict]:
    call = {"id": "c1", "name": "calculator", "arguments": {"expression": "2+3"}}
    return [
        {"ts": 10.0, "event": "run_start", "model": "m", "task": task, "tools": ["calculator"]},
        {
            "ts": 11.0,
            "event": "llm_response",
            "step": 1,
            "content": "",
            "tool_calls": [call],
            "prompt_tokens": 50,
            "completion_tokens": 5,
        },
        {"ts": 11.5, "event": "tool_result", "step": 1, "call": call, "ok": True, "output": "5"},
        {"ts": 12.0, "event": "llm_response", "step": 2, "content": "It is 5.", "tool_calls": []},
        {"ts": 12.0, "event": "run_end", "answer": "It is 5.", "stop_reason": "final_answer"},
    ]


def test_collect_joins_journal_and_trace(tmp_path):
    folder = tmp_path / "analyst"
    _write(
        folder / JOURNAL_NAME,
        [
            {
                "ts": 9.0,
                "agent": "Analyst",
                "task": "t1",
                "context": "c",
                "state": "submitted",
                "text": "add 2 and 3",
            },
            {
                "ts": 9.5,
                "agent": "Analyst",
                "task": "t1",
                "context": "c",
                "state": "working",
                "message": "Starting the agent.",
            },
            {
                "ts": 12.1,
                "agent": "Analyst",
                "task": "t1",
                "context": "c",
                "state": "completed",
                "answer": "It is 5.",
                "stats": {"steps": 2},
            },
        ],
    )
    _write(folder / "20261012-200000-t1.jsonl", _trace())
    (task,) = collect_agent(folder, TraceReader())
    assert task["id"] == "t1"
    assert task["state"] == "completed"
    assert (task["started"], task["ended"]) == (9.0, 12.1)
    assert [s["kind"] for s in task["steps"]] == ["start", "model", "tool", "model", "end"]
    assert task["tool_calls"] == 1
    assert task["steps"][2]["output"] == "5"
    assert task["answer"] == "It is 5."


def test_collect_reads_trace_only_runs_and_running_tasks(tmp_path):
    folder = tmp_path / "research"
    _write(folder / "20261011-235008-35092.jsonl", _trace("old run"))
    _write(folder / "20261012-200000-t2.jsonl", _trace()[:3])  # still going
    tasks = {t["id"]: t for t in collect_agent(folder, TraceReader())}
    assert tasks["35092"]["state"] == "completed"
    assert tasks["35092"]["text"] == "old run"
    assert tasks["35092"]["journal"] is False
    assert tasks["t2"]["state"] == "working"
    assert tasks["t2"]["ended"] is None


def test_reader_skips_a_half_written_line(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"event": "run_start"}\n{"event": "llm_res', encoding="utf-8")
    assert TraceReader().records(p) == [{"event": "run_start"}]


def test_long_tool_output_is_clipped(tmp_path):
    folder = tmp_path / "a"
    trace = _trace()
    trace[2]["output"] = "x" * 10_000
    _write(folder / "20261012-200000-t3.jsonl", trace)
    (task,) = collect_agent(folder, TraceReader())
    out = task["steps"][2]["output"]
    assert len(out) < 4100
    assert out.endswith("more chars]")


def test_journal_writer_appends_lines(tmp_path):
    j = JsonlJournal(tmp_path / "x" / JOURNAL_NAME, "Agent A")
    j({"task": "t", "state": "submitted"})
    j({"task": "t", "state": "working"})
    lines = (tmp_path / "x" / JOURNAL_NAME).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["state"] for line in lines] == ["submitted", "working"]
    assert json.loads(lines[0])["agent"] == "Agent A"


@pytest.mark.parametrize(
    ("host", "ok"),
    [
        ("127.0.0.1:9130", True),
        ("localhost:9130", True),
        ("LOCALHOST", True),
        ("[::1]:9130", True),
        ("evil.example:9130", False),
        ("127.0.0.1.evil.example", False),
        ("", False),
        (None, False),
    ],
)
def test_host_check(host, ok):
    assert _host_ok(host) is ok


@pytest.fixture
def watch_server(tmp_path):
    _write(tmp_path / "analyst" / "20261012-200000-t1.jsonl", _trace())
    watcher = Watcher(
        tmp_path,
        [AgentInfo("analyst", "Analyst", None, "m"), AgentInfo("verifier", "Verifier", None, "m")],
    )
    server = serve(watcher, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _get(port: int, path: str, host: str | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host} if host else {}
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    return resp.status, resp.getheader("Content-Type"), resp.read()


def test_server_serves_page_and_state(watch_server):
    status, ctype, body = _get(watch_server, "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b"AgentMesh watch" in body
    status, ctype, body = _get(watch_server, "/api/state")
    assert status == 200 and ctype.startswith("application/json")
    state = json.loads(body)
    names = [a["name"] for a in state["agents"]]
    assert names == ["Analyst", "Verifier"]
    assert state["agents"][0]["tasks"][0]["state"] == "completed"
    assert state["agents"][1]["tasks"] == []


def test_server_refuses_foreign_host(watch_server):
    status, _, _ = _get(watch_server, "/api/state", host="evil.example")
    assert status == 403


def test_server_binds_loopback_only(watch_server):
    import socket

    with socket.socket() as s:
        s.settimeout(1)
        hostname_ip = socket.gethostbyname(socket.gethostname())
        if hostname_ip.startswith("127."):
            pytest.skip("host name resolves to loopback")
        assert s.connect_ex((hostname_ip, watch_server)) != 0


def test_page_never_uses_innerhtml():
    page = (Path(__file__).parents[1] / "src" / "agentmesh" / "watch.html").read_text("utf-8")
    assert "innerHTML" not in page and "insertAdjacentHTML" not in page


def test_probe_marks_agent_down(tmp_path):
    import httpx

    watcher = Watcher(tmp_path, [AgentInfo("a", "A", "http://127.0.0.1:9", "m")])
    with httpx.Client(timeout=0.5) as http:
        watcher.probe_once(http)
    assert watcher.agents["a"].up is False
