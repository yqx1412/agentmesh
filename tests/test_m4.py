"""M4: retries, fallback, timeouts, cancellation and push notifications.

The kill tests run each agent as a real process (tests/agent_proc.py) and kill it while
it works on a task: the orchestrator must either recover on another agent or fail the step
cleanly, without hanging and without running the steps that needed it.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from agentmesh.client import PushReceiver, ask_push
from agentmesh.executor import Cancelled, RunOutcome
from agentmesh.journal import JOURNAL_NAME, JsonlJournal
from agentmesh.orchestrator import ChatReply, RunPolicy, discover_all, orchestrate
from agentmesh.server import build_app, build_card, is_loopback_url
from agentmesh.spec import AgentSpec, SkillSpec

AGENT_PROC = Path(__file__).with_name("agent_proc.py")
FAST = dict(backoff=0.05, poll_every=0.5)


# --- helpers ---------------------------------------------------------------------------


class ScriptedChat:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    async def __call__(self, messages, schema=None):
        return ChatReply(self.replies.pop(0) if self.replies else "combined")


def plan(*steps: tuple[str, str, list[str]]) -> str:
    return json.dumps(
        {
            "steps": [
                {"id": sid, "skill": skill, "input": f"do {skill}", "depends_on": deps}
                for sid, skill, deps in steps
            ]
        }
    )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Proc:
    """One test agent in its own process."""

    def __init__(self, tmp: Path, name: str, skill: str, seconds: float) -> None:
        self.name = name
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.journal = tmp / name / JOURNAL_NAME
        self.journal.parent.mkdir(parents=True)
        self.p = subprocess.Popen(
            [
                sys.executable,
                str(AGENT_PROC),
                name,
                str(self.port),
                skill,
                str(seconds),
                str(self.journal),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_up(self, timeout: float = 30.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            assert self.p.poll() is None, f"{self.name} exited early"
            try:
                httpx.get(f"{self.url}/.well-known/agent-card.json", timeout=1).raise_for_status()
                return
            except httpx.HTTPError:
                time.sleep(0.1)
        raise AssertionError(f"{self.name} never served its card")

    def states(self) -> list[str]:
        if not self.journal.exists():
            return []
        lines = self.journal.read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["state"] for line in lines if line.strip()]

    def kill(self) -> None:
        self.p.kill()
        self.p.wait(10)


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


async def until(check, timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < end, "timed out waiting"
        await asyncio.sleep(0.05)


async def run_and_kill(victim: Proc, urls: list[str], the_plan: str, policy: RunPolicy, tmp):
    """Start the orchestration, kill ``victim`` once it is working on its task, and return
    the finished orchestration and how long it took after the kill."""
    async with httpx.AsyncClient(timeout=30) as http:
        cards, errors = await discover_all(urls, http)
        assert errors == []
        job = asyncio.create_task(
            orchestrate(
                "t",
                cards,
                ScriptedChat(the_plan),
                http,
                trace_root=tmp / "traces",
                policy=policy,
            )
        )
        await until(lambda: victim.states().count("working") >= 3)  # mid-task, ticking
        victim.kill()
        killed = time.monotonic()
        out = await asyncio.wait_for(job, 30)
        return out, time.monotonic() - killed


# --- killing an agent mid-task -----------------------------------------------------------


@pytest.mark.parametrize("push", [False, True], ids=["stream", "push"])
async def test_killed_agent_recovers_on_another_agent(procs, tmp_path, push):
    a = procs("Alpha", "work", 60)
    b = procs("Beta", "work", 0.3)
    for p in (a, b):
        p.wait_up()
    receiver = await PushReceiver().serve() if push else None
    try:
        out, after = await run_and_kill(
            a,
            [a.url, b.url],
            plan(("s1", "work", [])),
            RunPolicy(step_timeout=30, retries=1, push=receiver, **FAST),
            tmp_path,
        )
    finally:
        if receiver:
            await receiver.close()
    r = out.results["s1"]
    assert out.state == "completed", out.message
    assert r.answer == "done by Beta"
    tried = [(x.agent, x.state) for x in r.attempts]
    # Alpha died, its retry found nothing listening, then Beta did the work.
    assert tried == [("Alpha", "unreachable"), ("Alpha", "unreachable"), ("Beta", "completed")]
    assert "completed" not in a.states()
    assert after < 15


async def test_killed_agent_without_fallback_fails_cleanly(procs, tmp_path):
    a = procs("Alpha", "work", 60)
    b = procs("Beta", "report", 0.3)  # needs s1, so it must never run
    for p in (a, b):
        p.wait_up()
    out, after = await run_and_kill(
        a,
        [a.url, b.url],
        plan(("s1", "work", []), ("s2", "report", ["s1"])),
        RunPolicy(step_timeout=30, retries=1, **FAST),
        tmp_path,
    )
    assert out.state == "failed"
    s1, s2 = out.results["s1"], out.results["s2"]
    assert s1.state == "failed" and s1.message.startswith("every attempt failed")
    assert [x.state for x in s1.attempts] == ["unreachable", "unreachable"]
    assert s2.state == "skipped" and "s1 (failed)" in s2.message
    assert b.states() == []  # Beta never got a task
    assert after < 15


async def test_killed_agent_in_push_mode_is_noticed_by_polling(procs, tmp_path):
    # No stream breaks when a push-mode agent dies: the poll is what notices it.
    a = procs("Alpha", "work", 60)
    a.wait_up()
    async with PushReceiver() as receiver:
        out, after = await run_and_kill(
            a,
            [a.url],
            plan(("s1", "work", [])),
            RunPolicy(step_timeout=30, retries=0, push=receiver, **FAST),
            tmp_path,
        )
    assert out.state == "failed"
    assert [x.state for x in out.results["s1"].attempts] == ["unreachable"]
    assert after < 10


async def test_timeout_cancels_a_streamed_task(procs, tmp_path):
    a = procs("Alpha", "work", 60)
    a.wait_up()
    async with httpx.AsyncClient(timeout=30) as http:
        cards, _ = await discover_all([a.url], http)
        out = await orchestrate(
            "t",
            cards,
            ScriptedChat(plan(("s1", "work", []))),
            http,
            policy=RunPolicy(step_timeout=1.5, retries=2, **FAST),
        )
    (attempt,) = out.results["s1"].attempts
    assert attempt.state == "timeout"
    assert "remote task canceled" in attempt.message
    await until(lambda: "canceled" in a.states(), timeout=5)
    assert "completed" not in a.states()


# --- in-process agents -------------------------------------------------------------------


def make_spec(name: str, port: int, skill: str, tmp: Path) -> AgentSpec:
    return AgentSpec(
        name=name,
        description=f"{name} for tests",
        port=port,
        workspace=tmp / name,
        skills=[SkillSpec(id=skill, name=skill, description=f"does {skill}")],
    )


class Fleet:
    """Agents served in-process; ``http`` routes each agent's URL (and the push webhook)
    to its app."""

    def __init__(self, tmp: Path, agents: dict[str, tuple[int, str, object]]) -> None:
        self.receiver = PushReceiver("http://127.0.0.1:9199/")
        mounts = {"http://127.0.0.1:9199": httpx.ASGITransport(app=self.receiver.app)}
        push_http = httpx.AsyncClient(mounts=dict(mounts), timeout=10)
        self.urls = []
        self.journals = {}
        for name, (port, skill, runner) in agents.items():
            s = make_spec(name, port, skill, tmp)
            self.journals[name] = tmp / "traces" / name / JOURNAL_NAME
            app = build_app(s, runner, JsonlJournal(self.journals[name], name), push_http=push_http)
            mounts[s.url] = httpx.ASGITransport(app=app)
            self.urls.append(s.url)
        self.http = httpx.AsyncClient(mounts=mounts, timeout=20)

    async def cards(self):
        cards, errors = await discover_all(self.urls, self.http)
        assert errors == []
        return cards


def done(text: str):
    def run(task, progress, flag):
        progress("working")
        return RunOutcome(answer=text, stop_reason="final_answer", steps=1)

    return run


def fails_then(text: str, failures: int):
    calls = {"n": 0}

    def run(task, progress, flag):
        calls["n"] += 1
        if calls["n"] <= failures:
            return RunOutcome(answer="", stop_reason="max_steps", steps=1)
        return RunOutcome(answer=text, stop_reason="final_answer", steps=1)

    return run


def slow(seen: dict):
    """Runs until cancelled; records whether the cancel flag reached it."""

    def run(task, progress, flag):
        seen["started"] = True
        end = time.monotonic() + 20
        while time.monotonic() < end:
            if flag.is_set():
                seen["cancelled"] = True
                raise Cancelled
            progress("tick")
            time.sleep(0.05)
        return RunOutcome(answer="too late", stop_reason="final_answer")

    return run


async def test_failed_attempt_is_retried_on_the_same_agent(tmp_path):
    fleet = Fleet(tmp_path, {"Solo": (9301, "work", fails_then("ok", 1))})
    out = await orchestrate(
        "t",
        await fleet.cards(),
        ScriptedChat(plan(("s1", "work", []))),
        fleet.http,
        policy=RunPolicy(retries=1, backoff=0),
    )
    r = out.results["s1"]
    assert out.state == "completed" and r.answer == "ok"
    assert [(a.agent, a.state) for a in r.attempts] == [("Solo", "failed"), ("Solo", "completed")]


async def test_retries_run_out_then_the_next_agent_takes_it(tmp_path):
    fleet = Fleet(
        tmp_path,
        {
            "First": (9311, "work", fails_then("never", 99)),
            "Second": (9312, "work", done("from second")),
        },
    )
    out = await orchestrate(
        "t",
        await fleet.cards(),
        ScriptedChat(plan(("s1", "work", []))),
        fleet.http,
        policy=RunPolicy(retries=1, backoff=0),
    )
    r = out.results["s1"]
    assert r.answer == "from second" and r.agent == "Second"
    assert [a.agent for a in r.attempts] == ["First", "First", "Second"]


async def test_no_fallback_stops_at_the_first_agent(tmp_path):
    fleet = Fleet(
        tmp_path,
        {
            "First": (9321, "work", fails_then("never", 99)),
            "Second": (9322, "work", done("from second")),
        },
    )
    out = await orchestrate(
        "t",
        await fleet.cards(),
        ScriptedChat(plan(("s1", "work", []))),
        fleet.http,
        policy=RunPolicy(retries=0, fallback=False),
    )
    assert out.state == "failed"
    assert [a.agent for a in out.results["s1"].attempts] == ["First"]


async def test_timeout_cancels_the_remote_task_and_is_not_retried(tmp_path):
    # In-process agents use push mode: httpx's ASGI transport buffers a streamed response
    # whole, so a stream would never show the task id before the timeout. The streamed path
    # is covered against a real process in test_timeout_cancels_a_streamed_task.
    seen: dict = {}
    fleet = Fleet(tmp_path, {"Slow": (9331, "work", slow(seen))})
    out = await orchestrate(
        "t",
        await fleet.cards(),
        ScriptedChat(plan(("s1", "work", []))),
        fleet.http,
        policy=RunPolicy(step_timeout=0.5, retries=3, backoff=0, push=fleet.receiver),
    )
    attempts = out.results["s1"].attempts
    assert [a.state for a in attempts] == ["timeout"]  # a timeout is never retried
    assert "remote task canceled" in attempts[0].message
    await until(lambda: seen.get("cancelled"), timeout=5)  # the agent stopped working


async def test_cancelling_the_orchestration_cancels_the_remote_task(tmp_path):
    seen: dict = {}
    fleet = Fleet(tmp_path, {"Slow": (9341, "work", slow(seen))})
    root = tmp_path / "otrace"
    job = asyncio.create_task(
        orchestrate(
            "t",
            await fleet.cards(),
            ScriptedChat(plan(("s1", "work", []))),
            fleet.http,
            trace_root=root,
            policy=RunPolicy(step_timeout=30, push=fleet.receiver),
        )
    )
    # Wait for the orchestrator to know the remote task id, not just for the agent to start:
    # a cancel between the two has no id to send, so the remote run would go on (a gap of
    # milliseconds, but the test hit it under load).
    await until(
        lambda: (
            any(
                "took it as task" in (json.loads(x).get("message") or "")
                for x in (root / "orchestrator" / JOURNAL_NAME)
                .read_text(encoding="utf-8")
                .splitlines()
                if x
            )
            if (root / "orchestrator" / JOURNAL_NAME).exists()
            else False
        )
    )
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    await until(lambda: seen.get("cancelled"), timeout=5)
    assert "canceled" in _states(fleet.journals["Slow"])
    assert _states(root / "orchestrator" / JOURNAL_NAME)[-1] == "canceled"


async def test_a_cancel_before_the_task_id_still_reaches_the_remote_task(tmp_path, monkeypatch):
    # The agent reports its task id only after the orchestration is cancelled. Before the
    # fix the cancel cut the call off and nothing was sent, so the remote run went on.
    from a2a.types import StreamResponse, Task

    import agentmesh.orchestrator as orch

    reported, cancelled = asyncio.Event(), []

    async def late_id(card, text, http, *, on_event=None, metadata=None, **_):
        await asyncio.sleep(0.3)
        on_event(StreamResponse(task=Task(id="remote-1")))
        reported.set()
        await asyncio.sleep(30)

    async def fake_cancel(card, task_id, http):
        cancelled.append(task_id)
        return "canceled"

    monkeypatch.setattr(orch, "ask", late_id)
    monkeypatch.setattr(orch, "cancel", fake_cancel)
    fleet = Fleet(tmp_path, {"Slow": (9345, "work", done("unused"))})
    job = asyncio.create_task(
        orchestrate(
            "t",
            await fleet.cards(),
            ScriptedChat(plan(("s1", "work", []))),
            fleet.http,
            policy=RunPolicy(step_timeout=30),
        )
    )
    await asyncio.sleep(0.1)  # the call is in flight, no task id yet
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert reported.is_set() and cancelled == ["remote-1"]


async def test_a_cancel_without_any_task_id_gives_up_after_the_wait(tmp_path, monkeypatch):
    import agentmesh.orchestrator as orch

    cancelled = []

    async def never(card, text, http, **_):
        await asyncio.sleep(30)

    async def fake_cancel(card, task_id, http):
        cancelled.append(task_id)
        return "canceled"

    monkeypatch.setattr(orch, "ask", never)
    monkeypatch.setattr(orch, "cancel", fake_cancel)
    fleet = Fleet(tmp_path, {"Slow": (9346, "work", done("unused"))})
    job = asyncio.create_task(
        orchestrate(
            "t",
            await fleet.cards(),
            ScriptedChat(plan(("s1", "work", []))),
            fleet.http,
            policy=RunPolicy(step_timeout=30, cancel_id_wait=0.2),
        )
    )
    await asyncio.sleep(0.1)
    started = time.monotonic()
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert cancelled == [] and time.monotonic() - started < 2


def _states(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [json.loads(x)["state"] for x in path.read_text(encoding="utf-8").splitlines() if x]


# --- push notifications ------------------------------------------------------------------


async def test_push_mode_delivers_every_update_without_polling(tmp_path):
    fleet = Fleet(tmp_path, {"Pusher": (9351, "work", done("pushed answer"))})
    (card,) = await fleet.cards()
    assert card.capabilities.push_notifications
    run = await ask_push(card, "go", fleet.http, fleet.receiver, poll_every=5)
    assert run.state == "completed" and run.answer == "pushed answer"
    assert run.pushes >= 2 and run.polls == 0
    assert "working" in run.states


async def test_orchestrator_in_push_mode(tmp_path):
    fleet = Fleet(tmp_path, {"Pusher": (9361, "work", done("pushed"))})
    out = await orchestrate(
        "t",
        await fleet.cards(),
        ScriptedChat(plan(("s1", "work", []))),
        fleet.http,
        policy=RunPolicy(push=fleet.receiver, poll_every=5),
    )
    assert out.state == "completed" and out.results["s1"].answer == "pushed"


async def test_agent_refuses_a_webhook_off_this_machine(tmp_path):
    fleet = Fleet(tmp_path, {"Pusher": (9371, "work", done("x"))})
    (card,) = await fleet.cards()
    elsewhere = PushReceiver("http://198.51.100.7/hook")
    with pytest.raises(Exception, match=r"(?i)push|url"):
        await ask_push(card, "go", fleet.http, elsewhere, poll_every=1)
    assert _states(fleet.journals["Pusher"]) == []  # no task was started


async def test_receiver_refuses_unknown_tokens():
    receiver = PushReceiver("http://127.0.0.1:9199/")
    token, inbox = receiver.expect()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=receiver.app), base_url="http://r"
    ) as c:
        body = {"statusUpdate": {"taskId": "t1", "status": {"state": "TASK_STATE_WORKING"}}}
        assert (await c.post("/", json=body)).status_code == 403
        wrong = {"X-A2A-Notification-Token": "no"}
        assert (await c.post("/", json=body, headers=wrong)).status_code == 403
        ok = await c.post("/", json=body, headers={"X-A2A-Notification-Token": token})
        assert ok.status_code == 204
    assert receiver.refused == 2 and inbox.qsize() == 1
    receiver.forget(token)
    assert receiver.expect()[0] != token


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://127.0.0.1:8000/hook", True),
        ("http://localhost/", True),
        ("https://[::1]:9/", True),
        ("http://127.5.5.5/", True),
        ("http://10.0.0.1/", False),
        ("http://192.168.1.10/", False),
        ("http://example.com/", False),
        ("http://127.0.0.1.example.com/", False),
        ("ftp://127.0.0.1/", False),
        ("http://127.0.0.1:notaport/", False),
        ("", False),
    ],
)
def test_webhooks_must_be_loopback(url, ok):
    assert is_loopback_url(url) is ok


def test_card_advertises_push(tmp_path):
    assert build_card(make_spec("X", 9381, "w", tmp_path)).capabilities.push_notifications


# --- agentmesh up: one dead agent must not take the fleet down ---------------------------


def fleet_member(tmp: Path, name: str, code: str):
    from agentmesh.fleet import Member

    m = Member(tmp / f"{name}.toml", make_spec(name, 9390, "w", tmp), tmp / "logs" / f"{name}.log")
    m.cmd = [sys.executable, "-c", code]
    return m


def test_up_restarts_a_dead_agent_and_keeps_the_others(tmp_path):
    import threading

    from agentmesh import fleet

    crasher = fleet_member(tmp_path, "Crasher", "import sys; print('up'); sys.exit(3)")
    steady = fleet_member(tmp_path, "Steady", "import time; time.sleep(60)")
    members = [crasher, steady]
    fleet.start(members)
    first_pid = steady.proc.pid
    stop = threading.Event()
    result: list[int] = []
    t = threading.Thread(
        target=lambda: result.append(
            fleet.watch(members, max_restarts=2, backoff=0.05, poll=0.05, stop_event=stop)
        )
    )
    t.start()
    try:
        end = time.monotonic() + 15
        while not crasher.down:
            assert time.monotonic() < end, "crasher never ran out of restarts"
            time.sleep(0.05)
        assert crasher.restarts == 2
        # The other agent is the same process, still running.
        assert steady.proc.pid == first_pid and steady.proc.poll() is None
        assert t.is_alive()
    finally:
        stop.set()
        t.join(15)
    assert result == [0]
    assert steady.proc.poll() is not None  # stopping the launcher stops everyone
    log = crasher.log.read_text(encoding="utf-8")
    assert log.count("up") == 3 and "--- restart 2 at" in log  # one log, appended


def test_up_exits_only_when_every_agent_is_down(tmp_path):
    from agentmesh import fleet

    members = [fleet_member(tmp_path, n, "import sys; sys.exit(1)") for n in ("A", "B")]
    fleet.start(members)
    t0 = time.monotonic()
    assert fleet.watch(members, max_restarts=0, poll=0.05) == 1
    assert time.monotonic() - t0 < 10
    assert all(m.down and m.restarts == 0 for m in members)
