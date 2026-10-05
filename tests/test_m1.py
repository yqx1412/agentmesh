"""M1 end to end, in-process: the real A2A server and SDK client over an ASGI transport."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import httpx
import pytest
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import (
    CancelTaskRequest,
    GetTaskRequest,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    TaskState,
)

from agentmesh.client import ask, discover
from agentmesh.executor import Cancelled, RunOutcome
from agentmesh.server import build_app, build_card
from agentmesh.spec import AgentSpec, SkillSpec, SpecError, load_spec

AGENTS = Path(__file__).parents[1] / "agents"


def make_spec(tmp_path: Path) -> AgentSpec:
    return AgentSpec(
        name="Test Agent",
        description="for tests",
        port=9999,
        workspace=tmp_path,
        skills=[SkillSpec(id="echo", name="Echo", description="echoes", tags=["t"])],
    )


def http_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), timeout=10)


def fake_runner(answer="done", stop="final_answer", steps=("step 1: calling read_file",)):
    def run(task, progress, flag):
        for s in steps:
            progress(s)
        return RunOutcome(
            answer=f"{answer}: {task}",
            stop_reason=stop,
            steps=2,
            tool_calls=1,
            prompt_tokens=100,
            completion_tokens=20,
        )

    return run


def test_shipped_spec_loads():
    spec = load_spec(AGENTS / "general.toml")
    assert spec.host == "127.0.0.1"
    assert spec.workspace.is_absolute()
    assert {s.id for s in spec.skills} == {"workspace_files", "calculation"}


def test_spec_rejects_duplicate_skills(tmp_path):
    p = tmp_path / "a.toml"
    p.write_text(
        'name="a"\ndescription="d"\nport=1\nworkspace="w"\n'
        '[[skills]]\nid="x"\nname="x"\ndescription="x"\n'
        '[[skills]]\nid="x"\nname="y"\ndescription="y"\n'
    )
    with pytest.raises(SpecError, match="duplicate skill"):
        load_spec(p)


async def test_card_is_discoverable(tmp_path):
    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, fake_runner())) as http:
        card = await discover(spec.url, http)
    assert card.name == "Test Agent"
    assert [s.id for s in card.skills] == ["echo"]
    assert card.capabilities.streaming
    iface = card.supported_interfaces[0]
    assert (iface.protocol_binding, iface.url) == ("JSONRPC", "http://127.0.0.1:9999")
    assert build_card(spec) == card


@pytest.mark.parametrize("streaming", [True, False])
async def test_completed_lifecycle(tmp_path, streaming):
    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, fake_runner())) as http:
        card = await discover(spec.url, http)
        run = await ask(card, "hello", http, streaming=streaming)
    assert run.answer == "done: hello"
    assert run.stats["stop_reason"] == "final_answer"
    assert run.stats["prompt_tokens"] == 100
    assert isinstance(run.stats["prompt_tokens"], int)
    if streaming:
        assert run.states == ["submitted", "working", "completed"]
        assert run.progress == ["Starting the agent.", "step 1: calling read_file"]
    else:
        assert run.state == "completed"


async def test_no_final_answer_is_failed(tmp_path):
    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, fake_runner(answer="half", stop="max_steps"))) as http:
        run = await ask(await discover(spec.url, http), "x", http)
    assert run.state == "failed"
    assert "max_steps" in run.final_message
    assert run.answer == ""


async def test_runner_exception_is_failed(tmp_path):
    def boom(task, progress, flag):
        raise RuntimeError("model down")

    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, boom)) as http:
        run = await ask(await discover(spec.url, http), "x", http)
    assert run.states[-1] == "failed"
    assert "model down" in run.final_message


async def test_empty_request_is_rejected(tmp_path):
    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, fake_runner())) as http:
        run = await ask(await discover(spec.url, http), "   ", http)
    assert run.state == "rejected"


async def test_cancel_stops_the_worker(tmp_path):
    started, stopped = threading.Event(), threading.Event()

    def slow(task, progress, flag):
        started.set()
        if not flag.wait(5):
            raise AssertionError("never cancelled")
        stopped.set()
        raise Cancelled()

    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, slow)) as http:
        card = await discover(spec.url, http)
        client = await create_client(
            agent=card, client_config=ClientConfig(streaming=False, httpx_client=http)
        )
        req = SendMessageRequest(
            message=new_text_message("long job", role=Role.ROLE_USER),
            configuration=SendMessageConfiguration(return_immediately=True),
        )
        stream = client.send_message(req)
        first = await anext(stream)  # return_immediately: the submitted task
        await stream.aclose()
        task_id = first.task.id
        await asyncio.to_thread(started.wait, 5)
        await client.cancel_task(CancelTaskRequest(id=task_id))
        assert await asyncio.to_thread(stopped.wait, 5)
        for _ in range(50):
            task = await client.get_task(GetTaskRequest(id=task_id))
            if task.status.state == TaskState.TASK_STATE_CANCELED:
                break
            await asyncio.sleep(0.05)
        assert task.status.state == TaskState.TASK_STATE_CANCELED


async def test_plain_json_rpc_without_the_sdk(tmp_path):
    """Any A2A 1.0 client works: here, hand-written JSON-RPC over HTTP."""
    spec = make_spec(tmp_path)
    async with http_for(build_app(spec, fake_runner())) as http:
        card = (await http.get(f"{spec.url}/.well-known/agent-card.json")).json()
        assert card["skills"][0]["id"] == "echo"
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {
                "message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hi"}]}
            },
        }
        r = await http.post(spec.url + "/", json=body, headers={"A2A-Version": "1.0"})
    task = r.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["text"] == "done: hi"


async def test_agentos_runner_with_a_scripted_model(tmp_path, monkeypatch):
    """The real AgentOS loop and tools behind A2A; only the model is scripted."""
    from agentos.llm import ChatResponse, Message, ToolCall

    import agentmesh.runtime as runtime

    class ScriptedLLM:
        def __init__(self, model, *a, **kw):
            self.model = model
            self.replies = [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(name="write_file", arguments={"path": "out.txt", "content": "42"})
                    ],
                ),
                Message(role="assistant", content="Wrote 42 to out.txt."),
            ]

        def chat(self, messages, tools):
            time.sleep(0.01)
            return ChatResponse(message=self.replies.pop(0), prompt_tokens=7, completion_tokens=3)

        def close(self):
            pass

    monkeypatch.setattr(runtime, "OllamaLLM", ScriptedLLM)
    spec = make_spec(tmp_path / "ws")
    async with http_for(build_app(spec, runtime.agentos_runner(spec, tmp_path / "traces"))) as h:
        run = await ask(await discover(spec.url, h), "write 42 to out.txt", h)
    assert run.state == "completed"
    assert run.answer == "Wrote 42 to out.txt."
    assert (tmp_path / "ws" / "out.txt").read_text() == "42"
    assert run.stats["tool_calls"] == 1 and run.stats["prompt_tokens"] == 14
    assert "step 1: calling write_file" in run.progress
    assert "write_file: ok" in run.progress
    assert len(list((tmp_path / "traces").glob("*.jsonl"))) == 1
