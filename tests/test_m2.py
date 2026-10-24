"""M2: the four specialist specs, per-agent tools, the answer guard, webtools and the fleet."""

from __future__ import annotations

import json
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
    unfetched_citations,
    unfetched_numbers,
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


CARD = tool_msg(
    "web__fetch_url",
    '{"url": "https://huggingface.co/Qwen/Qwen3-8B", "title": "Qwen3-8B", "text": '
    '"- Number of Layers: 36\\n- Number of Attention Heads (GQA): 32 for Q and 8 for KV\\n'
    '- Context Length: 32,768 natively and 131,072 tokens with YaRN.", "offset": 0}',
)
OLLAMA = tool_msg("web__fetch_url", '{"text": "- < 24 GiB VRAM: 4k context"}')
TASK_MSG = Message(role="user", content="Find Qwen3-8B's values for a GPU under 24 GiB.")


def test_unfetched_numbers():
    msgs = [TASK_MSG, CARD, OLLAMA]
    # Live, Research gave the head dimension as 5120, a number the card never states.
    answer = "1. Layers: 36\n2. KV heads: 8\n3. Head dimension: 5120\n4. Context: 32768 tokens"
    assert unfetched_numbers(answer, msgs, "fetch_url") == ["5120"]
    # "4k" on a page states 4,096; the task's own numbers and 0-2 don't need a page;
    # numbers in URLs and model names aren't values.
    ok = "Ollama: 4,096 tokens below 24 GiB, 1 window, x2 (Qwen3-8B)."
    assert unfetched_numbers(ok, msgs, "fetch_url") == []
    assert unfetched_numbers("See https://x.io/v3/123", msgs, "fetch_url") == []
    # Search snippets are not pages, and nothing fetched yet means nothing to check.
    snippet = tool_msg("web__web_search", '{"results": [{"snippet": "head_dim 5120"}]}')
    assert unfetched_numbers("5120", [snippet, CARD], "fetch_url") == ["5120"]
    assert unfetched_numbers("5120", [snippet], "fetch_url") == []
    # A failed fetch's error may be quoted.
    failed = tool_msg("web__fetch_url", "ERROR: HTTP 403 from https://x.io")
    assert unfetched_numbers("The page gave HTTP 403.", [failed, CARD], "fetch_url") == []


def test_number_guard_sends_unfetched_number_back():
    bad = Message(role="assistant", content="Head dimension: 5120")
    good = Message(role="assistant", content="The card gives no head dimension.")
    inner = ScriptedLLM([bad, good])
    resp = AnswerGuardLLM(inner, check_numbers="fetch_url").chat([TASK_MSG, CARD], [])
    assert resp.message is good
    assert "5120" in inner.seen[1][-1].content
    assert "config.json" in inner.seen[1][-1].content
    # Off unless the spec names the fetch tool.
    assert AnswerGuardLLM(ScriptedLLM([bad])).chat([TASK_MSG, CARD], []).message is bad


def test_research_checks_numbers():
    assert load_spec(AGENTS / "research.toml").check_numbers == "fetch_url"


def _fetch(url: str, content: str, call_id: str) -> list[Message]:
    """An assistant fetch_url call and its tool result, as the agent's history holds them."""
    call = ToolCall(id=call_id, name="web__fetch_url", arguments={"url": url})
    return [
        Message(role="assistant", content="", tool_calls=[call]),
        Message(role="tool", content=content, tool_name="web__fetch_url", tool_call_id=call_id),
    ]


PDF_URL = "https://developer.nvidia.com/downloads/assets/ace/model_card/qwen3-8b-instruct.pdf"


def test_unfetched_citations():
    task = Message(role="user", content="Qwen3-8B's license? Start at https://qwenlm.github.io/")
    hf = _fetch(
        "https://huggingface.co/Qwen/Qwen3-8B",
        '{"url": "https://huggingface.co/Qwen/Qwen3-8B", "text": "x"}',
        "a",
    )
    # The 2026-10-23b rerun: the PDF's fetch failed, and the answer cited it anyway.
    pdf = _fetch(PDF_URL, "ERROR: unsupported content type 'application/pdf'", "b")
    msgs = [task, *hf, *pdf]
    answer = f"Apache 2.0 [https://huggingface.co/Qwen/Qwen3-8B/], 32,768 tokens [{PDF_URL}]."
    assert unfetched_citations(answer, msgs, "fetch_url") == [PDF_URL]
    # Never fetched at all (the ssdnodes citation), versus the task's own URL and a local
    # address the answer reports as a value.
    other = "Default 4096 [https://www.ssdnodes.com/learn/x]. Ollama: http://localhost:11434"
    assert unfetched_citations(other + " https://qwenlm.github.io", msgs, "fetch_url") == [
        "https://www.ssdnodes.com/learn/x"
    ]


def test_unfetched_citations_accepts_redirects_and_paired_pages():
    final = "https://developer.download.nvidia.com/assets/ace/model_card/Qwen3-8B-Instruct.pdf"
    redirected = _fetch(PDF_URL, json.dumps({"url": final, "text": "Context Length: 32,768"}), "a")
    paired = _fetch(
        "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json",
        json.dumps(
            {
                "url": "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json",
                "text": "[This model's config.json, from https://huggingface.co/Qwen/Qwen3-8B/"
                "blob/main/config.json]\n{}\n\n[The model card, from "
                "https://huggingface.co/Qwen/Qwen3-8B]\nApache 2.0",
            }
        ),
        "b",
    )
    comma = _fetch("https://llm-explorer.com/model/Qwen/Qwen3-8B,6sQs", '{"text": "x"}', "c")
    answer = (
        f"[{PDF_URL}] [{final}] https://huggingface.co/Qwen/Qwen3-8B "
        "https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json "
        "https://llm-explorer.com/model/Qwen/Qwen3-8B,6sQs"
    )
    assert unfetched_citations(answer, [*redirected, *paired, *comma], "fetch_url") == []


def test_citation_guard_sends_unfetched_citation_back_and_survives_compaction():
    task = Message(role="user", content="Qwen3-8B's license?")
    hf = _fetch("https://huggingface.co/Qwen/Qwen3-8B", '{"text": "Apache 2.0"}', "a")
    bad = Message(role="assistant", content=f"Apache 2.0 [{PDF_URL}]")
    good = Message(role="assistant", content="Apache 2.0 [https://huggingface.co/Qwen/Qwen3-8B]")
    step = Message(
        role="assistant", content="", tool_calls=[ToolCall(id="z", name="x", arguments={})]
    )
    inner = ScriptedLLM([step, bad, good])
    guard = AnswerGuardLLM(inner, check_citations="fetch_url")
    guard.chat([task, *hf], [])  # a step that saw the fetch
    # Compacted: the fetch's messages are gone from the history the final call sees.
    resp = guard.chat([task, Message(role="user", content="Summary of earlier steps.")], [])
    assert resp.message is good
    assert PDF_URL in inner.seen[-1][-1].content
    assert "did not fetch successfully" in inner.seen[-1][-1].content
    # Off unless the spec names the fetch tool.
    assert AnswerGuardLLM(ScriptedLLM([bad])).chat([task], []).message is bad


def test_unfetched_numbers_reads_the_pages_each_claim_cites():
    # The 2026-10-23b rerun: claim 1 gave Ollama's 4,096 citing ssdnodes.com (never
    # fetched). 4096 is also Qwen3-8B's hidden_size, in the config.json fetched with the
    # card claim 2 cites, so a check against every fetched page passed it.
    task = Message(role="user", content="Ollama's default context below 24 GiB, and Qwen3-8B's?")
    card = "https://huggingface.co/Qwen/Qwen3-8B"
    paired = _fetch(
        card,
        json.dumps(
            {
                "url": card,
                "text": f"[This model's config.json, from {card}/blob/main/config.json]\n"
                f'{{"hidden_size": 4096}}\n\n[The model card, from {card}]\n'
                "Context Length: 32,768 natively",
            }
        ),
        "a",
    )
    other = _fetch("https://therouter.ai/q", '{"text": "Qwen3-8B: 32,768 tokens"}', "b")
    msgs = [task, *paired, *other]
    answer = (
        "Summary: Ollama uses 4,096 tokens; Qwen3-8B has 32,768.\n"
        "1. Ollama's default is 4,096 tokens. [https://www.ssdnodes.com/learn/x]\n"
        f"2. Qwen3-8B's native context is 32,768 tokens. [{card}]\n"
    )
    assert unfetched_numbers(answer, msgs, "fetch_url") == ["4,096"]
    # Citing the page that states it passes; so does the summary, whose numbers may come
    # from any page the answer cites.
    fixed = answer.replace("[https://www.ssdnodes.com/learn/x]", f"[{card}/blob/main/config.json]")
    assert unfetched_numbers(fixed, msgs, "fetch_url") == []
    # A claim citing a fetched page that lacks its number fails, even if another page has it.
    wrong = "1. Qwen3-8B: 32,768 tokens [https://therouter.ai/q]\n2. Head dim 4096 [https://therouter.ai/q]"
    assert unfetched_numbers(wrong, msgs, "fetch_url") == ["4096"]
    # No citation at all: any fetched page counts, as before. Local addresses aren't citations.
    assert unfetched_numbers("Ollama: 4096 on http://localhost:11434", msgs, "fetch_url") == []


def test_research_checks_citations():
    assert load_spec(AGENTS / "research.toml").check_citations == "fetch_url"


def test_web_search_drops_empty_hits_ranks_model_pages_first(monkeypatch):
    import sys
    import types

    from agentmesh import webtools

    hits = [{"title": "", "href": "", "body": ""}]
    asked = []

    def text(q, max_results):
        asked.append(max_results)
        return hits

    fake = types.SimpleNamespace(DDGS=lambda: types.SimpleNamespace(text=text))
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    with pytest.raises(ToolError, match="no results"):
        webtools.web_search("Ollama default context window")
    hits.append({"title": "Ollama", "href": "https://docs.ollama.com/faq", "body": "4096"})
    assert webtools.web_search("q", max_results=3)["results"] == [
        {"title": "Ollama", "url": "https://docs.ollama.com/faq", "snippet": "4096"}
    ]
    webtools.web_search("q", max_results=50)
    assert asked == [10, 10, 10]  # ten asked for, so a model page ranked 6-10 can move up
    # At least 5 come back (3 often missed the model card), at most 10.
    hits[:] = [{"title": str(i), "href": f"https://x.io/{i}", "body": ""} for i in range(12)]
    assert len(webtools.web_search("q", max_results=3)["results"]) == 5
    assert len(webtools.web_search("q", max_results=50)["results"]) == 10
    # A model's own Hugging Face page goes first, even from 7th; the others keep their order.
    hits[:] = [
        {"title": "PDF", "href": "https://developer.nvidia.com/qwen3-8b.pdf", "body": ""},
        {"title": "Specs", "href": "https://apxml.com/models/qwen3-8b", "body": ""},
        *({"title": "x", "href": f"https://x.io/{i}", "body": ""} for i in range(4)),
        {"title": "Card", "href": "https://huggingface.co/Qwen/Qwen3-8B", "body": ""},
    ]
    urls = [r["url"] for r in webtools.web_search("Qwen3-8B model card")["results"]]
    assert urls == [
        "https://huggingface.co/Qwen/Qwen3-8B",
        "https://developer.nvidia.com/qwen3-8b.pdf",
        "https://apxml.com/models/qwen3-8b",
        "https://x.io/0",
        "https://x.io/1",
    ]


def test_model_names_in_search_queries():
    from agentmesh.webtools import model_names

    assert model_names("Qwen3-8B native context length") == ["Qwen3-8B"]
    assert model_names("qwen3:8b context window") == ["qwen3-8b"]
    assert model_names("meta-llama/Llama-3.1-8B config.json") == ["meta-llama/Llama-3.1-8B"]
    for q in ["fp16 KV cache on an H100", "Ollama default context window", "llama.cpp draft"]:
        assert model_names(q) == [], q


def test_find_model_page_takes_the_exact_name_from_the_hub():
    from agentmesh.webtools import find_model_page

    asked = []

    def hub(req):
        asked.append(req.url.params["search"])
        return httpx.Response(200, json=[{"id": "someone/Qwen3-8B-GGUF"}, {"id": "Qwen/Qwen3-8B"}])

    page = find_model_page("qwen3-8b", transport=httpx.MockTransport(hub))
    assert page["url"] == "https://huggingface.co/Qwen/Qwen3-8B" and asked == ["qwen3-8b"]
    assert find_model_page("Qwen3-4B", transport=httpx.MockTransport(hub)) is None
    down = httpx.MockTransport(lambda r: httpx.Response(503))
    assert find_model_page("Qwen3-8B", transport=down) is None


def test_search_adds_the_model_page_when_no_hit_is_one(monkeypatch):
    import sys
    import types

    from agentmesh import webtools

    hits = [{"title": "List", "href": "https://contextwindows.dev/qwen3-8b", "body": "40960"}]
    fake = types.SimpleNamespace(DDGS=lambda: types.SimpleNamespace(text=lambda q, **kw: hits))
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    looked = []

    def page(name):
        looked.append(name)
        return {
            "title": "Qwen/Qwen3-8B",
            "url": "https://huggingface.co/Qwen/Qwen3-8B",
            "snippet": "",
        }

    monkeypatch.setattr(webtools, "find_model_page", page)
    urls = [r["url"] for r in webtools.search("Qwen3-8B native context length")]
    assert urls == ["https://huggingface.co/Qwen/Qwen3-8B", "https://contextwindows.dev/qwen3-8b"]
    # No lookup when a model page is already a hit, or the query names no model.
    hits.append({"title": "Card", "href": "https://huggingface.co/Qwen/Qwen3-8B", "body": ""})
    webtools.search("Qwen3-8B native context length")
    webtools.search("Ollama default context window")
    assert looked == ["Qwen3-8B"]


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


def test_fetch_keeps_lead_paragraph_rendered_as_span():
    # Mintlify-style docs page: the lead paragraphs are <span data-as="p"> before the first
    # heading. Trafilatura alone drops them (the MCP intro lost its "USB-C port" line).
    body = (
        "<html><head><title>What is MCP?</title></head><body><main>"
        '<div id="content">'
        '<span data-as="p">MCP is an open-source standard for connecting AI applications.</span>'
        '<span data-as="p">Think of MCP like a USB-C port for AI applications. Just as USB-C '
        "provides a standardized way to connect electronic devices, MCP provides one for AI.</span>"
        "<h2>What can MCP enable?</h2>"
        "<p>Agents can access your calendar and your notes, acting as a personal assistant, "
        "and chatbots can connect to many databases across an organization.</p>"
        "<h2>Why does MCP matter?</h2>"
        "<p>Depending on where you sit in the ecosystem, MCP can have a range of benefits "
        "for developers, for AI applications and for the people who use them.</p>"
        "</div></main></body></html>"
    )

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})

    page = fetch_page(
        "http://public.test/intro",
        resolve=resolver("93.184.215.14"),
        transport=httpx.MockTransport(handler),
    )
    assert "USB-C port" in page.text
    assert "Depending on where you sit" in page.text


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


def test_hugging_face_model_page_and_config_json_come_together():
    from agentmesh.webtools import Page, with_model_config

    asked = []

    def fetch(url):
        asked.append(url)
        return Page(url=url, title=None, text='{\n  "head_dim": 128\n}')

    card = Page(url="https://huggingface.co/Qwen/Qwen3-8B", title="Qwen3-8B", text="Layers: 36")
    out = with_model_config(card, fetch)
    assert asked == [
        "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json",
        "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/README.md",
    ]
    assert out.text.startswith(
        "[This model's config.json, from https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json]"
        "\n(max_position_embeddings is the longest position"
    )
    assert '"head_dim": 128' in out.text
    assert out.text.endswith(
        "[The model card, from https://huggingface.co/Qwen/Qwen3-8B]\nLayers: 36"
    )
    assert out.url == card.url

    # A fetched config.json gets the card after it (live, Research fetched only config.json
    # and took max_position_embeddings for the native context length). A blob URL is the
    # rendered view, so the raw file is fetched at the same revision.
    pages = {
        "https://huggingface.co/Qwen/Qwen3-8B/resolve/b96882/config.json": '{"head_dim": 128}',
        "https://huggingface.co/Qwen/Qwen3-8B": "Context Length: 32,768 natively",
        "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/README.md": (
            "---\nlicense: apache-2.0\n---\n# Qwen3-8B"
        ),
    }
    asked.clear()

    def by_url(url):
        asked.append(url)
        return Page(url=url, title=None, text=pages[url])

    blob = "https://huggingface.co/Qwen/Qwen3-8B/blob/b96882/config.json"
    view = Page(url=blob, title="config.json", text="head_dim 128 (rendered)")
    out = with_model_config(view, by_url, asked=blob)
    assert asked == list(pages)
    assert '{"head_dim": 128}' in out.text and out.text.endswith(
        "[The model card, from https://huggingface.co/Qwen/Qwen3-8B]\n"
        "Card metadata: license: apache-2.0\nContext Length: 32,768 natively"
    )
    assert out.url == blob

    # Other pages, deeper paths and a failed or non-JSON config.json leave the page alone.
    for url in [
        "https://example.com/Qwen/Qwen3-8B",
        "https://huggingface.co/Qwen/Qwen3-8B/tree/main",
    ]:
        page = Page(url=url, title=None, text="x")
        assert with_model_config(page, fetch) is page

    def fails(url):
        raise ToolError("HTTP 404")

    assert with_model_config(card, fails) is card
    html = lambda url: Page(url=url, title=None, text="<not json>")  # noqa: E731
    assert with_model_config(card, html) is card


def test_card_metadata_keeps_the_short_header_fields():
    from agentmesh.webtools import Page, card_metadata

    readme = (
        "---\nlanguage:\n- en\n- de\nlicense: llama3.1\nbase_model: meta-llama/Llama-3.1-8B\n"
        f'extra_gated_prompt: "{"LICENSE AGREEMENT " * 50}"\n'
        "extra_gated_button_content: Submit\n"
        "model-index:\n- name: x\n  results: []\ninference: false\n---\n# Card\n"
    )
    seen = []

    def fetch(url):
        seen.append(url)
        return Page(url=url, title=None, text=readme)

    line = card_metadata("meta-llama", "Llama-3.1-8B-Instruct", fetch)
    assert seen == [
        "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/README.md"
    ]
    # The gated-access form (with its licence agreement), the benchmark table and the bool
    # are left out.
    assert line == (
        "Card metadata: language: en, de; license: llama3.1; base_model: meta-llama/Llama-3.1-8B"
    )

    def fails(url):
        raise ToolError("HTTP 404")

    no_header = lambda url: Page(url=url, title=None, text="# Card only")  # noqa: E731
    bad_yaml = lambda url: Page(url=url, title=None, text="---\n: [\n---\n")  # noqa: E731
    for get in (fails, no_header, bad_yaml):
        assert card_metadata("a", "b", get) is None
