"""M5: logins between agents, per-skill access rules, and Verifier-signed writes."""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest
from a2a.client import ClientConfig, create_client
from a2a.types import GetTaskRequest
from agentos.tools import Tool, ToolError
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from agentmesh.access import GRANT, Access, Grant, Refused, read_write_result
from agentmesh.auth import (
    PRINCIPALS_FILE,
    AttestationError,
    AuthConfig,
    AuthError,
    auth_headers,
    init_auth,
    load_token,
    sign_attestation,
    supported_sources,
    verify_attestation,
)
from agentmesh.cli import main
from agentmesh.client import ask, discover
from agentmesh.executor import RunOutcome
from agentmesh.orchestrator import ChatReply, RunPolicy, discover_all, orchestrate
from agentmesh.runtime import apply_grant
from agentmesh.server import build_app
from agentmesh.spec import AgentSpec, SkillSpec, load_spec

AGENTS = Path(__file__).resolve().parents[1] / "agents"
VERDICT = (
    '1. SUPPORTED - "llama.cpp can use a draft model" [https://a.example/doc/]\n'
    '2. CONTRADICTED - "it says otherwise" [https://b.example]\n'
    "3. NOT FOUND - the page doesn't say [https://c.example]\n"
    "1 supported, 1 contradicted, 1 not found"
)


# --- fixtures --------------------------------------------------------------------------


def done(text: str, seen: list | None = None):
    def run(task, progress, flag):
        if seen is not None:
            seen.append({"task": task, "grant": GRANT.get()})
        progress("working")
        return RunOutcome(answer=text, stop_reason="final_answer", steps=1)

    return run


def research_spec(tmp: Path) -> AgentSpec:
    return AgentSpec(
        name="Research",
        description="finds claims",
        port=9501,
        workspace=tmp / "research",
        skills=[SkillSpec(id="web_research", name="r", description="research")],
    )


def verifier_spec(tmp: Path) -> AgentSpec:
    return AgentSpec(
        name="Verifier",
        description="checks claims",
        port=9502,
        workspace=tmp / "verifier",
        attest_as="verifier",
        skills=[SkillSpec(id="claim_verification", name="v", description="verify")],
    )


def knowledge_spec(tmp: Path) -> AgentSpec:
    return AgentSpec(
        name="Knowledge",
        description="stores facts",
        port=9503,
        workspace=tmp / "knowledge",
        skills=[
            SkillSpec(
                id="knowledge_lookup",
                name="l",
                description="look up",
                tools=["search", "recall_facts"],
            ),
            SkillSpec(
                id="knowledge_write",
                name="w",
                description="store",
                callers=["orchestrator"],
                tools=["recall_facts", "add_fact"],
                require_attestation="verifier",
                attested_tools=["add_fact"],
            ),
        ],
    )


class Mesh:
    """Three agents in-process with logins on, and one HTTP client per principal."""

    def __init__(self, tmp: Path, verdict: str = VERDICT) -> None:
        self.dir = tmp / "secrets"
        init_auth(self.dir)
        auth = AuthConfig.load(self.dir)
        self.knowledge_saw: list[dict] = []
        apps = [
            (research_spec(tmp), done("1. Claim A [https://a.example/doc]")),
            (verifier_spec(tmp), done(verdict)),
            (knowledge_spec(tmp), done("stored fact 1", self.knowledge_saw)),
        ]
        self.mounts = {
            s.url: httpx.ASGITransport(app=build_app(s, runner, auth=auth)) for s, runner in apps
        }
        self.urls = list(self.mounts)
        self.research, self.verifier, self.knowledge = self.urls

    def client(self, principal: str | None = None, token: str | None = None):
        headers = {}
        if principal:
            headers = auth_headers(self.dir, principal)
        if token is not None:
            headers = {"Authorization": f"Bearer {token}"}
        return httpx.AsyncClient(mounts=dict(self.mounts), headers=headers, timeout=20)


@pytest.fixture
def mesh(tmp_path):
    return Mesh(tmp_path)


class ScriptedChat:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    async def __call__(self, messages, schema=None):
        return ChatReply(self.replies.pop(0))


def plan(*steps: tuple[str, str, list[str]]) -> str:
    import json

    return json.dumps(
        {
            "steps": [
                {"id": sid, "skill": skill, "input": f"do {skill}", "depends_on": deps}
                for sid, skill, deps in steps
            ]
        }
    )


# --- the auth folder -------------------------------------------------------------------


def test_init_stores_hashes_and_a_token_per_principal(tmp_path):
    root = tmp_path / "s"
    init_auth(root)
    auth = AuthConfig.load(root)
    user, orch = load_token(root, "user"), load_token(root, "orchestrator")
    assert user != orch
    text = (root / PRINCIPALS_FILE).read_text(encoding="utf-8")
    assert user not in text and orch not in text  # servers never hold a usable token
    assert auth.principal_for(user) == "user"
    assert auth.principal_for(orch) == "orchestrator"
    assert auth.principal_for("guess") is None
    assert (root / "keys" / "verifier.key").exists() and (root / "keys" / "verifier.pub").exists()


def test_init_refuses_to_overwrite_unless_forced(tmp_path):
    root = tmp_path / "s"
    init_auth(root)
    old = load_token(root, "user")
    with pytest.raises(AuthError, match="--force"):
        init_auth(root)
    init_auth(root, force=True)
    assert load_token(root, "user") != old


def test_bad_principal_names_are_refused(tmp_path):
    with pytest.raises(AuthError, match="bad principal"):
        init_auth(tmp_path / "s", ["../evil"])


# --- logins ----------------------------------------------------------------------------


async def test_card_is_public_and_says_a_login_is_needed(mesh):
    async with mesh.client() as anon:
        card = await discover(mesh.research, anon)
    assert "bearer" in card.security_schemes
    assert card.security_schemes["bearer"].http_auth_security_scheme.scheme == "bearer"


@pytest.mark.parametrize("token", [None, "not-a-token"])
async def test_jsonrpc_refuses_requests_without_a_known_token(mesh, token):
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "x"}]}},
    }
    async with mesh.client(token=token) as http:
        r = await http.post(mesh.research + "/", json=body)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")


async def test_a_logged_in_caller_gets_its_task_done(mesh):
    async with mesh.client("user") as http:
        run = await ask(await discover(mesh.research, http), "find claims", http)
    assert run.state == "completed" and run.answer.startswith("1. Claim A")


async def test_a_principal_cannot_read_another_principals_task(mesh):
    async with mesh.client("user") as user, mesh.client("orchestrator") as orch:
        card = await discover(mesh.research, user)
        run = await ask(card, "find claims", user)
        mine = await create_client(
            agent=card, client_config=ClientConfig(streaming=False, httpx_client=user)
        )
        assert (await mine.get_task(GetTaskRequest(id=run.task_id))).id == run.task_id
        theirs = await create_client(
            agent=card, client_config=ClientConfig(streaming=False, httpx_client=orch)
        )
        with pytest.raises(Exception, match=r"(?i)not found"):
            await theirs.get_task(GetTaskRequest(id=run.task_id))


# --- per-skill access ------------------------------------------------------------------


async def test_a_caller_not_on_the_list_is_rejected(mesh):
    async with mesh.client("user") as http:
        card = await discover(mesh.knowledge, http)
        run = await ask(card, "store this", http, metadata={"skill": "knowledge_write"})
    assert run.state == "rejected"
    assert "'user' may not use knowledge_write" in run.final_message
    assert mesh.knowledge_saw == []


async def test_an_agent_with_several_skills_needs_to_be_told_which(mesh):
    async with mesh.client("user") as http:
        card = await discover(mesh.knowledge, http)
        unnamed = await ask(card, "what is radix sort", http)
        unknown = await ask(card, "x", http, metadata={"skill": "drop_tables"})
        lookup = await ask(card, "what is radix sort", http, metadata={"skill": "knowledge_lookup"})
    assert unnamed.state == "rejected" and "say which skill" in unnamed.final_message
    assert unknown.state == "rejected" and "no skill 'drop_tables'" in unknown.final_message
    assert lookup.state == "completed"
    grant = mesh.knowledge_saw[-1]["grant"]
    assert grant.skill == "knowledge_lookup" and grant.caller == "user"
    assert not grant.allows_tool("domaingraph__add_fact")  # a lookup can't write


async def test_a_write_without_a_signed_verdict_is_rejected(mesh):
    async with mesh.client("orchestrator") as http:
        card = await discover(mesh.knowledge, http)
        run = await ask(
            card, "store: X [https://a.example]", http, metadata={"skill": "knowledge_write"}
        )
    assert run.state == "rejected"
    assert "no signed verifier result" in run.final_message
    assert mesh.knowledge_saw == []


async def test_a_forged_or_altered_verdict_is_rejected(mesh):
    forged = sign_attestation(
        Ed25519PrivateKey.generate(),
        "verifier",
        agent="Verifier",
        task="t",
        caller="orchestrator",
        input="check",
        answer=VERDICT,
    )
    real = sign_attestation(
        AuthConfig.load(mesh.dir).signing_key("verifier"),
        "verifier",
        agent="Verifier",
        task="t",
        caller="orchestrator",
        input="check",
        answer=VERDICT,
    )
    body, sig = real.split(".")
    altered = body[:-4] + ("AAAA" if not body.endswith("AAAA") else "BBBB") + "." + sig
    async with mesh.client("orchestrator") as http:
        card = await discover(mesh.knowledge, http)
        for token in (forged, altered):
            run = await ask(
                card,
                "store it",
                http,
                metadata={"skill": "knowledge_write", "attestations": [token]},
            )
            assert run.state == "rejected"
            assert "no valid verifier signature" in run.final_message
    assert mesh.knowledge_saw == []


# --- the whole chain -------------------------------------------------------------------


async def test_verifier_signs_and_knowledge_stores_only_what_it_supported(mesh):
    async with mesh.client("orchestrator") as http:
        cards, errors = await discover_all(mesh.urls, http)
        assert errors == []
        out = await orchestrate(
            "research and store",
            cards,
            ScriptedChat(
                plan(
                    ("s1", "web_research", []),
                    ("s2", "claim_verification", ["s1"]),
                    ("s3", "knowledge_write", ["s2"]),
                ),
                "done",
            ),
            http,
            policy=RunPolicy(retries=0, backoff=0),
        )
    assert out.state == "completed", out.message
    assert out.results["s1"].attestation == ""  # Research doesn't sign
    signed = out.results["s2"].attestation
    payload = verify_attestation(
        signed, AuthConfig.load(mesh.dir).public_key("verifier"), "verifier"
    )
    assert payload["answer"] == VERDICT and payload["caller"] == "orchestrator"
    assert out.results["s3"].attestations == [signed]
    grant = mesh.knowledge_saw[0]["grant"]
    assert grant.skill == "knowledge_write" and grant.caller == "orchestrator"
    assert dict(grant.budget) == {"https://a.example/doc": 1}
    assert grant.signed == [payload["task"]]


async def test_a_plan_that_skips_the_verifier_cannot_write(mesh):
    async with mesh.client("orchestrator") as http:
        cards, _ = await discover_all(mesh.urls, http)
        out = await orchestrate(
            "research and store",
            cards,
            ScriptedChat(
                plan(("s1", "web_research", []), ("s2", "knowledge_write", ["s1"])), "done"
            ),
            http,
            policy=RunPolicy(retries=0, backoff=0),
        )
    assert out.state == "failed"
    assert out.results["s2"].attempts[0].state == "rejected"
    assert "no signed verifier result" in out.results["s2"].attempts[0].message
    assert mesh.knowledge_saw == []


async def test_a_verdict_that_supports_nothing_cannot_write(tmp_path):
    mesh = Mesh(tmp_path, verdict='1. CONTRADICTED - "no" [https://b.example]')
    async with mesh.client("orchestrator") as http:
        cards, _ = await discover_all(mesh.urls, http)
        out = await orchestrate(
            "t",
            cards,
            ScriptedChat(
                plan(("s1", "claim_verification", []), ("s2", "knowledge_write", ["s1"])), "x"
            ),
            http,
            policy=RunPolicy(retries=0, backoff=0),
        )
    assert out.results["s2"].attempts[0].state == "rejected"
    assert "marked no claim with a source SUPPORTED" in out.results["s2"].attempts[0].message


# --- attestations and verdicts ---------------------------------------------------------


def test_attestation_round_trip_and_its_failures(tmp_path):
    init_auth(tmp_path / "s", signers=["verifier", "other"])
    auth = AuthConfig.load(tmp_path / "s")
    key, pub = auth.signing_key("verifier"), auth.public_key("verifier")
    t = sign_attestation(key, "verifier", agent="V", task="t1", caller="o", input="i", answer="a")
    assert verify_attestation(t, pub, "verifier")["task"] == "t1"
    with pytest.raises(AttestationError, match="not 'other'"):
        verify_attestation(t, pub, "other")
    with pytest.raises(AttestationError, match="signature"):
        verify_attestation(t, auth.public_key("other"), "verifier")
    with pytest.raises(AttestationError, match="limit"):
        verify_attestation(t, pub, "verifier", now=time.time() + 7200)
    with pytest.raises(AttestationError, match="not an attestation"):
        verify_attestation("garbage", pub, "verifier")


def test_only_supported_lines_count_as_sources():
    text = (
        '1. SUPPORTED - "a" [https://A.example/x/]\n'
        '2. **SUPPORTED** - "b" [https://a.example/x#part]\n'
        '3. CONTRADICTED - "c" [https://b.example]\n'
        "4. NOT FOUND - nothing [https://c.example]\n"
        "5. UNVERIFIABLE - no source given\n"
        "The SUPPORTED claims are 1 and 2 [https://d.example]\n"
    )
    assert dict(supported_sources(text)) == {"https://a.example/x": 2}


def test_supported_sources_accepts_angle_bracket_urls():
    # Verifier sometimes wraps URLs in angle brackets: [<https://...>]
    from agentmesh.auth import supported_claims as sc

    text = (
        '1. SUPPORTED - "in an AVL tree heights differ by at most one" [<https://en.wikipedia.org/wiki/AVL_tree>]\n'
        '2. SUPPORTED - "balance factor is height difference" [<https://en.wikipedia.org/wiki/AVL_tree>]\n'
        '3. CONTRADICTED - "wrong" [<https://other.example>]\n'
    )
    sources = supported_sources(text)
    assert sources["https://en.wikipedia.org/wiki/AVL_tree"] == 2
    assert "https://other.example" not in sources

    claims = sc(
        "1. claim a [<https://en.wikipedia.org/wiki/AVL_tree>]\n2. claim b [<https://en.wikipedia.org/wiki/AVL_tree>]\n",
        text,
    )
    assert len(claims) == 2
    assert all(url == "https://en.wikipedia.org/wiki/AVL_tree" for url, _ in claims)


def test_supported_sources_accepts_double_numbered_lines():
    # Verifier prepends its own line number when the orchestrator already numbered
    # the claims, producing "1. 1. SUPPORTED - ..." -- both numbering patterns must match.
    from agentmesh.auth import supported_claims as sc

    text = (
        '1. 1. SUPPORTED - "AVL trees are self-balancing BSTs" [https://en.wikipedia.org/wiki/AVL_tree]\n'
        '2. 2. SUPPORTED - "balance factor is height difference" [https://en.wikipedia.org/wiki/AVL_tree]\n'
        '3. 3. CONTRADICTED - "wrong claim" [https://other.example]\n'
    )
    sources = supported_sources(text)
    assert sources["https://en.wikipedia.org/wiki/AVL_tree"] == 2
    assert "https://other.example" not in sources

    inp = (
        "1. AVL trees are self-balancing BSTs [https://en.wikipedia.org/wiki/AVL_tree]\n"
        "2. balance factor is height difference [https://en.wikipedia.org/wiki/AVL_tree]\n"
    )
    claims = sc(inp, text)
    assert len(claims) == 2


CLAIMS_IN = (
    "Check these claims:\n"
    "1. llama.cpp supports speculative decoding with a draft model. [https://a.example/doc]\n"
    "2. llama.cpp was written in Rust. [https://a.example/doc]\n"
    "3. Qwen3-8B has 36 layers. [https://b.example/card]\n"
)
CLAIMS_OUT = (
    '1. SUPPORTED - "--model-draft sets the draft model" [https://a.example/doc]\n'
    '2. CONTRADICTED - "written in C/C++" [https://a.example/doc]\n'
    '3. SUPPORTED - "Number of Layers: 36" [https://b.example/card]\n'
)


def test_supported_claims_pair_each_verdict_with_the_claim_it_checked():
    from agentmesh.auth import supported_claims

    got = supported_claims(CLAIMS_IN, CLAIMS_OUT)
    assert [url for url, _ in got] == ["https://a.example/doc", "https://b.example/card"]
    assert "speculative decoding" in got[0][1] and "--model-draft" in got[0][1]
    assert "Rust" not in got[0][1] and "36 layers" in got[1][1]
    # A verdict whose input numbers no claims still gives the quote.
    assert "Number of Layers" in supported_claims("free text", CLAIMS_OUT)[1][1]


def test_statement_matches_rewording_but_not_another_claim():
    from agentmesh.access import statement_matches

    ev = "llama.cpp supports speculative decoding with a draft model. --model-draft sets it"
    assert statement_matches("llama.cpp supports speculative decoding", ev)
    assert statement_matches("Speculative decoding is supported by llama.cpp (draft model)", ev)
    assert not statement_matches("llama.cpp was written in Rust", ev)
    layers = "Qwen3-8B has 36 layers. Number of Layers: 36"
    assert statement_matches("Qwen3-8B has 36 layers", layers)
    assert not statement_matches("Qwen3-8B has 40 layers", layers)  # a number must match
    assert not statement_matches("the", layers)  # nothing to check


def test_a_write_must_state_a_claim_the_verifier_supported(tmp_path):
    """The URL used to be enough: any statement could ride on a source that supported
    some other claim. Now it must be one the Verifier checked with that source."""
    init_auth(tmp_path / "s")
    auth = AuthConfig.load(tmp_path / "s")
    token = sign_attestation(
        auth.signing_key("verifier"),
        "verifier",
        agent="Verifier Agent",
        task="v1",
        caller="orchestrator",
        input=CLAIMS_IN,
        answer=CLAIMS_OUT,
    )
    spec = load_spec(AGENTS / "knowledge.toml")
    grant = Access(spec, auth).admit("orchestrator", "knowledge_write", [token])
    stored = []
    (add,) = apply_grant([_tool("add_fact", lambda a: stored.append(a) or {"id": 1})], grant)
    with pytest.raises(ToolError, match="did not check this statement"):
        add.fn({"statement": "llama.cpp was written in Rust", "source": "https://a.example/doc"})
    with pytest.raises(ToolError, match="did not check this statement"):
        add.fn({"statement": "Qwen3-8B has 40 layers", "source": "https://b.example/card"})
    add.fn(
        {"statement": "llama.cpp supports speculative decoding", "source": "https://a.example/doc"}
    )
    add.fn({"statement": "Qwen3-8B has 36 layers.", "source": "https://b.example/card"})
    assert len(stored) == 2
    with pytest.raises(ToolError, match="already stored"):
        add.fn(
            {
                "statement": "llama.cpp supports speculative decoding",
                "source": "https://a.example/doc",
            }
        )


def _tool(name: str, fn) -> Tool:
    return Tool(name=name, description=name, args_model=None, fn=fn, input_schema={}, source="mcp")


def test_grant_narrows_tools_and_gates_attested_writes():
    stored = []
    tools = [
        _tool("domaingraph__recall_facts", lambda a: "[]"),
        _tool("domaingraph__add_fact", lambda a: stored.append(a) or {"id": len(stored)}),
        _tool("domaingraph__forget_fact", lambda a: "gone"),
    ]
    grant = Grant("orchestrator", "knowledge_write", ["recall_facts", "add_fact"], ["add_fact"])
    grant.budget.update({"https://a.example/doc": 1})
    kept = {t.name: t for t in apply_grant(tools, grant)}
    assert sorted(kept) == ["domaingraph__add_fact", "domaingraph__recall_facts"]
    add = kept["domaingraph__add_fact"].fn
    with pytest.raises(ToolError, match="must cite its source"):
        add({"statement": "x"})
    with pytest.raises(ToolError, match="did not mark any claim"):
        add({"statement": "x", "details": {"source": "https://evil.example"}})
    assert add({"statement": "x", "details": {"source": "https://a.example/doc/"}}) == {"id": 1}
    with pytest.raises(ToolError, match="already stored"):
        add({"statement": "y", "details": {"source": "https://a.example/doc"}})
    assert len(stored) == 1
    assert apply_grant(tools, None) == tools  # no logins, no change


def test_a_duplicate_write_is_named_as_one_to_the_model():
    results = iter(['{"id": 7, "statement": "x", "new": true}', '{"id": 7, "new": false}'])
    grant = Grant("o", "w", None, ["add_fact"])
    grant.budget.update({"https://a.example": 2})
    (add,) = apply_grant([_tool("add_fact", lambda a: next(results))], grant)
    first = add.fn({"statement": "x", "source": "https://a.example"})
    assert "ALREADY STORED" not in first
    second = add.fn({"statement": "x", "source": "https://a.example"})
    assert second.startswith("ALREADY STORED") and "id 7" in second
    assert [(w.id, w.new) for w in grant.writes] == [(7, True), (7, False)]
    report = grant.write_report()
    assert "id 7: stored as new: x" in report
    assert "id 7: already stored, not added again: x" in report


def test_write_result_reading_tolerates_odd_results():
    rec = read_write_result({"statement": "  a   b "}, "not json")
    assert (rec.statement, rec.id, rec.new) == ("a b", None, None)
    rec = read_write_result({"statement": "a"}, {"id": 3, "new": False})
    assert (rec.id, rec.new) == (3, False)
    assert Grant("o", "w").write_report() == ""


async def test_knowledge_answer_carries_the_tool_write_record(tmp_path):
    """The model says "stored" for a duplicate; the code-built record says otherwise."""
    dir_ = tmp_path / "secrets"
    init_auth(dir_)
    auth = AuthConfig.load(dir_)

    def runner(task, progress, flag):
        grant = GRANT.get()
        (add,) = apply_grant([_tool("add_fact", lambda a: '{"id": 296, "new": false}')], grant)
        add.fn({"statement": "llama.cpp can use a draft model", "source": "https://a.example/doc"})
        return RunOutcome(answer="Successfully stored fact 296.", stop_reason="final_answer")

    spec = knowledge_spec(tmp_path)
    app = build_app(spec, runner, auth=auth)
    token = sign_attestation(
        auth.signing_key("verifier"),
        "verifier",
        agent="Verifier",
        task="t",
        caller="orchestrator",
        input="check",
        answer=VERDICT,
    )
    mounts = {spec.url: httpx.ASGITransport(app=app)}
    headers = auth_headers(dir_, "orchestrator")
    async with httpx.AsyncClient(mounts=mounts, headers=headers, timeout=20) as http:
        card = await discover(spec.url, http)
        run = await ask(
            card,
            "store it",
            http,
            metadata={"skill": "knowledge_write", "attestations": [token]},
        )
    assert run.state == "completed"
    assert "Write record (from the tool results):" in run.answer
    assert "id 296: already stored, not added again" in run.answer


async def test_the_final_answer_ends_with_the_write_record(tmp_path):
    """The summary model says "stored"; the orchestrator appends the code-built record."""
    m = Mesh(tmp_path)
    record = "Stored it.\n\nWrite record (from the tool results):\n- id 296: already stored, x"
    knowledge = knowledge_spec(tmp_path)
    m.mounts[knowledge.url] = httpx.ASGITransport(
        app=build_app(knowledge, done(record), auth=AuthConfig.load(m.dir))
    )
    async with m.client("orchestrator") as http:
        cards, _ = await discover_all(m.urls, http)
        out = await orchestrate(
            "research and store",
            cards,
            ScriptedChat(
                plan(
                    ("s1", "web_research", []),
                    ("s2", "claim_verification", ["s1"]),
                    ("s3", "knowledge_write", ["s2"]),
                ),
                "The fact was stored.",
            ),
            http,
            policy=RunPolicy(retries=0, backoff=0),
            allow_writes=True,
        )
    assert out.state == "completed", out.message
    assert out.answer.startswith("The fact was stored.")
    assert out.answer.endswith(
        "Write record (from the tool results):\ns3 (Knowledge):\n- id 296: already stored, x"
    )


def test_a_failed_write_does_not_use_up_the_claim():
    def boom(args):
        raise RuntimeError("neo4j down")

    grant = Grant("o", "w", None, ["add_fact"])
    grant.budget.update({"https://a.example": 1})
    (add,) = apply_grant([_tool("add_fact", boom)], grant)
    with pytest.raises(RuntimeError):
        add.fn({"source": "https://a.example"})
    assert grant.budget["https://a.example"] == 1


# --- the real specs and the CLI --------------------------------------------------------


def test_the_fleet_specs_carry_the_m5_rules():
    knowledge = load_spec(AGENTS / "knowledge.toml")
    rules = {s.id: s for s in knowledge.skills}
    assert rules["knowledge_write"].callers == ["orchestrator"]
    assert rules["knowledge_write"].require_attestation == "verifier"
    assert rules["knowledge_write"].attested_tools == ["add_fact"]
    assert "add_fact" not in (rules["knowledge_lookup"].tools or [])
    assert load_spec(AGENTS / "verifier.toml").attest_as == "verifier"


def test_attested_tools_need_a_signer():
    with pytest.raises(ValueError, match="needs require_attestation"):
        SkillSpec(id="w", name="w", description="w", attested_tools=["add_fact"])


def test_an_agent_that_checks_signatures_needs_the_public_key(tmp_path):
    init_auth(tmp_path / "s", signers=[])
    with pytest.raises(AuthError, match=r"verifier\.pub"):
        Access(knowledge_spec(tmp_path), AuthConfig.load(tmp_path / "s"))


def test_admit_without_a_login_is_refused(tmp_path):
    init_auth(tmp_path / "s")
    access = Access(research_spec(tmp_path), AuthConfig.load(tmp_path / "s"))
    with pytest.raises(Refused, match="not logged in"):
        access.admit("", "web_research", [])
    assert access.admit("user", "", []).skill == "web_research"  # its only skill


def test_serve_refuses_to_start_without_an_auth_folder(tmp_path, capsys):
    code = main(["serve", str(AGENTS / "research.toml"), "--auth-dir", str(tmp_path / "none")])
    assert code == 2
    assert "agentmesh auth init" in capsys.readouterr().err


def test_auth_init_command(tmp_path, capsys):
    root = tmp_path / "s"
    assert main(["auth", "init", "--auth-dir", str(root), "--principal", "user"]) == 0
    assert (root / "tokens" / "user.token").exists()
    assert not (root / "tokens" / "orchestrator.token").exists()
    assert main(["auth", "init", "--auth-dir", str(root)]) == 2
