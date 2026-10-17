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

from agentmesh.access import GRANT, Access, Grant, Refused
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
