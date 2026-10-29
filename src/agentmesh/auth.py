"""M5: logins between agents, and signed results.

**Logins.** Every caller is a *principal* (``user``, ``orchestrator``, ...) with its own
random bearer token. An agent's JSON-RPC endpoint refuses a request without a known token
(HTTP 401); the Agent Card stays public so discovery still works. Servers keep only the
SHA-256 of each token, so a server's files never hold a token that logs in elsewhere.

**Signed results (attestations).** An agent can sign what it concluded: the Verifier signs
each completed task's input and answer with an Ed25519 key only it holds. Anyone with the
public key can check that the text came from the Verifier unchanged, and nobody else (the
orchestrator that relays it included) can produce one. The Knowledge agent uses this to
refuse writes that the Verifier didn't approve (see ``access.py``).

The auth folder (``agentmesh auth init``, default ``secrets/``, git-ignored)::

    principals.toml        name -> sha256 of its token      read by every agent server
    tokens/<name>.token    the token itself                 read only by that client
    keys/<name>.key        Ed25519 private key (PEM)        read only by the signing agent
    keys/<name>.pub        Ed25519 public key (PEM)         read by agents that check

All of this lives on one machine under one OS user, so the files are protected by where
they are, not by OS permissions per agent: no agent's tools can reach outside its own
workspace (file tools and the run_python sandbox both refuse it). Separate OS users or
containers per agent would be the next step; this module doesn't depend on that change.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import re
import secrets
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

PRINCIPALS_FILE = "principals.toml"
DEFAULT_DIR = Path("secrets")
DEFAULT_PRINCIPALS = ("user", "orchestrator")
DEFAULT_SIGNERS = ("verifier",)
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

# How old a signed result may be when it is used (seconds).
ATTESTATION_MAX_AGE = 3600.0


class AuthError(Exception):
    """The auth folder is missing, incomplete or invalid."""


class AttestationError(Exception):
    """A signed result is malformed, forged, altered, from the wrong signer or too old."""


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _check_name(name: str) -> str:
    if not NAME.match(name):
        raise AuthError(f"bad principal name {name!r}: use a-z, 0-9, _ and -, up to 32")
    return name


# --- creating the folder ---------------------------------------------------------------


def init_auth(
    root: Path,
    principals: tuple[str, ...] | list[str] = DEFAULT_PRINCIPALS,
    signers: tuple[str, ...] | list[str] = DEFAULT_SIGNERS,
    *,
    force: bool = False,
) -> list[Path]:
    """Create a token per principal and a key pair per signer. Refuses to overwrite an
    existing folder unless ``force`` (which issues new tokens and keys: every running agent
    and client must then be restarted). Returns the files written."""
    for n in [*principals, *signers]:
        _check_name(n)
    if (root / PRINCIPALS_FILE).exists() and not force:
        raise AuthError(f"{root / PRINCIPALS_FILE} exists; pass --force to issue new secrets")
    (root / "tokens").mkdir(parents=True, exist_ok=True)
    (root / "keys").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    lines = ["# Principal -> sha256 of its bearer token. Written by `agentmesh auth init`.", ""]
    for name in dict.fromkeys(principals):
        token = secrets.token_urlsafe(32)
        path = root / "tokens" / f"{name}.token"
        _write_private(path, token + "\n")
        written.append(path)
        lines.append(f'{name} = "{_hash(token)}"')
    _write_private(root / PRINCIPALS_FILE, "\n".join(lines) + "\n")
    written.append(root / PRINCIPALS_FILE)
    for name in dict.fromkeys(signers):
        key = Ed25519PrivateKey.generate()
        priv = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        pub = key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        _write_private(root / "keys" / f"{name}.key", priv.decode())
        (root / "keys" / f"{name}.pub").write_text(pub.decode(), encoding="utf-8", newline="\n")
        written += [root / "keys" / f"{name}.key", root / "keys" / f"{name}.pub"]
    return written


def _write_private(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    with contextlib.suppress(OSError):  # owner-only on POSIX; Windows keeps the folder's ACL
        path.chmod(0o600)


# --- reading it ------------------------------------------------------------------------


@dataclass
class AuthConfig:
    """What an agent server needs: the token hashes, and the folder for keys."""

    root: Path
    token_hashes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> AuthConfig:
        path = root / PRINCIPALS_FILE
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AuthError(f"no {path}; run `agentmesh auth init` first") from exc
        except tomllib.TOMLDecodeError as exc:
            raise AuthError(f"{path}: invalid TOML: {exc}") from exc
        hashes = {}
        for name, digest in data.items():
            _check_name(name)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise AuthError(f"{path}: {name} must be a sha256 hex digest")
            hashes[name] = digest
        if not hashes:
            raise AuthError(f"{path} lists no principals")
        return cls(root, hashes)

    def principal_for(self, token: str) -> str | None:
        """The principal a token belongs to, compared in constant time."""
        digest = _hash(token)
        found = None
        for name, known in self.token_hashes.items():
            if hmac.compare_digest(digest, known):
                found = name
        return found

    def signing_key(self, name: str) -> Ed25519PrivateKey:
        path = self.root / "keys" / f"{_check_name(name)}.key"
        try:
            key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        except FileNotFoundError as exc:
            raise AuthError(f"no signing key {path}") from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise AuthError(f"{path} is not an Ed25519 key")
        return key

    def public_key(self, name: str) -> Ed25519PublicKey:
        path = self.root / "keys" / f"{_check_name(name)}.pub"
        try:
            key = serialization.load_pem_public_key(path.read_bytes())
        except FileNotFoundError as exc:
            raise AuthError(f"no public key {path}") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise AuthError(f"{path} is not an Ed25519 key")
        return key


def load_token(root: Path, name: str) -> str:
    path = root / "tokens" / f"{_check_name(name)}.token"
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise AuthError(f"no token for {name!r} at {path}; run `agentmesh auth init`") from exc
    if not token:
        raise AuthError(f"{path} is empty")
    return token


def auth_headers(root: Path, name: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {load_token(root, name)}"}


# --- the HTTP gate ---------------------------------------------------------------------


class RequireLogin:
    """ASGI middleware: every request except ``public`` paths (GET) needs a known bearer
    token. The principal goes into ``scope["user"]``, where the A2A SDK picks it up as the
    caller (it also scopes the task store, so a principal sees only its own tasks)."""

    def __init__(self, app: Any, auth: AuthConfig, public: set[str]) -> None:
        self.app = app
        self.auth = auth
        self.public = public

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("method") in ("GET", "HEAD") and scope.get("path") in self.public:
            await self.app(scope, receive, send)
            return
        header = ""
        for key, value in scope.get("headers") or []:
            if key == b"authorization":
                header = value.decode("latin-1")
        scheme, _, token = header.partition(" ")
        who = self.auth.principal_for(token.strip()) if scheme.lower() == "bearer" else None
        if who is None:
            await _unauthorized(send, "missing token" if not header else "unknown token")
            return
        from starlette.authentication import AuthCredentials, SimpleUser

        scope["user"] = SimpleUser(who)
        scope["auth"] = AuthCredentials(["authenticated"])
        await self.app(scope, receive, send)


async def _unauthorized(send: Any, why: str) -> None:
    body = json.dumps({"error": f"unauthorized: {why}"}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="agentmesh"'),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


# --- attestations ----------------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_attestation(
    key: Ed25519PrivateKey,
    issuer: str,
    *,
    agent: str,
    task: str,
    caller: str,
    input: str,
    answer: str,
    now: float | None = None,
) -> str:
    """``<base64url payload>.<base64url signature>``: one string, so it survives protobuf
    ``Struct`` metadata unchanged (a Struct would turn the numbers into floats)."""
    payload = {
        "v": 1,
        "iss": issuer,
        "agent": agent,
        "task": task,
        "caller": caller,
        "iat": int(now if now is not None else time.time()),
        "input": input,
        "answer": answer,
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return f"{_b64(body)}.{_b64(key.sign(body))}"


def verify_attestation(
    token: str,
    key: Ed25519PublicKey,
    issuer: str,
    *,
    max_age: float = ATTESTATION_MAX_AGE,
    now: float | None = None,
) -> dict[str, Any]:
    """The signed payload, or AttestationError saying what is wrong with it."""
    try:
        body_b64, sig_b64 = token.split(".")
        body, sig = _unb64(body_b64), _unb64(sig_b64)
    except ValueError as exc:
        raise AttestationError("not an attestation") from exc
    try:
        key.verify(sig, body)
    except InvalidSignature as exc:
        raise AttestationError(f"the signature is not {issuer}'s, or the text was changed") from exc
    payload = json.loads(body)
    if payload.get("iss") != issuer:
        raise AttestationError(f"signed as {payload.get('iss')!r}, not {issuer!r}")
    age = (now if now is not None else time.time()) - float(payload.get("iat", 0))
    if age > max_age:
        raise AttestationError(f"signed {age:.0f} s ago; the limit is {max_age:g} s")
    if age < -60:
        raise AttestationError("signed in the future")
    return payload


# --- what a Verifier verdict supports ---------------------------------------------------

_URL = re.compile(r"<(https?://[^>\s\"']+)>|(https?://[^\s\]\)>\"'<]+)")
_SUPPORTED = re.compile(r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*){1,2}\**\s*SUPPORTED\b")


def norm_url(url: str) -> str:
    """Compare URLs without a fragment, trailing punctuation, trailing slash or case of
    the scheme and host."""
    url = url.strip().rstrip(".,;:")
    url = url.split("#", 1)[0]
    m = re.match(r"^(https?)://([^/]+)(.*)$", url, re.IGNORECASE)
    if m:
        url = f"{m.group(1).lower()}://{m.group(2).lower()}{m.group(3)}"
    return url.rstrip("/")


def supported_sources(answer: str) -> Counter[str]:
    """URL -> how many claims the Verifier marked SUPPORTED with it as the source.

    The Verifier's format is one line per claim: ``<n>. SUPPORTED - "<quote>" [<url>]``.
    Only those lines count; CONTRADICTED, NOT FOUND and UNVERIFIABLE give nothing.
    """
    found: Counter[str] = Counter()
    for line in answer.splitlines():
        if _SUPPORTED.match(line):
            for url in dict.fromkeys(norm_url(g1 or g2) for g1, g2 in _URL.findall(line)):
                found[url] += 1
    return found


_NUMBERED = re.compile(r"^\s*(?:[-*]\s*)?(\d+)[.)]\s*(.*)$")


def supported_claims(input: str, answer: str) -> list[tuple[str, str]]:
    """``(url, evidence)`` for every claim the Verifier marked SUPPORTED. The evidence is
    the claim as the Verifier was asked it (the same-numbered line of its input, when the
    input numbers its claims) plus the quote it found, so a write can be checked against
    what was actually verified, not only against the URL."""
    asked: dict[str, str] = {}
    for line in input.splitlines():
        if m := _NUMBERED.match(line):
            asked.setdefault(m[1], _URL.sub(" ", m[2]))
    out = []
    for line in answer.splitlines():
        if not _SUPPORTED.match(line):
            continue
        n = _NUMBERED.match(line)[1]  # type: ignore[index]  # _SUPPORTED implies it
        evidence = f"{asked.get(n, '')} {_URL.sub(' ', line)}"
        for url in dict.fromkeys(norm_url(g1 or g2) for g1, g2 in _URL.findall(line)):
            out.append((url, evidence))
    return out
