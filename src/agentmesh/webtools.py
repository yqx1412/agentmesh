"""webtools: an MCP server with ``web_search`` and ``fetch_url`` for the Research and
Verifier agents.

Run it with ``python -m agentmesh.webtools`` (stdio transport); an agent's
``agentos.toml`` lists it as an MCP server and picks which of the two tools it gets.

``fetch_url`` only reaches public addresses. Every hop of a redirect is resolved and
refused if any address is loopback, private, link-local or otherwise not global, so a
page can't steer an agent at Neo4j, Ollama or this machine's other local services. The
check runs before the connection, so a DNS answer that changes between the check and
the connect (rebinding) is not covered.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

USER_AGENT = "agentmesh-research/0.2 (+https://github.com/yqx1412/agentmesh)"
MAX_BYTES = 3_000_000
MAX_REDIRECTS = 5
DEFAULT_CHARS = 8_000
MAX_CHARS = 20_000
TEXT_TYPES = ("text/plain", "text/markdown", "text/csv", "application/json", "text/x-")

Resolver = Callable[[str, int], list[str]]


def _resolve(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"cannot resolve {host!r}: {exc}") from exc
    return sorted({str(info[4][0]) for info in infos})


def _is_public(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_url(url: str, resolve: Resolver = _resolve) -> str:
    """Return the URL if it is http(s) and every address its host resolves to is public."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolError(f"only http and https URLs can be fetched, not {parts.scheme or 'none'!r}")
    host = parts.hostname
    if not host:
        raise ToolError(f"URL has no host: {url!r}")
    if parts.username or parts.password:
        raise ToolError("URLs with credentials are refused")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    addrs = resolve(host, port)
    if not addrs:
        raise ToolError(f"cannot resolve {host!r}")
    blocked = [a for a in addrs if not _is_public(a)]
    if blocked:
        raise ToolError(f"refused: {host} resolves to a non-public address ({blocked[0]})")
    return url


@dataclass
class Page:
    url: str
    title: str | None
    text: str


def _to_text(body: bytes, content_type: str, url: str) -> tuple[str | None, str]:
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    raw = body.decode(charset, errors="replace")
    if content_type.startswith(TEXT_TYPES):
        return None, raw
    if "html" not in content_type and "xml" not in content_type:
        raise ToolError(f"unsupported content type {content_type.split(';')[0]!r}")
    import trafilatura

    text = trafilatura.extract(raw, url=url, include_tables=True, include_comments=False)
    meta = trafilatura.extract_metadata(raw, default_url=url)
    title = meta.title if meta is not None else None
    if not text:
        # Pages with little prose (indexes, listings): fall back to all visible text.
        from lxml import html as lxml_html

        tree = lxml_html.fromstring(raw)
        for bad in tree.xpath("//script|//style|//noscript"):
            bad.drop_tree()
        text = " ".join(tree.text_content().split())
    return title, text


def fetch_page(
    url: str,
    *,
    resolve: Resolver = _resolve,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 20.0,
) -> Page:
    """GET a page, following up to 5 redirects and checking every hop, and extract its text."""
    current = url
    with httpx.Client(
        transport=transport,
        timeout=timeout,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            check_url(current, resolve)
            try:
                with client.stream("GET", current) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise ToolError(f"redirect without a Location from {current}")
                        current = urljoin(current, location)
                        continue
                    if resp.status_code >= 400:
                        raise ToolError(f"HTTP {resp.status_code} from {current}")
                    body = bytearray()
                    for chunk in resp.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_BYTES:
                            raise ToolError(f"page is larger than {MAX_BYTES // 1_000_000} MB")
                    ctype = resp.headers.get("content-type", "text/html").lower()
            except httpx.HTTPError as exc:
                raise ToolError(f"fetch failed: {type(exc).__name__}: {exc}") from exc
            title, text = _to_text(bytes(body), ctype, current)
            return Page(url=current, title=title, text=text.strip())
    raise ToolError(f"more than {MAX_REDIRECTS} redirects")


def page_window(page: Page, offset: int, max_chars: int) -> dict:
    if offset < 0:
        raise ToolError("offset must be >= 0")
    max_chars = max(500, min(max_chars, MAX_CHARS))
    piece = page.text[offset : offset + max_chars]
    end = offset + len(piece)
    return {
        "url": page.url,
        "title": page.title,
        "text": piece,
        "offset": offset,
        "total_chars": len(page.text),
        "next_offset": end if end < len(page.text) else None,
    }


def search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    from ddgs import DDGS

    if not query.strip():
        raise ToolError("query is empty")
    max_results = max(1, min(max_results, 10))
    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # ddgs raises its own exception types per backend
        raise ToolError(f"search failed: {type(exc).__name__}: {exc}") from exc
    return [
        {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
        for h in hits or []
    ]


server = MCPServer(
    "webtools", instructions="Web search (DuckDuckGo) and text extraction from public pages."
)


@server.tool()
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web. Returns up to max_results (at most 10) hits with title, url and a
    short snippet. Snippets are not sources: fetch a page before relying on it."""
    return {"query": query, "results": search(query, max_results)}


@server.tool()
def fetch_url(url: str, offset: int = 0, max_chars: int = DEFAULT_CHARS) -> dict:
    """Fetch a public web page and return its main text, max_chars at a time (default
    8000, at most 20000). If next_offset is not null, call again with that offset to read
    further. Only http(s) URLs on public addresses can be fetched."""
    return page_window(fetch_page(url), offset, max_chars)


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
