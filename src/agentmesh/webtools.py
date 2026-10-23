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
import re
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
MIN_RESULTS = 5
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


def _mark_paragraphs(raw: str) -> str:
    """Turn elements that render as paragraphs (``data-as="p"``) into real ``<p>`` tags.

    Some docs sites (Mintlify, e.g. modelcontextprotocol.io) emit paragraphs as
    ``<span data-as="p">``. Trafilatura drops the loose spans that come before the first
    heading, which is where a page's lead paragraph sits -- the MCP intro lost its
    "USB-C port" sentence that way.
    """
    if 'data-as="p"' not in raw and "data-as='p'" not in raw:
        return raw
    from lxml import html as lxml_html

    try:
        tree = lxml_html.fromstring(raw)
    except (ValueError, lxml_html.etree.ParserError):
        return raw
    for el in tree.xpath("//*[@data-as='p']"):
        el.tag = "p"
    return lxml_html.tostring(tree, encoding="unicode")


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

    raw = _mark_paragraphs(raw)
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
    # At least 5 returned: Research usually asks for 3, and over its saved "Qwen3-8B
    # official model card or config.json" searches the Hugging Face card was in 9 of 39
    # three-hit lists but in 19 of 29 with five or more. Ten are always asked for, so a
    # model page ranked 6-10 still reaches the top (see below).
    max_results = max(MIN_RESULTS, min(max_results, 10))
    try:
        hits = DDGS().text(query, max_results=10)
    except Exception as exc:  # ddgs raises its own exception types per backend
        raise ToolError(f"search failed: {type(exc).__name__}: {exc}") from exc
    results = [
        {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
        for h in hits or []
        if h.get("href")
    ]
    # A model's own Hugging Face page first: Research fetches the first hit most of the
    # time (174 of 229 saved Research runs), and when that was a PDF, a spec database or a
    # guessed GitHub path it often never reached the model card further down the list.
    return sorted(results, key=lambda r: not _primary_model_page(r["url"]))[:max_results]


def _primary_model_page(url: str) -> bool:
    return bool(_HF_MODEL.match(url) or _HF_CONFIG.match(url))


server = MCPServer(
    "webtools", instructions="Web search (DuckDuckGo) and text extraction from public pages."
)


@server.tool()
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web. Returns between 5 and 10 hits (max_results, raised to 5 if lower)
    with title, url and a short snippet. Snippets are not sources: fetch a page before
    relying on it."""
    results = search(query, max_results)
    if not results:
        # Live, an empty hit ({"title": "", "url": ""}) came back as a success and Research
        # answered the question from another search's pages instead of searching again.
        raise ToolError(f"no results for {query!r}; search again with fewer or other words")
    return {"query": query, "results": results}


@server.tool()
def fetch_url(url: str, offset: int = 0, max_chars: int = DEFAULT_CHARS) -> dict:
    """Fetch a public web page and return its main text, max_chars at a time (default
    8000, at most 20000). If next_offset is not null, call again with that offset to read
    further. Only http(s) URLs on public addresses can be fetched."""
    return page_window(with_model_config(fetch_page(url), asked=url), offset, max_chars)


# A Hugging Face model page: https://huggingface.co/<org>/<model>, nothing after it.
_HF_MODEL = re.compile(r"^https://huggingface\.co/([\w.-]+)/([\w.-]+)/?$")
# The same model's config.json, at any revision (blob, raw or resolve).
_HF_CONFIG = re.compile(
    r"^https://huggingface\.co/([\w.-]+)/([\w.-]+)/(?:blob|raw|resolve)/([^/]+)/config\.json$"
)
CONFIG_NOTE = (
    "(max_position_embeddings is the longest position the code allows. If it differs from "
    "the context length the model card states, the card's is the model's native one.)"
)


def with_model_config(
    page: Page, fetch: Callable[[str], Page] | None = None, *, asked: str | None = None
) -> Page:
    """A Hugging Face model card together with the repository's config.json, config first.

    Cards often leave out architecture values: Qwen3-8B's gives layers and heads but no
    head dimension, which is in config.json. Live, Research then reported 5,120 or an
    assumed 128, and when it looked for config.json it tried github.com paths that don't
    exist. The other way round, a Research step that fetched only config.json took its
    max_position_embeddings (40,960) as the native context length the card gives (32,768),
    so a fetched config.json gets the card after it. Fetching the other half can fail;
    the page is then returned as it was."""
    get = fetch or fetch_page
    url = asked or page.url
    if m := _HF_MODEL.match(url):
        card = page
        try:
            config = get(f"https://huggingface.co/{m[1]}/{m[2]}/resolve/main/config.json")
        except ToolError:
            return page
    elif m := _HF_CONFIG.match(url):
        try:
            # A blob URL is the rendered file view; the raw file is what the card pairs with.
            config = page
            if not page.text.lstrip().startswith("{"):
                config = get(f"https://huggingface.co/{m[1]}/{m[2]}/resolve/{m[3]}/config.json")
            card = get(f"https://huggingface.co/{m[1]}/{m[2]}")
        except ToolError:
            return page
    else:
        return page
    if not config.text.lstrip().startswith("{") or card.text.lstrip().startswith("{"):
        return page
    shown = f"https://huggingface.co/{m[1]}/{m[2]}"
    meta = card_metadata(m[1], m[2], get)
    text = (
        f"[This model's config.json, from {shown}/blob/main/config.json]\n{CONFIG_NOTE}\n"
        f"{config.text.strip()}\n\n[The model card, from {shown}]\n"
        + (f"{meta}\n" if meta else "")
        + card.text.strip()
    )
    return Page(url=page.url, title=page.title, text=text)


def card_metadata(org: str, model: str, fetch: Callable[[str], Page]) -> str | None:
    """The card's header fields (license, base model, ...) as one line, or None.

    The rendered card shows them only in its page header ("License: apache-2.0"), which
    trafilatura drops, so the fetched text had neither "license" nor "apache" and Research
    said the license wasn't on the page. They come from the YAML front matter of the
    repository's README.md. Only short values are kept, and none of the gated-access form
    (extra_gated_*, which on a gated model includes the whole licence agreement);
    model-index holds benchmark tables and is left out too."""
    import yaml

    try:
        raw = fetch(f"https://huggingface.co/{org}/{model}/resolve/main/README.md").text
    except ToolError:
        return None
    m = re.match(r"\s*---\r?\n(.*?)\r?\n---\s*(?:\n|$)", raw, re.S)
    if not m:
        return None
    try:
        data = yaml.safe_load(m[1])
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    fields = []
    for key, value in data.items():
        if str(key).startswith("extra_gated"):
            continue  # the gated-access form's text, not facts about the model
        if (
            isinstance(value, list)
            and len(value) <= 10
            and all(isinstance(v, str | int | float) for v in value)
        ):
            value = ", ".join(str(v) for v in value)
        if isinstance(value, str | int | float) and not isinstance(value, bool):
            value = " ".join(str(value).split())
            if value and len(value) <= 200:
                fields.append(f"{key}: {value}")
    return f"Card metadata: {'; '.join(fields)}" if fields else None


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
