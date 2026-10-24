"""Finding and comparing the URLs an answer cites.

Shared by the eval (which scores citations against the pages fetched) and Research's
answer guard (which sends back an answer citing a page it never fetched).
"""

from __future__ import annotations

import re

_URL = re.compile(r"https?://[^\s\]\[)(<>\"'`,|]+")


def normalize_url(url: str) -> str:
    url = url.strip().rstrip(".;:!?*_")
    url = url.split("#", 1)[0]
    scheme, _, rest = url.partition("://")
    host, slash, path = rest.partition("/")
    path = path.rstrip("/")
    return f"{scheme.lower()}://{host.lower()}{slash if path else ''}{path}"


def urls_in(text: str) -> list[str]:
    return list(dict.fromkeys(normalize_url(u) for u in _URL.findall(text)))
