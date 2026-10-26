"""Finding and comparing the URLs an answer cites.

Shared by the eval (which scores citations against the pages fetched) and Research's
answer guard (which sends back an answer citing a page it never fetched).
"""

from __future__ import annotations

import re

_URL = re.compile(r"https?://[^\s\]\[)(<>\"'`,|]+")
# A fetched Hugging Face card carries its config.json and the other way round, each under
# a "[..., from <url>]" heading (webtools.with_model_config): both parts were fetched.
PART_FROM = re.compile(r"^\[[^\]\n]*, from (https?://[^\s\]]+)\]$", re.M)


def normalize_url(url: str) -> str:
    url = url.strip().rstrip(".;:!?*_")
    url = url.split("#", 1)[0]
    scheme, _, rest = url.partition("://")
    host, slash, path = rest.partition("/")
    path = path.rstrip("/")
    return f"{scheme.lower()}://{host.lower()}{slash if path else ''}{path}"


def urls_in(text: str) -> list[str]:
    return list(dict.fromkeys(normalize_url(u) for u in _URL.findall(text)))


def strip_citations(answer: str, bad: list[str]) -> str:
    """Remove each unfetched URL in ``bad`` from ``answer``, taking any bracket, paren or
    separator that only wrapped it so no empty "[]" or dangling "[, ]" is left behind.

    Used only after the answer guard's nudge budget is spent (see ``AnswerGuardLLM.chat``):
    the model kept citing a page it never fetched, so rather than ship that citation the
    guard drops it. Matching is on the raw URL as it appears, normalized for comparison, so
    a citation written with a trailing "/" or "#frag" still goes. A URL that is a prefix of
    a longer fetched one is never in ``bad`` (``unfetched_citations`` compares normalized),
    so this cannot truncate a good citation."""
    targets = {normalize_url(u) for u in bad}
    if not targets:
        return answer

    def drop(m: re.Match[str]) -> str:
        return "" if normalize_url(m.group(0)) in targets else m.group(0)

    out = _URL.sub(drop, answer)
    # Tidy the punctuation the citation left behind: an "[]"/"()" the URL vacated, a "[, ]"
    # from a two-URL list, a dangling separator against a bracket, then doubled spaces.
    out = re.sub(r"[\[(]\s*[,;]?\s*[\])]", "", out)
    out = re.sub(r"([\[(])\s*[,;]\s*", r"\1", out)
    out = re.sub(r"\s*[,;]\s*([\])])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()
