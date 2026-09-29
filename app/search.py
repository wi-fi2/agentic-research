"""Search providers with a free-first fallback chain: SearXNG -> DuckDuckGo -> Tavily -> Brave."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Awaitable, Callable

import httpx

from app.config import settings
from app.schemas import SearchResult

log = logging.getLogger(__name__)

BLOCKED_DOMAINS = {"ft.com", "wsj.com", "bloomberg.com", "economist.com", "barrons.com", "pinterest.com"}

RawHit = dict  # {"url", "title", "snippet", "published_date"?}


def _domain(url: str) -> str:
    m = re.match(r"https?://(?:www\.)?([^/:]+)", url)
    return m.group(1).lower() if m else ""


def is_blocked(url: str) -> bool:
    d = _domain(url)
    return any(d == b or d.endswith("." + b) for b in BLOCKED_DOMAINS)


def sanitize_query(q: str) -> str:
    """Strip search operators / odd punctuation that commonly break SERP APIs."""
    q = re.sub(r"\b(site|inurl|intitle|filetype):\S+", " ", q)
    q = re.sub(r"[^\w\s\-.'\"/+#]", " ", q)
    return re.sub(r"\s+", " ", q).strip()[:200]


async def _searxng(q: str, n: int) -> list[RawHit]:
    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as c:
        r = await c.get(settings.SEARXNG_URL.rstrip("/") + "/search", params={"q": q, "format": "json"})
        r.raise_for_status()
        return [
            {"url": x["url"], "title": x.get("title", ""), "snippet": x.get("content", ""),
             "published_date": x.get("publishedDate")}
            for x in r.json().get("results", [])[:n]
        ]


async def _ddg_backend(q: str, n: int, backend: str) -> list[RawHit]:
    from ddgs import DDGS  # imported lazily; blocking client, run in a thread

    def run() -> list[RawHit]:
        try:
            hits = DDGS().text(q, max_results=n, backend=backend)
        except Exception:  # ddgs raises when a backend finds nothing or is blocked
            return []
        return [{"url": x.get("href", ""), "title": x.get("title", ""), "snippet": x.get("body", "")} for x in hits]

    try:
        return await asyncio.wait_for(asyncio.to_thread(run), timeout=settings.HTTP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return []


async def _ddg(q: str, n: int) -> list[RawHit]:
    """Race the configured ddgs backends and return the first non-empty result set.

    ddgs' default "auto" backend tries engines one after another, and several of them block
    or return nothing under load (measured 3.4 s/query with 2 of 3 queries empty, while the
    yahoo and bing backends returned full results in 1-2 s). "auto" is the last resort.
    """
    backends = [b.strip() for b in settings.DDG_BACKENDS.split(",") if b.strip()]
    pending = {asyncio.create_task(_ddg_backend(q, n, b)) for b in backends}
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                hits = task.result()
                if hits:
                    return hits
    finally:
        for task in pending:
            task.cancel()
    return await _ddg_backend(q, n, "auto")


async def _tavily(q: str, n: int) -> list[RawHit]:
    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS * 2) as c:
        r = await c.post(
            "https://api.tavily.com/search",
            json={"api_key": settings.TAVILY_API_KEY, "query": q, "max_results": n, "search_depth": "basic"},
        )
        r.raise_for_status()
        return [
            {"url": x["url"], "title": x.get("title", ""), "snippet": x.get("content", ""),
             "published_date": x.get("published_date")}
            for x in r.json().get("results", [])
        ]


async def _brave(q: str, n: int) -> list[RawHit]:
    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as c:
        r = await c.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": q, "count": n},
            headers={"X-Subscription-Token": settings.BRAVE_API_KEY, "Accept": "application/json"},
        )
        r.raise_for_status()
        return [
            {"url": x["url"], "title": x.get("title", ""), "snippet": re.sub("<[^>]+>", "", x.get("description", "")),
             "published_date": x.get("age")}
            for x in r.json().get("web", {}).get("results", [])
        ]


PROVIDERS: dict[str, tuple[Callable[[str, int], Awaitable[list[RawHit]]], Callable[[], bool]]] = {
    "searxng": (_searxng, lambda: bool(settings.SEARXNG_URL)),
    "ddg": (_ddg, lambda: True),
    "tavily": (_tavily, lambda: bool(settings.TAVILY_API_KEY)),
    "brave": (_brave, lambda: bool(settings.BRAVE_API_KEY)),
}


def enabled_providers() -> list[str]:
    return [n.strip() for n in settings.SEARCH_PROVIDER_ORDER.split(",") if n.strip() in PROVIDERS and PROVIDERS[n.strip()][1]()]


async def search(query: str, sub_query_id: str, n: int | None = None) -> tuple[list[SearchResult], str]:
    """Search with provider fallback, backoff, and an operator-stripping rewrite on empty results."""
    n = n or settings.RESULTS_PER_QUERY
    attempts = [query]
    clean = sanitize_query(query)
    if clean and clean != query:
        attempts.append(clean)
    errors = []
    for provider in enabled_providers():
        fn = PROVIDERS[provider][0]
        for q in attempts:
            for attempt in range(2):
                try:
                    hits = await fn(q, n + 3)
                    break
                except Exception as e:  # network, 429, parse
                    errors.append(f"{provider}: {type(e).__name__}: {str(e)[:80]}")
                    hits = []
                    await asyncio.sleep(1.5 * (2**attempt) * random.uniform(0.75, 1.25))
            hits = [h for h in hits if h.get("url", "").startswith("http") and not is_blocked(h["url"])]
            if hits:
                return [
                    SearchResult(
                        sub_query_id=sub_query_id,
                        url=h["url"],
                        title=(h.get("title") or h["url"])[:300],
                        snippet=(h.get("snippet") or "")[:1000],
                        rank=i,
                        published_date=h.get("published_date"),
                        provider=provider,
                    )
                    for i, h in enumerate(hits[:n])
                ], provider
    if errors:
        log.warning("search failed for %r: %s", query, errors)
    return [], "none"
