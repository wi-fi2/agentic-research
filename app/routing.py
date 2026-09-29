"""Source routing: send each sub-question to the free sources best suited to it, and rank the
evidence by source authority and recency as well as relevance.

Sub-question types come from Laya yes/no questions on the short search query, chosen on a
25-item labelled set (see PLAN.md §10), plus a small
keyword backstop in code. A sub-question may have several types or none; web search always
runs, so routing only adds sources.

Specialist sources (all free, no key): OpenAlex (papers + abstracts), Wikipedia (background,
added once per run for the main question), DuckDuckGo news (dated news for market questions),
and extra "official"/"documentation" web queries for policy and technical questions.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urlparse

import httpx

from app.config import settings
from app.schemas import SearchResult

log = logging.getLogger(__name__)

TYPE_QUESTIONS = {
    "academic": "Would published scientific studies or research papers answer this query?",
    "policy": "Is this query about what a law, regulation, standard or government rule requires?",
    "market": "Is this query about recent business news, company results, sales, prices or markets?",
    "technical": "Is this query about how a software tool, database or technology works or is configured?",
}
# Classify the short search query only (as evaluated): adding the LLM's facet description pushed
# the academic/technical nouls up on unrelated facets (e.g. "Benchmark results ..." -> academic 0.78).
TYPE_THRESHOLDS = {"academic": 0.4, "policy": 0.4, "market": 0.4, "technical": 0.4}
# Code backstop for obvious wording Laya missed on the labelled set (chosen after seeing the misses).
TYPE_KEYWORDS = {
    "academic": r"\b(stud(y|ies)|trials?|meta-analys[ie]s|papers?|research|evidence|experiments?|peer.reviewed)\b",
    "policy": r"\b(law|act|regulat\w*|directive|compliance|legal|statute|gdpr|standard)\b",
    "market": r"\b(revenue|sales|market share|capacity|funding|prices?|shipments?|earnings|manufacturers?|news)\b",
    "technical": r"\b(api|sdk|database|configur\w*|documentation|library|framework|kubernetes|python)\b",
}
USER_AGENT = "AgenticResearch/1.0 (open-source research assistant)"


async def classify(sub_questions: dict[str, str], judge) -> dict[str, list[str]]:
    """{sq_id: [types]} from Laya nouls (if available) unioned with the keyword backstop."""
    out: dict[str, set[str]] = {sid: set() for sid in sub_questions}
    if judge is not None and sub_questions:
        try:
            questions = {t: {"type": "noul", "instructions": q} for t, q in TYPE_QUESTIONS.items()}
            ids = list(sub_questions)
            results = await judge._predict_cached([sub_questions[i] for i in ids], questions)
            for sid, r in zip(ids, results):
                for t, thr in TYPE_THRESHOLDS.items():
                    if float(r["answers"][t]["noul"]) >= thr:
                        out[sid].add(t)
        except Exception as e:
            log.warning("Laya routing failed, keywords only: %s", e)
    for sid, text in sub_questions.items():
        for t, pat in TYPE_KEYWORDS.items():
            if re.search(pat, text, re.I):
                out[sid].add(t)
    return {sid: sorted(ts) for sid, ts in out.items()}


# ---------------------------------------------------------------- specialist sources

def _openalex_abstract(inv: Optional[dict]) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


async def openalex(query: str, sq_id: str, n: int = 4) -> list[SearchResult]:
    """Papers with abstracts. The abstract becomes the document text (no fetch: OA links are often PDFs)."""
    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as c:
        r = await c.get("https://api.openalex.org/works", params={
            "search": query, "per_page": n, "filter": "has_abstract:true",
            "select": "id,doi,title,publication_year,cited_by_count,open_access,primary_location,abstract_inverted_index",
        })
        r.raise_for_status()
    out = []
    for i, w in enumerate(r.json().get("results", [])):
        abstract = _openalex_abstract(w.get("abstract_inverted_index"))
        if len(abstract) < 200:
            continue
        url = (w.get("doi") or ((w.get("primary_location") or {}).get("landing_page_url")) or w["id"])
        year = w.get("publication_year")
        venue = (((w.get("primary_location") or {}).get("source")) or {}).get("display_name") if w.get("primary_location") else None
        header = f"{w.get('title', '')} ({year}; cited by {w.get('cited_by_count', 0)})"
        out.append(SearchResult(
            sub_query_id=sq_id, url=url, title=(w.get("title") or url)[:300], snippet=abstract[:1000], rank=i,
            published_date=str(year) if year else None, provider="openalex", kind="paper",
            prefetched_text=f"{header}\n{venue or ''}\nAbstract: {abstract}",
        ))
    return out


async def wikipedia(query: str, sq_id: str, n: int = 1) -> list[SearchResult]:
    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}) as c:
        r = await c.get("https://en.wikipedia.org/w/rest.php/v1/search/page", params={"q": query, "limit": n})
        r.raise_for_status()
    return [
        SearchResult(sub_query_id=sq_id, url=f"https://en.wikipedia.org/wiki/{quote(p['key'])}", title=p.get("title", p["key"]),
                     snippet=re.sub("<[^>]+>", "", p.get("excerpt") or ""), rank=i, provider="wikipedia", kind="encyclopedia")
        for i, p in enumerate(r.json().get("pages", []))
    ]


async def news(query: str, sq_id: str, n: int = 5) -> list[SearchResult]:
    from ddgs import DDGS

    def run() -> list[dict]:
        try:
            return DDGS().news(query, max_results=n, backend="bing")
        except Exception:
            return []

    hits = await asyncio.wait_for(asyncio.to_thread(run), timeout=settings.HTTP_TIMEOUT_SECONDS)
    return [
        SearchResult(sub_query_id=sq_id, url=h["url"], title=h.get("title", h["url"])[:300], snippet=h.get("body", "")[:1000],
                     rank=i, published_date=(h.get("date") or "")[:10] or None, provider="news", kind="news")
        for i, h in enumerate(hits) if h.get("url", "").startswith("http")
    ]


def extra_web_queries(query: str, types: list[str]) -> list[str]:
    """Query variants that pull primary sources for policy / technical facets."""
    out = []
    if "policy" in types:
        out.append(f"{query} official text regulation")
    if "technical" in types:
        out.append(f"{query} official documentation")
    return out


# ---------------------------------------------------------------- authority + recency

HIGH_AUTHORITY = (
    "europa.eu", "who.int", "oecd.org", "un.org", "worldbank.org", "imf.org", "doi.org", "arxiv.org",
    "ncbi.nlm.nih.gov", "nih.gov", "nature.com", "science.org", "sciencedirect.com", "springer.com",
    "wiley.com", "acm.org", "ieee.org", "thelancet.com", "bmj.com", "nejm.org", "plos.org", "openalex.org",
    "iso.org", "w3.org", "ietf.org", "rfc-editor.org", "python.org", "readthedocs.io",
)
MEDIUM_HIGH = ("wikipedia.org", "github.com", "reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "nytimes.com",
               "theguardian.com", "economist.com", "stackoverflow.com")
LOW_AUTHORITY = ("pinterest.", "quora.com", "reddit.com", "youtube.com", "tiktok.com", "facebook.com", "linkedin.com/posts",
                 "medium.com", "scribd.com", "slideshare.net")


def authority(url: str, kind: str = "web") -> float:
    """Heuristic 0-1 source authority from the URL (primary / official / peer-reviewed rank highest)."""
    if kind == "paper":
        return 1.0
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")
    full = host + p.path.lower()
    if any(b in full for b in LOW_AUTHORITY):
        return 0.3
    if (re.search(r"\.(gov|mil|edu|int)(\.[a-z]{2})?$", host) or re.search(r"\.(gov|ac|edu)\.[a-z]{2}$", host)
            or any(host == d or host.endswith("." + d) for d in HIGH_AUTHORITY)):
        return 1.0
    if host.startswith(("docs.", "developer.", "developers.", "learn.")) or "/docs/" in p.path.lower() \
            or "/documentation/" in p.path.lower():
        return 0.9
    if any(host == d or host.endswith("." + d) for d in MEDIUM_HIGH):
        return 0.8
    return 0.5


def _year(date: Optional[str]) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", date or "")
    return int(m.group(0)) if m else None


def recency(date: Optional[str], types: list[str], now: Optional[datetime] = None) -> float:
    """Down-weight old evidence for fast-moving facets (market / policy / technical). Unknown date: neutral."""
    if not ({"market", "policy", "technical"} & set(types)):
        return 1.0
    y = _year(date)
    if y is None:
        return 0.95
    age = (now or datetime.now(timezone.utc)).year - y
    if "market" in types:
        return 1.0 if age <= 1 else 0.85 if age <= 3 else 0.65
    return 1.0 if age <= 3 else 0.9 if age <= 6 else 0.75


def source_score(relevance: float, url: str, kind: str, date: Optional[str], types: list[str]) -> float:
    """Ranking score for evidence selection: relevance first, then authority and recency."""
    return relevance * (0.75 + 0.25 * authority(url, kind)) * recency(date, types)
