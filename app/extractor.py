"""Page retrieval + boilerplate stripping: httpx + Trafilatura, Jina Reader fallback, snippet last resort."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections import defaultdict
from typing import Optional

import httpx
import trafilatura

from app.config import settings
from app.schemas import RetrievedDocument, SearchResult

MIN_CHARS = 300
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 AgenticResearch/1.0"
)
INJECTION_PATTERNS = re.compile(
    r"(ignore (all |any )?(previous|prior|above) (instructions|prompts)|disregard (the )?(system|previous) prompt|"
    r"you are now (a|an) |as an ai language model,? you must|<\s*/?\s*(system|assistant)\s*>)",
    re.I,
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_injection_markup(text: str) -> str:
    """Remove HTML comments and neutralise tags that could break our XML evidence fences."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return text.replace("<untrusted_web_evidence", "&lt;untrusted_web_evidence").replace(
        "</untrusted_web_evidence", "&lt;/untrusted_web_evidence"
    )


def regex_injection_flag(text: str) -> bool:
    return bool(INJECTION_PATTERNS.search(text))


class CircuitBreaker:
    """Trip a domain after 3 consecutive failures for the remainder of the run."""

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self.failures: dict[str, int] = defaultdict(int)

    @staticmethod
    def domain(url: str) -> str:
        m = re.match(r"https?://([^/:]+)", url)
        return m.group(1).lower() if m else url

    def open(self, url: str) -> bool:
        return self.failures[self.domain(url)] >= self.threshold

    def record(self, url: str, ok: bool) -> None:
        d = self.domain(url)
        self.failures[d] = 0 if ok else self.failures[d] + 1


class Extractor:
    def __init__(self) -> None:
        self.breaker = CircuitBreaker()
        self.sem = asyncio.Semaphore(settings.FETCH_CONCURRENCY)
        self.client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"},
            timeout=settings.HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _direct(self, url: str) -> tuple[Optional[str], Optional[str], int]:
        for attempt in range(2):
            try:
                r = await self.client.get(url)
            except (httpx.TimeoutException, httpx.TransportError):
                return None, None, 0
            if r.status_code >= 500 and attempt == 0:
                await asyncio.sleep(2)
                continue
            if r.status_code != 200 or "html" not in r.headers.get("content-type", "html"):
                return None, None, r.status_code
            html = r.text
            text, date = await asyncio.to_thread(self._trafilatura, html, url)
            return text, date, 200
        return None, None, 0

    @staticmethod
    def _trafilatura(html: str, url: str) -> tuple[Optional[str], Optional[str]]:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True, favor_precision=True)
        date = None
        try:
            meta = trafilatura.extract_metadata(html)
            date = meta.date if meta else None
        except Exception:
            pass
        return text, date

    async def _jina(self, url: str) -> Optional[str]:
        """Free keyless reader (renders JS pages). Rate-limited, so only used as a fallback."""
        try:
            r = await self.client.get(
                "https://r.jina.ai/" + url, headers={"X-Return-Format": "text"}, timeout=settings.HTTP_TIMEOUT_SECONDS * 2
            )
            if r.status_code == 200:
                return r.text
        except (httpx.TimeoutException, httpx.TransportError):
            pass
        return None

    async def fetch(self, hit: SearchResult) -> Optional[RetrievedDocument]:
        """Tiered retrieval. Returns None only when not even a usable snippet exists."""
        url = hit.url
        async with self.sem:
            t0 = time.perf_counter()
            text, date, status, method = None, None, 0, "trafilatura"
            if not self.breaker.open(url):
                text, date, status = await self._direct(url)
                if not text or len(text) < MIN_CHARS:
                    if settings.USE_JINA_READER:
                        jt = await self._jina(url)
                        if jt and len(jt) >= MIN_CHARS:
                            text, method = jt, "jina"
                self.breaker.record(url, bool(text and len(text) >= MIN_CHARS))
            if not text or len(text) < MIN_CHARS:
                if len(hit.snippet) < 40:
                    return None  # dead letter: never passed to the LLM
                text, method = hit.snippet, "snippet"
            text = strip_injection_markup(text)
            return RetrievedDocument(
                url=url,
                title=hit.title,
                extracted_text=text[:60000],
                content_hash=sha256(text),
                http_status=status or 200,
                method=method,
                published_date=date or hit.published_date,
                retrieval_latency_ms=(time.perf_counter() - t0) * 1000,
            )
