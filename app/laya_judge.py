"""Laya (open-source System 1 decision model) running locally: passage relevance and first-pass
claim verification. $0 per call; no data leaves the machine.

Laya is an encoder that returns typed probabilities instead of generated text. The model is
loaded once per process, warmed up, and guarded by a lock (one forward pass at a time).
Question phrasings and thresholds below were chosen on a small hand-labelled set (see
PLAN.md §2); re-measure on your own data before tightening them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from app import store
from app.config import settings
from app.evidence import bm25_scores

log = logging.getLogger(__name__)

# Measured: relevance noul separated relevant/irrelevant passages best at ~0.75-0.8 (92%).
RELEVANT_THRESHOLD = 0.75
# Injection screening is NOT done with Laya: on our probe set its injection/jailbreak nouls scored
# clean passages (0.81) and site navigation (1.0) as high as real injections. The regex screen in
# extractor.py plus XML isolation of web text in every LLM prompt stay the defence.
INJECTION_THRESHOLD = 0.5
# Verification cascade: below this confidence (or when the claim has numbers the source lacks)
# the claim is escalated to the LLM verifier. Measured ~90% accuracy on kept pairs, ~50% escalation.
VERIFY_MIN_CONFIDENCE = 0.2

STATE_CHARS = 1600  # English checkpoint: 512 tokens total, minus the question head
PASSAGE_BATCH = 16

_NUM = re.compile(r"\d+(?:[.,]\d+)?%?")

VERIFY_QUESTION = {
    "relation": {
        "type": "choice",
        "instructions": "Does `text` support, contradict, or not mention `statement`?",
        "criteria": {
            "supports": "the text states the statement or clearly implies it is true",
            "contradicts": "the text states something that makes the statement false",
            "says_nothing": "the text does not mention what the statement asserts",
        },
    }
}


@dataclass
class JudgeUsage:
    calls: int = 0
    input_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return 0.0  # runs locally


class _Model:
    """Process-wide singleton: load once in the background, share across runs."""

    agent: Any = None
    error: Optional[str] = None
    ready = threading.Event()
    lock = threading.Lock()
    _started = False

    @classmethod
    def start(cls) -> None:
        if cls._started or not settings.LAYA_ENABLED:
            return
        cls._started = True
        threading.Thread(target=cls._load, name="laya-loader", daemon=True).start()

    @classmethod
    def _load(cls) -> None:
        os.environ.setdefault("USE_TF", "0")
        t0 = time.perf_counter()
        try:
            import laya

            kwargs: dict[str, Any] = {}
            if settings.LAYA_SUBFOLDER:
                kwargs["subfolder"] = settings.LAYA_SUBFOLDER
            if settings.LAYA_DEVICE:
                kwargs["device"] = settings.LAYA_DEVICE
            agent = laya.load(settings.LAYA_MODEL, **kwargs)
            # Warm-up at a representative shape so the first real call is not slow.
            agent.predict_batch(["warm-up passage about databases"] * 2,
                                {"a": {"type": "noul", "instructions": "Is the text about databases?"}, **VERIFY_QUESTION})
            cls.agent = agent
            log.info("Laya loaded (%s) in %.1fs", agent, time.perf_counter() - t0)
        except Exception as e:  # missing package, download failure, OOM
            cls.error = f"{type(e).__name__}: {e}"
            log.warning("Laya unavailable, falling back to BM25 + LLM verification: %s", cls.error)
        finally:
            cls.ready.set()


def start_loading() -> None:
    _Model.start()


def status() -> str:
    if not settings.LAYA_ENABLED:
        return "disabled"
    if not _Model.ready.is_set():
        return "loading"
    return "ready" if _Model.agent is not None else "error"


async def get_judge(wait_seconds: float = 90.0) -> Optional["LayaJudge"]:
    """Return a judge if Laya is (or becomes, within `wait_seconds`) ready; None -> fallbacks."""
    if not settings.LAYA_ENABLED:
        return None
    _Model.start()
    if not _Model.ready.is_set():
        await asyncio.to_thread(_Model.ready.wait, wait_seconds)
    return LayaJudge() if _Model.agent is not None else None


def numbers_match(claim: str, source: str) -> bool:
    """Laya is weak on numeric precision, so figures are checked in code: every number in the
    claim must literally appear in the source, else the claim is escalated."""
    return all(n in source for n in set(_NUM.findall(claim)))


_QUOTE_TABLE = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "\u2010": "-", "\u2011": "-",
                               "\u2012": "-", "\u2212": "-", "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u00ad": ""})
_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…|\[\.\.\.\])\s*")
QUOTE_CONTEXT_CHARS = 700
FUZZY_QUOTE_MIN = 0.75
_TOK = re.compile(r"[a-z0-9]+(?:[.,'%-][a-z0-9]+)*%?")


def _fuzzy_find(q: str, s: str) -> tuple[int, int]:
    """Token-level fuzzy match: the window of the source whose word sequence best matches the
    quote (difflib ratio >= FUZZY_QUOTE_MIN). Tolerates small wording slips in a quote while
    still rejecting quotes the source does not contain. Returns (char_pos, char_len) or (-1, 0)."""
    from difflib import SequenceMatcher

    qt = _TOK.findall(q)
    toks = [(m.group(), m.start(), m.end()) for m in _TOK.finditer(s)]
    if len(qt) < 3 or not toks:
        return -1, 0
    words = [t[0] for t in toks]
    qset, best, best_i, best_n = set(qt), 0.0, -1, 0
    for n in {len(qt) - 1, len(qt), len(qt) + 2}:
        if n <= 0:
            continue
        for i in range(0, max(1, len(words) - n + 1)):
            if words[i] not in qset:  # cheap pre-filter: a window must start on a quote word
                continue
            r = SequenceMatcher(None, qt, words[i: i + n], autojunk=False).ratio()
            if r > best:
                best, best_i, best_n = r, i, n
    if best < FUZZY_QUOTE_MIN:
        return -1, 0
    j = min(len(toks), best_i + best_n) - 1
    return toks[best_i][1], toks[j][2] - toks[best_i][1]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTE_TABLE)).strip()


def _locate_single(q: str, s: str) -> tuple[int, int, str]:
    """Locate one normalised fragment in the normalised source: (pos, size, status)."""
    ql, sl = q.lower(), s.lower()
    pos, size, status = sl.find(ql), len(ql), "exact"
    if pos < 0:
        if len(ql.split()) < 5:  # short fragments must match exactly
            return -1, 0, "missing"
        pos, size = _fuzzy_find(ql, sl)
        if pos < 0 or not numbers_match(q, s[pos: pos + size]):  # wording may drift, figures may not
            return -1, 0, "missing"
        status = "fuzzy"
    return pos, size, status


def locate_quote(quote: str, source: str) -> tuple[Optional[str], str]:
    """Find a writer-supplied quote in the source. Returns (context, status).

    Quotes stitched with an ellipsis ("A ... B") are split and every fragment of 3+ words must be
    found. status is "exact" (verbatim after normalising whitespace, quotes, dashes and hyphens),
    "fuzzy" (token-level match for small wording slips, 5+ word fragments, figures must match),
    or "missing". context is a window around the matched text, sized for Laya's context.
    """
    s = _norm(source)
    frags = [f.strip(" .\"'") for f in _ELLIPSIS.split(_norm(quote))]
    frags = [f for f in frags if len(f.split()) >= 3]
    if not frags:
        return None, "missing"
    spans, statuses = [], []
    for f in frags:
        pos, size, st = _locate_single(f, s)
        if st == "missing":
            return None, "missing"
        spans.append((pos, pos + size))
        statuses.append(st)
    status = "exact" if all(x == "exact" for x in statuses) else "fuzzy"
    lo, hi = min(p for p, _ in spans), max(e for _, e in spans)
    if hi - lo > STATE_CHARS - 200:  # fragments far apart: join a window around each
        parts = [s[max(0, p - 120): e + 120] for p, e in spans]
        return " … ".join(parts)[:STATE_CHARS], status
    pad = max(0, (QUOTE_CONTEXT_CHARS - (hi - lo)) // 2)
    start, end = max(0, lo - pad), min(len(s), hi + pad)
    left = s.rfind(". ", 0, max(start, lo - 40))
    start = left + 2 if left != -1 and lo - left < QUOTE_CONTEXT_CHARS else start
    return s[start:end].strip(), status


def best_window(claim: str, text: str, size: int = STATE_CHARS) -> str:
    """The ~size-char window of `text` most lexically similar to the claim (fits Laya's context)."""
    if len(text) <= size:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    windows, cur = [], ""
    for i, s in enumerate(sentences):
        cur = s
        j = i + 1
        while j < len(sentences) and len(cur) + len(sentences[j]) + 1 <= size:
            cur += " " + sentences[j]
            j += 1
        windows.append(cur[:size])
    scores = bm25_scores(claim, windows)
    return windows[max(range(len(windows)), key=scores.__getitem__)] if windows else text[:size]


def _cache_key(state: Any, question: dict) -> str:
    """Laya is deterministic for a fixed checkpoint, so (checkpoint, question, state) keys a result."""
    raw = json.dumps([settings.LAYA_MODEL, settings.LAYA_SUBFOLDER, question, state], sort_keys=True, default=str)
    return "laya:" + hashlib.sha256(raw.encode()).hexdigest()


class LayaJudge:
    def __init__(self) -> None:
        self.usage = JudgeUsage()
        self.cache_hits = 0

    async def _predict_cached(self, states: list, questions: dict) -> list[dict]:
        """predict_batch with a SQLite cache in front: only uncached states reach the model."""
        keys = [_cache_key(s, questions) for s in states]
        cached = store.cache_get_many(keys)
        self.cache_hits += len(cached)
        missing = [i for i, k in enumerate(keys) if k not in cached]
        results: dict[int, dict] = {}
        for start in range(0, len(missing), PASSAGE_BATCH):
            idx = missing[start: start + PASSAGE_BATCH]
            for i, r in zip(idx, await self._predict_batch([states[i] for i in idx], questions)):
                answers = r["answers"]
                results[i] = answers
                store.cache_put(keys[i], answers)
        return [{"answers": cached[k]} if k in cached else {"answers": results[i]} for i, k in enumerate(keys)]

    async def _predict_batch(self, states: list, questions: dict) -> list[dict]:
        def run() -> list[dict]:
            with _Model.lock:
                return _Model.agent.predict_batch(states, questions)

        results = await asyncio.to_thread(run)
        self.usage.calls += 1
        for r in results:
            self.usage.input_tokens += int((r.get("usage") or {}).get("input_tokens") or 0)
        return results

    async def judge_relevance(self, shortlists: dict[str, list[tuple[str, str, str]]],
                              sub_questions: dict[str, str]) -> dict[str, dict[str, float]]:
        """P(relevant) for each sub-question's BM25 shortlist.

        `shortlists` maps sq_id -> [(passage_id, title, text)]. Each passage is only scored against
        the sub-questions it was shortlisted for: a full ~400-token passage costs ~330 ms per
        question on Apple MPS (~740 ms on CPU), so scoring every passage x every facet is wasteful.
        Returns {passage_id: {sq_id: p}}.
        """
        out: dict[str, dict[str, float]] = {}
        for sq_id, items in shortlists.items():
            question = {"rel": {"type": "noul", "instructions":
                                f'Does the text contain information that answers the question "{sub_questions[sq_id][:220]}"?'}}
            states = [f"{title}\n{text}"[:STATE_CHARS] for _, title, text in items]
            for (pid, _, _), r in zip(items, await self._predict_cached(states, question)):
                out.setdefault(pid, {})[sq_id] = float(r["answers"]["rel"]["noul"])
        return out

    async def verify(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        """First-pass NLI for (claim, evidence_text) pairs in one batch. The caller picks the
        evidence (a located quote's surroundings, or the best-matching window); it is trimmed to
        fit the context. Returns (relation, confidence)."""
        if not pairs:
            return []
        states = [{"statement": c, "text": t[:STATE_CHARS]} for c, t in pairs]
        return [(r["answers"]["relation"]["choice"], float(r["answers"]["relation"]["confidence"]))
                for r in await self._predict_cached(states, VERIFY_QUESTION)]
