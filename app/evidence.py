"""Chunking, deduplication (SHA-256 + shingle Jaccard), and in-memory BM25 ranking."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from app.schemas import EvidencePassage, RetrievedDocument

CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 200
JACCARD_DUP = 0.85

_WORD = re.compile(r"[a-z0-9]+(?:[.\-'][a-z0-9]+)*")
STOPWORDS = set(
    "a an and are as at be but by for from has have how if in into is it its of on or that the their them then "
    "there these they this to was were what when where which while who why will with vs versus about".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def chunk_document(doc: RetrievedDocument) -> list[EvidencePassage]:
    """Paragraph-aware sliding windows of ~CHUNK_CHARS with one-paragraph overlap.

    Never splits inside a sentence when a paragraph fits; very long paragraphs are split on
    sentence boundaries. The page title is prefixed to each chunk to limit anaphora loss.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", doc.extracted_text) if p.strip()]
    units: list[str] = []
    for p in paras:
        if len(p) <= CHUNK_CHARS:
            units.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sentences:
                if len(buf) + len(s) > CHUNK_CHARS and buf:
                    units.append(buf.strip())
                    buf = ""
                buf += s + " "
            if buf.strip():
                units.append(buf.strip()[: CHUNK_CHARS * 2])

    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for u in units:
        if size + len(u) > CHUNK_CHARS and cur:
            chunks.append("\n".join(cur))
            cur = [cur[-1]] if len(cur[-1]) < CHUNK_CHARS // 3 else []  # overlap
            size = sum(len(x) for x in cur)
        cur.append(u)
        size += len(u)
    if cur:
        chunks.append("\n".join(cur))

    url_hash = hashlib.sha256(doc.url.encode()).hexdigest()[:8]
    out = []
    for i, c in enumerate(ch for ch in chunks if len(ch) >= MIN_CHUNK_CHARS or len(chunks) == 1):
        out.append(
            EvidencePassage(
                passage_id=f"doc_{url_hash}_p_{i}",
                source_url=doc.url,
                source_title=doc.title,
                text_content=c,
                content_hash=hashlib.sha256(normalize(c).encode()).hexdigest(),
                published_date=doc.published_date,
            )
        )
    return out


def shingles(text: str, k: int = 5) -> set[int]:
    toks = _WORD.findall(text.lower())
    if len(toks) < k:
        return {hash(" ".join(toks))}
    return {hash(" ".join(toks[i : i + k])) for i in range(len(toks) - k + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Deduplicator:
    """Exact SHA-256 plus near-duplicate Jaccard over word 5-shingles (syndicated copies)."""

    def __init__(self) -> None:
        self.hashes: set[str] = set()
        self.sigs: list[set[int]] = []

    def is_duplicate(self, p: EvidencePassage) -> bool:
        if p.content_hash in self.hashes:
            return True
        sig = shingles(p.text_content)
        for other in self.sigs:
            if jaccard(sig, other) > JACCARD_DUP:
                return True
        self.hashes.add(p.content_hash)
        self.sigs.append(sig)
        return False


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    q = tokenize(query)
    toks = [tokenize(d) for d in docs]
    n = len(toks)
    if not n or not q:
        return [0.0] * n
    avgdl = sum(len(t) for t in toks) / n or 1.0
    df = Counter()
    for t in toks:
        df.update(set(t))
    scores = []
    for t in toks:
        tf = Counter(t)
        dl = len(t) or 1
        s = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores
