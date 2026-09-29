"""Domain models for state and data interchange (condensed from spec §5)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNCERTAIN = "uncertain"
    UNVERIFIED = "unverified"


class SubQuery(BaseModel):
    id: str
    query_text: str
    facet_description: str = ""
    coverage: float = 0.0
    is_resolved: bool = False
    types: List[str] = Field(default_factory=list)  # academic | policy | market | technical


class SearchResult(BaseModel):
    sub_query_id: str
    url: str
    title: str
    snippet: str
    rank: int = 0
    published_date: Optional[str] = None
    provider: str = ""
    kind: str = "web"  # web | paper | news | encyclopedia
    prefetched_text: Optional[str] = None  # e.g. an OpenAlex abstract: used as the document, no fetch


class RetrievedDocument(BaseModel):
    url: str
    title: str
    extracted_text: str
    content_hash: str
    http_status: int = 200
    method: str = "trafilatura"  # trafilatura | jina | snippet | api
    published_date: Optional[str] = None
    kind: str = "web"
    retrieval_latency_ms: float = 0.0
    scraped_at: datetime = Field(default_factory=utcnow)


class EvidencePassage(BaseModel):
    passage_id: str
    source_url: str
    source_title: str
    text_content: str
    content_hash: str
    published_date: Optional[str] = None
    bm25: Dict[str, float] = Field(default_factory=dict)  # sub_query_id -> score
    relevance: Dict[str, float] = Field(default_factory=dict)  # sub_query_id -> P(relevant)
    injection_risk: float = 0.0
    judged: bool = False


class Source(BaseModel):
    """A citable source in the registry, built before synthesis ([S1], [S2], ...)."""

    source_id: str
    url: str
    title: str
    published_date: Optional[str] = None
    passage_ids: List[str] = Field(default_factory=list)
    accessed_at: datetime = Field(default_factory=utcnow)
    kind: str = "web"
    authority: float = 0.5


class Claim(BaseModel):
    claim_id: str
    statement: str
    source_ids: List[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    confidence: float = 0.0
    supporting_source_ids: List[str] = Field(default_factory=list)
    contradicting_source_ids: List[str] = Field(default_factory=list)
    explanation: str = ""
    quotes: Dict[str, str] = Field(default_factory=dict)  # source_id -> quote the writer gave
    quote_status: Dict[str, str] = Field(default_factory=dict)  # source_id -> exact | fuzzy | missing


class RunMetrics(BaseModel):
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_usd: float = 0.0
    judge_calls: int = 0
    judge_input_tokens: int = 0
    claims_escalated: int = 0
    quotes_exact: int = 0
    quotes_fuzzy: int = 0
    quotes_missing: int = 0
    laya_cache_hits: int = 0
    routed_queries: int = 0
    search_calls: int = 0
    cache_hits: int = 0
    pages_fetched: int = 0
    pages_failed: int = 0
    passages_indexed: int = 0
    duplicates_removed: int = 0
    passages_quarantined: int = 0
    iterations: int = 0
    coverage: float = 0.0
    duration_seconds: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return round(self.llm_cost_usd, 6)  # Laya runs locally: $0


class ModelChoice(BaseModel):
    provider: str = Field(max_length=40)
    model: str = Field(min_length=1, max_length=200)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8, max_length=1000)
    depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")
    # Optional per-run model picks; None = the configured provider chain decides.
    writer: Optional[ModelChoice] = None  # synthesis
    helper: Optional[ModelChoice] = None  # decomposition, reformulation, claim checks


DEPTH_PRESETS = {
    "quick": {"max_iterations": 1, "sub_queries": 3, "top_k": 4, "routing": False},  # speed first
    "standard": {"max_iterations": 2, "sub_queries": 5, "top_k": 5},
    "deep": {"max_iterations": 3, "sub_queries": 7, "top_k": 6},
}


class FinalReport(BaseModel):
    run_id: str
    question: str
    title: str
    markdown: str
    sources: List[Source]
    claims: List[Claim]
    sub_queries: List[SubQuery]
    metrics: RunMetrics
    stop_reason: str = ""
    warnings: List[str] = Field(default_factory=list)
    verification_mode: str = "none"  # laya | laya+llm | llm | none
    generated_at: datetime = Field(default_factory=utcnow)
