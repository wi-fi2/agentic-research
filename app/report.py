"""Post-synthesis processing: citation validation, claim extraction, annotation, bibliography, QA.

Deterministic code only; no model calls. The synthesizer writes `[S#]` tags that map to a
source registry built *before* synthesis; this module audits them (spec §7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.schemas import Claim, ClaimStatus, RunMetrics, Source

# `[S3: "quote"]`; the quote may itself contain curly quotes (e.g. “GWh”), so match lazily up to `"]`.
QUOTED_CITE_RE = re.compile(r"\[\s*(S\d+)\s*[:,]?\s*[\"“](.{3,400}?)[\"”]\s*\]")
# Tags placed after a sentence's full stop ("... output. [S3] Next sentence") belong to that sentence.
TRAILING_TAGS_RE = re.compile(r"([.!?])((?:\s*\[(?:S\d+|UNVERIFIED:[^\]]*)\](?:⟦\d+⟧)?)+)(?=\s|$)")
QUOTE_MARK_RE = re.compile(r"⟦(\d+)⟧")
CITE_RE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[*A-Z0-9])")
NON_FACT_SECTIONS = ("analyst interpretation", "open questions", "research gaps", "methodology")
# A sentence whose subject is the evidence itself and which reports a gap ("The evidence does not
# specify...", "Evidence on X is limited...", "No reliable figure is available...") describes the
# research, not the world, so it is not fact-checked. Mixed sentences (a fact plus a gap) still are.
GAP_STATEMENT_RE = re.compile(
    r"^\W*(?:(?:the|available|current|published|existing)\s+)*(?:evidence|sources?|data|literature|studies|research|reporting|information)\b"
    r"[^.;]{0,90}?\b(?:is|are|remains?|was|were|appears?)?\s*(?:limited|lacking|scarce|sparse|thin|unclear|inconclusive|insufficient|silent)\b"
    r"|^\W*(?:(?:the|available|current|published|existing)\s+)*(?:evidence|sources?|data|literature|studies|research|reporting)\b"
    r"[^.;]{0,60}?\b(?:does|do|did)\s+not\s+(?:specify|mention|report|provide|address|state|include|give|detail|establish)\b"
    r"|^\W*no\s+(?:\w+\s+){0,3}(?:data|evidence|information|figures?|studies)\b[^.;]{0,60}\b(?:available|reported|published|found|provided)\b",
    re.I)
MISSING_CITATION = "[UNVERIFIED: Missing Citation]"
NO_EVIDENCE = "[UNVERIFIED: Evidence Not Found]"


def normalize_citations(text: str, valid: set[str]) -> tuple[str, list[str], int]:
    """Canonicalise `[S1, S3]` -> `[S1][S3]`; replace ids not in the registry. Returns (text, ids, orphans)."""
    ids: list[str] = []
    orphans = 0

    def repl(m: re.Match) -> str:
        nonlocal orphans
        out = []
        for raw in re.split(r"\s*[,;]\s*", m.group(1)):
            sid = raw if raw.startswith("S") else "S" + raw
            if sid in valid:
                if sid not in ids:
                    ids.append(sid)
                out.append(f"[{sid}]")
            else:
                orphans += 1
                out.append(MISSING_CITATION)
        return "".join(out)

    return CITE_RE.sub(repl, text), ids, orphans


@dataclass
class Segment:
    text: str
    claim_id: str | None = None
    uncited_fact: bool = False


@dataclass
class Line:
    raw: str
    segments: list[Segment] | None = None  # None -> emit raw


@dataclass
class ReportDraft:
    title: str
    lines: list[Line]
    claims: list[Claim]
    orphan_citations: int
    uncited_facts: int
    cited_ids: list[str] = field(default_factory=list)


def _is_structural(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith(("#", "|", "```", ">", "---")) or bool(re.match(r"^[-*_]{3,}$", s))


def parse_draft(markdown: str, valid_ids: set[str], question: str) -> ReportDraft:
    markdown = re.sub(r"^```(?:markdown)?\s*\n|\n```\s*$", "", markdown.strip())
    markdown = re.sub("[\u200b\u200c\u200d\u2060\ufeff]", "", markdown)  # zero-width chars LLMs leave around tags
    # Some models (e.g. gpt-oss) write citations with their native brackets: 【S1: "..."】 / ［S1］.
    markdown = markdown.translate(str.maketrans({"【": "[", "】": "]", "［": "[", "］": "]", "〔": "[", "〕": "]"}))
    # Drop any bibliography the model wrote anyway; code renders the audited one.
    markdown = re.split(r"\n#+\s*(references|bibliography|sources)\s*\n", markdown, flags=re.I)[0]

    title = question.strip().rstrip("?")
    lines: list[Line] = []
    claims: list[Claim] = []
    cited: list[str] = []
    orphans = uncited = 0
    section = ""
    in_code = False

    quotes: list[tuple[str, str]] = []

    def stash(m: re.Match) -> str:
        quotes.append((m.group(1), m.group(2).strip()))
        return f"[{m.group(1)}]⟦{len(quotes) - 1}⟧"

    # `[S3: "exact words"]` -> `[S3]` + a marker, so quotes (which may contain periods) do not
    # break sentence splitting; markers are resolved per sentence and then removed.
    markdown = QUOTED_CITE_RE.sub(stash, markdown)
    markdown = TRAILING_TAGS_RE.sub(lambda m: " " + m.group(2).strip() + m.group(1), markdown)

    for raw in markdown.splitlines():
        s = raw.strip()
        if s.startswith("```"):
            in_code = not in_code
        if s.startswith("# ") and not lines:
            title = s[2:].strip()
            continue
        if s.startswith("#"):
            section = s.lstrip("#").strip().lower()
        norm, ids, orph = normalize_citations(raw, valid_ids)
        orphans += orph
        for i in ids:
            if i not in cited:
                cited.append(i)
        if in_code or _is_structural(raw):
            lines.append(Line(QUOTE_MARK_RE.sub("", norm)))
            continue

        m = re.match(r"^(\s*(?:[-*+]|\d+[.)])\s+)?(.*)$", norm)
        prefix, body = (m.group(1) or ""), m.group(2)
        fact_section = not any(k in section for k in NON_FACT_SECTIONS)
        pieces = SENT_SPLIT.split(body)
        # Re-attach fragments that are only citation tags (e.g. "Fact. [S1]") to the previous sentence.
        merged: list[str] = []
        for p in pieces:
            if merged and re.fullmatch(r"(\[[^\]]+\]\s*)+[.]?", p.strip()):
                merged[-1] += " " + p
            else:
                merged.append(p)

        segs: list[Segment] = [Segment(prefix)] if prefix else []
        for sent in merged:
            sids = [x for x in re.findall(r"\[(S\d+)\]", sent) if x in valid_ids]
            words = len(re.findall(r"\w+", sent))
            marks = [quotes[int(i)] for i in QUOTE_MARK_RE.findall(sent)]
            sent = QUOTE_MARK_RE.sub("", sent)
            is_gap = bool(GAP_STATEMENT_RE.search(re.sub(r"[*_`]", "", sent)))
            bare = re.sub(r"\s*\[[^\]]*\]", "", sent).strip()
            if sids and len(re.findall(r"\w+", bare)) < 2 and fact_section:
                # Only tags / punctuation left (e.g. ". [S13]"): they belong to the previous claim.
                prev = next((sg for sg in reversed(segs) if sg.claim_id), None)
                if prev is not None:
                    pc = next(c for c in claims if c.claim_id == prev.claim_id)
                    pc.source_ids += [x for x in sids if x not in pc.source_ids]
                    pc.quotes.update({sid: q for sid, q in marks if sid in sids and sid not in pc.quotes})
                    prev.text = prev.text.rstrip() + " " + " ".join(f"[{x}]" for x in sids)
                    continue
            if sids and fact_section and not is_gap:
                cid = f"c{len(claims) + 1}"
                statement = re.sub(r"\s*\[[^\]]*\]", "", sent).strip()
                statement = re.sub(r"[*_`]", "", statement)
                cited_quotes = {sid: q for sid, q in marks if sid in sids}
                claims.append(Claim(claim_id=cid, statement=statement, source_ids=list(dict.fromkeys(sids)),
                                    quotes=cited_quotes))
                segs.append(Segment(sent, claim_id=cid))
            elif fact_section and words >= 8 and not sids and not is_gap and MISSING_CITATION not in sent \
                    and not sent.rstrip().endswith(":"):
                uncited += 1
                segs.append(Segment(sent, uncited_fact=True))
            else:
                segs.append(Segment(sent))
        lines.append(Line(norm, segs))

    return ReportDraft(title, lines, claims, orphans, uncited, cited)


def _annotate(sent: str, claim: Claim) -> str | None:
    if claim.status == ClaimStatus.REFUTED:
        return None
    label = {
        ClaimStatus.UNCERTAIN: " *[Conflicting evidence]*",
        ClaimStatus.UNVERIFIED: " *[Unverified]*",
    }.get(claim.status, "")
    if not label:
        return sent
    stripped = sent.rstrip()
    trail = stripped[-1] if stripped.endswith((".", "!", "?")) else ""
    body = stripped[:-1] if trail else stripped
    return f"{body}{label}{trail}"


def _append_tag(sent: str, tag: str) -> str:
    stripped = sent.rstrip()
    trail = stripped[-1] if stripped.endswith((".", "!", "?")) else ""
    body = stripped[:-1] if trail else stripped
    return f"{body} {tag}{trail}"


def render_body(draft: ReportDraft) -> str:
    by_id = {c.claim_id: c for c in draft.claims}
    out: list[str] = []
    for line in draft.lines:
        if line.segments is None:
            out.append(line.raw)
            continue
        segs = line.segments
        prefix = ""
        if segs and re.fullmatch(r"\s*(?:[-*+]|\d+[.)])\s+", segs[0].text):
            prefix, segs = segs[0].text, segs[1:]
        parts: list[str] = []
        for seg in segs:
            if seg.claim_id:
                t = _annotate(seg.text, by_id[seg.claim_id])
                if t is not None:
                    parts.append(t.strip())
            elif seg.uncited_fact:
                parts.append(_append_tag(seg.text, NO_EVIDENCE).strip())
            elif seg.text.strip():
                parts.append(seg.text.strip())
        if parts:  # a line whose every sentence was refuted disappears entirely
            out.append(prefix + " ".join(parts))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def render_appendix(draft: ReportDraft, sources: list[Source], metrics: RunMetrics, verification_mode: str) -> str:
    claims = draft.claims
    counts = {s: sum(1 for c in claims if c.status == s) for s in ClaimStatus}
    parts = []
    if verification_mode != "none":
        engine = {
            "laya": "Laya, a local System 1 decision model",
            "laya+llm": "Laya, a local System 1 decision model, with low-confidence and numeric claims escalated to an LLM verifier",
        }.get(verification_mode, "an LLM verifier")
        parts.append("## Verification Summary\n")
        parts.append(f"Every cited sentence was checked against its cited sources by an independent verifier: {engine}.\n")
        parts.append("| Status | Claims |\n|---|---|")
        parts.append(f"| Supported | {counts[ClaimStatus.SUPPORTED]} |")
        parts.append(f"| Conflicting evidence | {counts[ClaimStatus.UNCERTAIN]} |")
        parts.append(f"| Unverified | {counts[ClaimStatus.UNVERIFIED]} |")
        parts.append(f"| Removed (refuted by cited source) | {counts[ClaimStatus.REFUTED]} |")
        parts.append(f"| Uncited factual sentences | {draft.uncited_facts} |\n")
        qs = [st for c in claims for st in c.quote_status.values()]
        if qs:
            parts.append(f"Quote grounding: {qs.count('exact')} quotes found verbatim in their source, "
                         f"{qs.count('fuzzy')} with minor wording differences, {qs.count('missing')} not found "
                         "(those claims were re-checked against the full source).\n")
        conflicts = [c for c in claims if c.status == ClaimStatus.UNCERTAIN]
        if conflicts:
            parts.append("### Points of Disagreement and Contradictory Evidence\n")
            for c in conflicts:
                sup = ", ".join(f"[{s}]" for s in c.supporting_source_ids) or "none"
                con = ", ".join(f"[{s}]" for s in c.contradicting_source_ids) or "none"
                parts.append(f"- {c.statement} — supported by {sup}; contradicted by {con}.")
            parts.append("")
        removed = [c for c in claims if c.status == ClaimStatus.REFUTED]
        if removed:
            parts.append("### Claims Removed After Verification\n")
            for c in removed:
                con = ", ".join(f"[{s}]" for s in c.contradicting_source_ids)
                parts.append(f"- ~~{c.statement}~~ — contradicted by {con}.")
            parts.append("")

    quoted = [c for c in claims if c.quotes and c.status != ClaimStatus.REFUTED]
    if quoted:
        parts.append("## Evidence Quotes\n")
        parts.append("The exact source text each claim rests on, as located in the source by code.\n")
        mark = {"exact": "found verbatim", "fuzzy": "close match", "missing": "not found in source"}
        for c in quoted:
            words = c.statement.split()
            short = " ".join(words[:18]) + ("…" if len(words) > 18 else "")
            ev = "; ".join(
                f'[{sid}] "{q.strip()}"' + (f" *({mark[c.quote_status[sid]]})*" if sid in c.quote_status else "")
                for sid, q in c.quotes.items())
            parts.append(f"- {short} — {ev}")
        parts.append("")

    cited = set(draft.cited_ids)
    used = [s for s in sources if s.source_id in cited]
    parts.append("## References\n")
    for s in used:
        pub = f", published {s.published_date}" if s.published_date else ""
        title = s.title.replace("[", "(").replace("]", ")")
        label = {"paper": "research paper", "news": "news", "encyclopedia": "encyclopedia"}.get(s.kind) or (
            "official / primary source" if s.authority >= 1.0 else "documentation" if s.authority >= 0.9 else "")
        label = f" · *{label}*" if label else ""
        parts.append(f"- **[{s.source_id}]** [{title}]({s.url}) — {_domain(s.url)}{pub}{label}. Accessed {s.accessed_at:%Y-%m-%d}.")
    uncited_sources = [s for s in sources if s.source_id not in cited]
    if uncited_sources:
        parts.append("\n*Also consulted (not cited):* " + "; ".join(f"[{_domain(s.url)}]({s.url})" for s in uncited_sources))
    parts.append(
        f"\n---\n*Method: {metrics.iterations} search iteration(s), {metrics.pages_fetched} pages read, "
        f"{metrics.passages_indexed} passages indexed, evidence coverage {metrics.coverage:.0%}. "
        f"Run cost ${metrics.total_cost_usd:.4f}.*"
    )
    return "\n".join(parts)


def quality_checks(markdown: str, draft: ReportDraft, sources: list[Source]) -> list[str]:
    warnings = []
    words = len(re.findall(r"\w+", markdown))
    if words < 150:
        warnings.append(f"Report is short ({words} words).")
    if "executive summary" not in markdown.lower():
        warnings.append("Report is missing an Executive Summary section.")
    if draft.orphan_citations:
        warnings.append(f"{draft.orphan_citations} citation(s) referenced unknown sources and were replaced.")
    registry = {s.source_id for s in sources}
    stray = set(re.findall(r"\[(S\d+)\]", markdown)) - registry
    if stray:
        warnings.append(f"Bibliography integrity: unresolved ids {sorted(stray)}.")
    total = len(draft.claims) + draft.uncited_facts
    if total and len(draft.claims) / total < 0.8:
        warnings.append(f"Citation recall is {len(draft.claims) / total:.0%} (target 100%).")
    if not draft.claims:
        warnings.append("No cited claims were found in the report.")
    return warnings
