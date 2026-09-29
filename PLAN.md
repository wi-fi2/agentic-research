# Agentic Research — Build Plan (near-zero cost edition)

Source of requirements: `## 1.md` (FR-1…FR-7, NFRs, guardrails, data models, workflow).
This plan keeps that design and changes one thing: **the cost model**. The spec targets
under $0.08/run on paid budget models. This build targets **$0.00 in LLM spend** on free
tiers, with all judgment steps on **Laya**, an open-source System 1 decision model that runs locally for $0.

---

## 1. Where the cost comes from and how each part goes to ~$0

| Spec component | Spec default (paid) | This build (free first) | Fallback chain |
|---|---|---|---|
| Web search | Tavily ($0.005–0.01/call) | **DuckDuckGo via `ddgs`** (no key) | SearXNG (self-host) → Tavily free 1k/mo → Brave |
| Page extraction | Trafilatura | **Trafilatura** (local, free) | Jina Reader `r.jina.ai` (keyless) → search snippet |
| Decompose / reformulate | GPT-4o-mini | **Groq free** (`llama-3.1-8b-instant`, 14.4k RPD) | Gemini free → OpenRouter `:free` → Ollama local |
| Synthesis | Sonnet / GPT-4o | **Gemini free** (Flash, large context) | Groq free → OpenRouter free → Ollama |
| Reranking | `bge-reranker-base` (1.5 GB RAM) | **BM25 (in code) → Laya Noul rerank (local)** | BM25 only |
| Sufficiency gate | Fast LLM call per loop | **Computed in code from Laya rerank scores** (no extra call) | BM25 coverage heuristic |
| Claim NLI verification | LLM call per claim×passage | **Cascade: numbers checked in code → Laya Choice → low-confidence claims to one batched free-LLM call** | One batched free-LLM JSON call |
| Prompt-injection filter | Regex / XML isolation | **XML isolation + regex screen** (Laya tested and rejected for this, see §2) | — |
| Storage / cache | SQLite | **SQLite**: runs, shares, 24 h search+fetch cache | — |
| PDF / DOCX | WeasyPrint (Cairo system libs) | **fpdf2 + python-docx** (pure Python, bundled DejaVu font) | Browser print CSS |
| Hosting | Docker / Postgres | **Single Docker container**: HF Spaces / Render / Cloud Run free tiers | Local `uvicorn` |

On a free tier, the real limit is **rate limits** (requests per day, tokens per minute),
not dollars. So the design goal is *fewer generative LLM calls per run*. The pipeline makes
**3–5 LLM calls per run**: 1 decompose, 0–2 reformulate, 1 synthesis, and 0–1 fallback
verification. At Groq/Gemini free quotas that allows hundreds of runs per day.

---

## 2. Where Laya (System 1) fits, and why it lowers cost

Laya (`pip install laya`, Apache-2.0, 421M-parameter English checkpoint) is a non-generative
*System 1* decision model. It returns typed probabilities (Noul = P(yes), Choice = distribution plus
confidence) for questions about a short text. It runs **in-process on CPU / Apple MPS / CUDA**, so every
judgment costs **$0** and no evidence leaves the machine. It replaces the earlier TypeSafe Jev API
(~$0.005/run). Generation stays with the free LLMs.

| Judgment in the spec | How Laya is used | Replaces |
|---|---|---|
| FR-3 rank chunks by relevance | BM25 shortlists 8 passages per sub-question; Laya scores only those (passage, sub-question) pairs with one Noul ("Does the text contain information that answers the question …?"), batched with `predict_batch` | Cross-encoder / paid rerank API |
| FR-7 sufficiency | **No call.** Code counts sub-questions with ≥ 2 Laya-relevant passages (P ≥ 0.75) from distinct URLs. Coverage ≥ 0.8 → stop | One LLM call per loop |
| FR-4 claim verification | Cascade. (1) Code: numbers in the claim must appear in the source, else escalate. (2) Laya Choice `supports / contradicts / says_nothing` on `{statement, text}`, with text = the best-matching ~1,600-char window. (3) Confidence < 0.2 → escalate. Escalated claims go to **one** batched free-LLM call | N×M LLM NLI calls |
| Prompt-injection screen | **Not Laya.** Regex screen plus XML isolation | — |

**Measured before shipping** (English checkpoint, Apple M-series MPS). The hand-labelled probe sets
were small (20 claim/source pairs, 13 relevance pairs, 10 injection probes), so treat these numbers
as rough:

| Check | Result | Decision |
|---|---|---|
| Relevance Noul, 3 phrasings | 85–92 % accuracy; best threshold 0.75–0.8 | Used, threshold 0.75 |
| Verification, 3 designs (claim in instruction; `{statement,text}` batched; two Nouls) | 60–75 % raw. Most misses were numbers and dates | Batched `{statement,text}` + numeric check in code |
| Verification cascade at confidence ≥ 0.2 | 90 % accurate on the claims Laya keeps; 50 % escalated | Used |
| Injection / jailbreak Nouls (own phrasing + `laya.guard_questions()`) | Clean DuckDB text 0.81, site navigation 1.0, while some real injections scored 0.29–0.31 | Rejected; regex + XML isolation instead |
| Latency, full ~400-token passage | ~330 ms per question on MPS, ~740 ms on CPU | Only shortlisted pairs are scored, not passages × all facets |

**On real runs** (Groq free tier + Laya, "DuckDB vs Polars", standard depth):
- Laya scored 55 passage–facet pairs in ~18 s.
- 14 of 18 claims escalated to the LLM, mostly for low confidence on paraphrased sentences.
- Total 80 s, **$0.0000**.

So Laya's main value here is free, local relevance ranking and sufficiency. On verification it
settles a minority of claims, and the free LLM handles the rest. Tighten or loosen
`VERIFY_MIN_CONFIDENCE` in `app/laya_judge.py` after measuring on your own questions.

Trade-offs of running locally:
- About 2 GB RAM, plus ~850 MB of weights downloaded on first start. The Docker image bakes them in.
- Free hosts with 512 MB (Render free) are too small. Hugging Face Spaces free CPU (16 GB) is fine.
- The model loads once in a background thread (~25–35 s). Runs that start earlier wait up to 90 s for it, then fall back to BM25 + LLM verification.

## 3. Architecture (as built)

```
Browser SPA (static HTML/JS, marked + DOMPurify)
   │  POST /api/research   ──►  background asyncio task
   │  GET  /api/research/{id}/events  (SSE live progress)
   ▼
FastAPI (app/main.py)
   └─ ResearchAgent (app/agent.py): explicit bounded state machine, no LangGraph
        1 decompose (LLM fast) ─► 2 search plan (code sanitize/dedupe)
        3 search (DDG/SearXNG/Tavily, cached) ─► 4 fetch+extract (httpx+trafilatura, Jina fallback, circuit breaker)
        5 chunk (~1,800 chars, overlap) ─► 6 dedupe (SHA-256 + 5-shingle Jaccard>0.85) + BM25 shortlist
        6b Laya judge (relevance per shortlisted pair, local) ─► 7 sufficiency (code)
        8 reformulate gaps (LLM fast) ──► loop to 3   (max_iterations, cost/time budget)
        9 synthesis (LLM synth, [S#] tags, XML-isolated evidence)
       10 claim extraction (code) + numeric check + Laya → LLM cascade ─► annotate / move / remove claims
       11 citation validation (regex vs registry) ─► 12 quality checks (code) ─► report
   ├─ store.py   SQLite: runs, share slugs, search/fetch cache
   └─ exporters.py  Markdown / DOCX (python-docx) / PDF (fpdf2)
```

Deviations from the spec, and the reasons:
- **No LangGraph.** The spec's own "simplest viable architecture" (§11) recommends a plain asyncio + Pydantic state machine. It has fewer dependencies and is easier to test.
- **No cross-encoder.** Laya does this job and also verifies claims, with one model in memory.
- **No WeasyPrint.** It needs Cairo/Pango system libraries. fpdf2 is pure Python and deploys anywhere.

---

## 4. Web app features

- Research form: question plus depth (`quick` = 1 loop / 3 sub-questions, `standard` = 2 / 5, `deep` = 3 / 7).
- Live progress timeline over SSE. Counters for sources, passages, LLM calls, Laya tokens, and $ cost.
- Report view: rendered markdown, clickable `[S#]` citation chips, claim-status badges
  (supported / conflicting / unverified / removed), a verification table, a bibliography with access dates, and run metrics.
- History of past runs (SQLite).
- **Share**: creates an unguessable public read-only link `/r/{slug}`, which can be revoked. Copy-link,
  native Web Share API, and X / LinkedIn / WhatsApp / Email intents.
- **Download**: Markdown, **DOCX**, **PDF** (server-rendered), plus Print (browser print CSS).
- Protection for public deploys: optional `ACCESS_TOKEN` and a per-IP run rate limit, so strangers
  can't use up your free quotas. Shared links stay public and read-only.

---

## 5. Guardrails mapped to spec §6

| Spec guardrail | Implementation |
|---|---|
| Max iterations | `depth` → `max_iterations` 1/2/3, hard stop |
| Cost & token budget | `COST_BUDGET_USD` (default $0.05) and `MAX_RUN_SECONDS` (default 150); stops the loop early and goes straight to synthesis |
| Timeouts | HTTP 8 s, LLM 60 s, Laya load wait 90 s |
| Backoff + jitter | 1.5 s base, ×2, ±25 %, max 10 s on 429/5xx/529 |
| Circuit breaker | 3 consecutive failures → domain disabled for the run |
| Empty search | Query rewrite (strip operators/punctuation) → next provider |
| Duplicates | SHA-256 exact + Jaccard > 0.85 |
| Paywalls/403 | Non-200 or < 300 chars → Jina fallback → snippet; paywall domain blocklist |
| Conflicts | Mixed support/contradiction → `uncertain`, listed under "Points of Disagreement" |
| Outdated info | Published date kept in metadata and shown in the bibliography |
| Injection | `<untrusted_web_evidence>` XML isolation + regex screen |
| Hallucinated citations | Regex audit. Unknown `[S#]` → `[UNVERIFIED: Missing Citation]` |
| Grounding mandate | Uncited factual sentences → `[UNVERIFIED: Evidence Not Found]` |

---

## 6. Phases (condensed from spec §8)

1. **Core pipeline**: config, schemas, search, extractor, evidence (chunk/dedupe/BM25), llm chain, agent, CLI. ✅
2. **Citations**: `[S#]` registry built before synthesis, regex validation, bibliography. ✅
3. **Verification + loops**: Laya judge + verification cascade, sufficiency loop, reformulation. ✅
4. **Web app**: FastAPI + SSE + SPA, history, share, exports. ✅
5. **Hardening**: Dockerfile, rate limit/access token, unit tests. ✅ Langfuse tracing and the eval runner (spec §10) are the next step. The run trace JSON is already stored for each run.

## 7. Running at $0

| Need | Free option | Env var |
|---|---|---|
| LLM (synthesis) | Google AI Studio key (Gemini Flash free tier) | `GEMINI_API_KEY` |
| LLM (fast calls) | Groq free key | `GROQ_API_KEY` |
| LLM (backup) | OpenRouter `:free` models / local Ollama | `OPENROUTER_API_KEY` / `OLLAMA_BASE_URL` |
| Search | DuckDuckGo (no key); optional self-hosted SearXNG | `SEARXNG_URL` |
| Judgments | Laya, local ($0) | `LAYA_ENABLED`, `LAYA_DEVICE` |
| Hosting | Hugging Face Spaces free CPU (16 GB RAM) / any host with ≥ 2.5 GB RAM | — |

Free-tier quotas change often. The provider chain moves to the next provider on 429/413/5xx, so
losing one tier degrades the service without breaking it.

---

## 8. Speed pass 1 (2026-09-29): measured before / after

Same question ("DuckDB vs Polars", standard depth), empty cache, Groq free tier + local Laya, 65 s cool-down between runs.

| Stage | Before (2 runs) | After (2 runs) | Change |
|---|---|---|---|
| Search, per round | 12.0–13.3 s | 4.3–8.8 s | ddgs `yahoo,bing` raced in parallel; "auto" is the fallback |
| LLM claim check | 11.4–16.1 s | 1.2–2.1 s | Verify role on `gpt-oss-20b`, its own Groq per-minute bucket |
| Laya relevance + verify | 17–19 s + 5–6.5 s | 13.5 s + 4.3–7.2 s | unchanged (next: fewer pairs + cache) |
| **Total** | **72.6–80.4 s** | **45.1–49.3 s** | **~38% faster, still $0** |

Robustness added along the way: strict JSON-mode rejections (`json_validate_failed`) now retry without JSON mode,
and a failing or rate-limited Groq model falls through to another model on the same provider. The first "after"
run (on `qwen3.8-27b`) hit exactly this: 13 claims were left unverified. It is excluded from the table.

## 9. Pass 2 (2026-09-29): smaller shortlist, Laya cache, quote-grounded citations

Same protocol as §8 (same question, standard depth, Groq free tier + local Laya, 65 s cool-downs).

| Metric | Pass 1 (2 runs) | Pass 2 cold (3 runs) | Notes |
|---|---|---|---|
| Total | 45.1 / 49.3 s | 37.4 / 45.2 / 41.8 s | ~12% faster; ~46% faster than the original 72.6–80.4 s |
| Laya relevance scoring | 13.5 s | 10.5–12.1 s | shortlist 8 → 5 per facet (40 → 25 pairs in round 1) |
| Laya claim check | 4.3–7.2 s | ~1.2–2.5 s | evidence is now a ~700-char window around the writer's quote |
| Claims escalated to LLM | 72–75% | 50–75% | **no clear improvement**: the hypothesis that quotes would make Laya confident is not confirmed |
| Writer quotes found in source | n/a | 75–83% (exact or fuzzy) | 17–25% of quotes were not in the cited source; those claims are re-checked by the LLM |
| Warm re-run of the same question | n/a | 24.5 s, round-1 Laya 0 s (33 cached scores) | plan cache makes sub-questions identical, so search, fetch and Laya caches all hit |

In one run, verification removed 4 claims. Checked by hand against the source: 3 were real writer errors (a
misattributed "1.6× faster", mmap credited to the wrong engine, a false "only one timing"). The 4th was a
debatable statement about how thin the evidence is.

## 10. Source routing (2026-09-29)

**What it does.**
- **Classification:** each sub-question's short search query gets four Laya yes/no questions (academic / policy / market / technical), plus a keyword backstop in code.
- **Specialist sources:** routed facets add free, keyless sources. OpenAlex abstracts are used directly as documents. DuckDuckGo news (bing backend) serves market facets. Up to 2 "official text" / "official documentation" web variants are added for policy and technical facets. Every run adds the top Wikipedia page for the main question.
- **Guaranteed scoring and a reserved slot:** Laya always scores the lead chunks of specialist results for the facet that fetched them. The best one that passes the relevance bar gets a reserved evidence slot.
- **Ranking:** evidence is ranked by `relevance × (0.75 + 0.25·authority) × recency`. Authority is a URL heuristic (official / peer-reviewed / docs high, social / UGC low). Recency only matters for market, policy and technical facets.
- **Toggle:** routing runs on Standard and Deep. Quick skips it for speed. `SOURCE_ROUTING=false` turns it off everywhere.

**Classifier evaluation** (25 hand-labelled sub-questions, English checkpoint):

| Design | Result |
|---|---|
| One Choice over 5 types, wording A | 40–44% |
| One Choice, wording B | 76%. All 5 "general background" items wrong, and confidence too low to gate on |
| Four Nouls, per-type thresholds (chosen) | 17/17 routes correct, 17/20 found. Never fires on "general" items, so Wikipedia is added for every run instead |

In real runs, classifying query + LLM facet description pushed the "academic" noul to 0.75–0.82 on DuckDB facets. That text
differs from what was evaluated, so only the query is classified now.

**Benchmark** (4 question types, standard depth, empty cache, 1 run per cell, so treat differences as indicative):

| Question | Off | Routing (final) |
|---|---|---|
| Technical (DuckDB vs Polars) | 35.4 s, 3 sources, quotes found 7/14 | 52.2 s, 5 sources incl. duckdb.org, quotes 12/13 |
| Policy (EU AI Act GPAI) | 29.4 s, 6 sources, quotes 17/17 | 52.7 s, 5 sources, quotes 10/10 (search stage slow this run) |
| Academic (IF vs calorie restriction) | 29.1 s, 8 sources, quotes 11/18 | 44.4 s, 6 sources incl. an OpenAlex paper, quotes 15/15 |
| Market (sodium-ion capacity) | 41.4 s, 2 sources, quotes 7/14 | 37.0 s, 9 dated news sources, quotes 12/15 |
| **Total** | 43/49 claims supported (88%); 19 sources; quotes 42/63 (67%) | 43/49 claims supported (88%); 25 sources; quotes 49/53 (92%) |

Trade-off: broader, more checkable evidence (+30% cited sources, writer quotes found 67% → 92%) with the same
claim-support rate, for about +10–13 s per run. Most of that is extra Laya scoring and searches on the free
backends. Two latency fixes are already in: lead chunks only for specialist docs (market run went from 86 to 49 Laya pairs), and
at most 2 extra web variants per run.

## 11. Quote grounding pass (2026-09-29)

**Dataset.** 224 claim/source pairs from 16 benchmark runs (the writer's quote plus the cited source text from each run's cache).
Reference labels came from free LLMs (`gpt-oss-20b`, `qwen3.8-27b`) on a stratified sample. Only 79 pairs got usable labels, and the
two-model overlap was too small (2 pairs) to measure label reliability, so the numbers below are rough.

**Quote locator** (measured on the 224 pairs):

| | Before | After |
|---|---|---|
| Quote found verbatim | 108 | 190 |
| Close match | 72 | 15 |
| Not found | 42 | 17 |

Causes of the old "not found" results:
- **Stitched fragments:** 25 quotes joined fragments with "…". Each fragment is now matched separately.
- **Short quotes:** 10 were short but verbatim. Exact matches of 3+ words are now accepted.
- **Hyphen variants:** Unicode non-breaking hyphens (U+2011) were not normalised.
- **Genuinely absent:** 7 quotes are not in any source.

**Laya as verifier, measured honestly.** Every variant sits on the same coverage/accuracy trade-off:
- 700-char window around the quote (current): conf ≥ 0.2 decides 28% of pairs at 77%.
- The quote alone: conf ≥ 0.5 decides 27% at 81%.
- A 300-char window, or a yes/no supports question: no better.

On the pairs Laya decides, always answering "supports" would score 82–90%. **Laya's verification verdicts add no measurable
accuracy over that baseline**, so the LLM verifier does the real work. Laya still checks every claim at no cost (1–2 s). A
confident Laya "supports" is at least as reliable as the baseline, and it saves an LLM call for those claims.

**Whole pipeline vs reference** (58 fully labelled claims): 76% agreement (always-supported baseline 81%).
- Errors: 5 over-credited, 5 under-credited, 2 missed refutations.
- 3 of the 5 over-credits were sentences about the evidence ("The evidence does not specify…"). These are now written untagged
  by rule, and pure gap statements are exempt from fact-checking (4 of 187 dataset claims).

**Bugs found and fixed along the way:**
- **Tags after the full stop:** tags written after a sentence's full stop ("… output. [S3] Next sentence") were attributed to
  the next sentence. This only occurred with the fallback writer model (1 of 17 reports), and affected claims' sources as well
  as quotes.
- **Leaked quote tags:** quotes containing curly quotes (“GWh”) were not parsed, and the raw tag leaked into the report.
- **Empty claims:** leftover punctuation (". [S13]") became a claim of its own.
- **Groq daily limits:** `gpt-oss-120b` allows 200k tokens/day on the free tier, roughly 15–20 standard runs. Exhausted models are now
  skipped for an hour, and `gpt-oss-20b` is the second writer.
- **Groq output caps:** `qwen3.8-27b` caps output at 1k tokens/min, so the app retries it with a budget that fits. When every
  model on the provider is rate-limited briefly, it waits once and retries.

**New in reports:** an *Evidence Quotes* section lists each claim's exact source quote and whether it was found
verbatim, as a close match, or not at all, so readers can check the grounding themselves.
