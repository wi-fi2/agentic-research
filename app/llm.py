"""Generative LLM access through a chain of free-tier, OpenAI-compatible providers.

Each role ("fast" for decomposition/reformulation, "synth" for report writing, "verify" for
escalated claim checks, on its own model so it has its own rate-limit bucket) has an
ordered provider list. On 429 / 413 / 5xx / timeout the next provider is tried, so a
depleted free quota degrades gracefully instead of failing the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    fast_model: str
    synth_model: str
    verify_model: str = ""

    def model_for(self, role: str) -> str:
        if role == "fast":
            return self.fast_model
        if role == "verify":
            return self.verify_model or self.synth_model
        return self.synth_model


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLMUnavailable(RuntimeError):
    pass


def _providers() -> dict[str, Provider]:
    s = settings
    table = {
        "gemini": Provider("gemini", s.GEMINI_BASE_URL, s.GEMINI_API_KEY, s.GEMINI_FAST_MODEL, s.GEMINI_SYNTH_MODEL,
                           s.GEMINI_VERIFY_MODEL),
        "groq": Provider("groq", s.GROQ_BASE_URL, s.GROQ_API_KEY, s.GROQ_FAST_MODEL, s.GROQ_SYNTH_MODEL,
                         s.GROQ_VERIFY_MODEL),
        "openrouter": Provider(
            "openrouter", s.OPENROUTER_BASE_URL, s.OPENROUTER_API_KEY, s.OPENROUTER_FAST_MODEL, s.OPENROUTER_SYNTH_MODEL
        ),
        "ollama": Provider("ollama", s.OLLAMA_BASE_URL, "ollama", s.OLLAMA_FAST_MODEL, s.OLLAMA_SYNTH_MODEL),
    }
    usable = {}
    for name, p in table.items():
        if name == "ollama" and s.OLLAMA_BASE_URL:
            usable[name] = p
        elif name != "ollama" and p.api_key:
            usable[name] = p
    return usable


def _prices() -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for part in filter(None, (x.strip() for x in settings.LLM_PRICE_PER_MTOK.split(","))):
        try:
            name, pair = part.split("=")
            i, o = pair.split("/")
            out[name.strip()] = (float(i), float(o))
        except ValueError:
            log.warning("Ignoring malformed LLM_PRICE_PER_MTOK entry %r", part)
    return out


def configured_providers() -> list[str]:
    return list(_providers())


def chain_for(role: str) -> list[Provider]:
    order = {"fast": settings.FAST_PROVIDER_ORDER, "verify": settings.VERIFY_PROVIDER_ORDER}.get(
        role, settings.SYNTH_PROVIDER_ORDER)
    provs = _providers()
    return [provs[n.strip()] for n in order.split(",") if n.strip() in provs]


# Model ids that are not chat/text models (speech, embeddings, image/video, safety classifiers).
_NON_CHAT = re.compile(r"whisper|tts|guard|embed|imagen|veo|lyria|aqa|orpheus|playai|native-audio|"
                       r"-image|robotics|computer-use|prompt-guard|safeguard", re.I)


def _selectable(provider: str, model: str) -> bool:
    if _NON_CHAT.search(model):
        return False
    # OpenRouter lists paid models too; only offer :free ones so a selection can never cost money.
    return provider != "openrouter" or model.endswith(":free")


async def list_models() -> dict[str, list[str]]:
    """Selectable chat models per configured provider: the provider's live /models list when it
    answers, else just the configured models. Used by the UI's model picker and to validate picks."""
    out: dict[str, list[str]] = {}
    async with httpx.AsyncClient(timeout=10) as client:
        provs = _providers()
        lists = await asyncio.gather(*(_available_models(client, p) for p in provs.values()))
    for (name, p), available in zip(provs.items(), lists):
        configured = {p.fast_model, p.synth_model, p.verify_model} - {""}
        models = {m for m in (available or set()) | configured if _selectable(name, m)}
        out[name] = sorted(models)
    return out


RETRYABLE = {408, 409, 413, 429, 500, 502, 503, 504, 529}

# Free-tier catalogues churn; if a configured model disappears, fall back to the first of these that exists.
MODEL_PREFERENCES = {
    "groq": {
        "fast": ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "llama-3.1-8b-instant"],
        # gpt-oss-20b second: its own 200k tokens/day. qwen3.8-27b caps output at ~1k tokens/min,
        # too little for a ~3k-token report.
        "synth": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"],
        "verify": ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
    },
    "gemini": {
        "fast": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"],
        "synth": ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        "verify": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"],
    },
}
_model_cache: dict[str, Optional[set[str]]] = {}
# Models whose daily token quota ran out: (provider, model) -> monotonic time until which to skip them.
_exhausted: dict[tuple[str, str], float] = {}
EXHAUSTED_COOLDOWN_S = 3600
MIN_OUTPUT_TOKENS = 600  # never shrink an output budget below this to fit a model's output cap


class ContextTooLarge(LLMUnavailable):
    """Every provider rejected the request for size; `token_limit` is the smallest limit reported."""

    def __init__(self, message: str, token_limit: int) -> None:
        super().__init__(message)
        self.token_limit = token_limit


async def _available_models(client: httpx.AsyncClient, prov: Provider) -> Optional[set[str]]:
    if prov.name not in _model_cache:
        try:
            r = await client.get(prov.base_url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {prov.api_key}"}, timeout=10)
            ids = {m["id"].removeprefix("models/") for m in r.json().get("data", [])} if r.status_code == 200 else None
        except Exception:
            ids = None
        _model_cache[prov.name] = ids or None  # None -> unknown; use configured name as-is
    return _model_cache[prov.name]


async def resolve_model(client: httpx.AsyncClient, prov: Provider, role: str) -> str:
    configured = prov.model_for(role)
    available = await _available_models(client, prov)
    if available is None or configured in available:
        return configured
    for candidate in MODEL_PREFERENCES.get(prov.name, {}).get(role, []):
        if candidate in available:
            log.warning("%s model %r is unavailable; using %r instead", prov.name, configured, candidate)
            return candidate
    newest = newest_gemini_flash(available, lite=(role != "synth")) if prov.name == "gemini" else None
    if newest:
        log.warning("gemini model %r is unavailable; using newest Flash %r", configured, newest)
        return newest
    return configured


def newest_gemini_flash(available: set[str], lite: bool) -> Optional[str]:
    """Pick the highest-versioned plain text Flash model (e.g. gemini-3.8-flash) from a model list."""
    pat = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?$")
    found = []
    for m in available:
        g = pat.match(m)
        if g:
            found.append((float(g.group(1)), bool(g.group(2)) == lite, m))
    if not found:
        return None
    return max(found, key=lambda x: (x[1], x[0]))[2]


def _is_exhausted(prov: str, model: str) -> bool:
    import time

    until = _exhausted.get((prov, model))
    return until is not None and time.monotonic() < until


def mark_exhausted(prov: str, model: str) -> None:
    import time

    _exhausted[(prov, model)] = time.monotonic() + EXHAUSTED_COOLDOWN_S
    log.warning("%s/%s daily token quota exhausted; skipping it for %d min", prov, model, EXHAUSTED_COOLDOWN_S // 60)


async def candidate_models(client: httpx.AsyncClient, prov: Provider, role: str, limit: int = 2) -> list[str]:
    """The configured model plus known-available alternates on the same provider, skipping models
    whose daily quota is exhausted."""
    first = await resolve_model(client, prov, role)
    available = await _available_models(client, prov)
    pool = [first] + [m for m in MODEL_PREFERENCES.get(prov.name, {}).get(role, [])
                      if m != first and available is not None and m in available]
    out = [m for m in pool if not _is_exhausted(prov.name, m)][:limit]
    return out or [first]


def _is_reasoning_model(model: str) -> bool:
    return any(k in model for k in ("gpt-oss", "qwen3", "deepseek-r1", "o3", "o4"))


def _retry_after(r: httpx.Response) -> float:
    h = r.headers.get("retry-after")
    if h:
        try:
            return float(h)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", r.text)
    return (int(m.group(1) or 0) * 60 + float(m.group(2))) if m else 2.0


def _token_limit(text: str) -> Optional[int]:
    m = re.search(r"Limit (\d+), Requested (\d+)", text)
    return int(m.group(1)) if m else None


async def complete(
    messages: list[dict[str, str]],
    role: str = "fast",
    *,
    json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    override: Optional[tuple[str, str]] = None,
) -> LLMResult:
    """Run a chat completion against the first provider in the role's chain that succeeds.

    Rate limits (429) are waited out when the wait is short; oversized requests (413) and
    other failures fall through to the next provider. If every provider rejected the
    request for size, ContextTooLarge tells the caller how small to make it.

    `override` = (provider, model) picked by the user for this run: that exact model is tried
    first, and the role's normal chain stays behind it as the fallback.
    """
    chain = chain_for(role)
    pinned: Optional[Provider] = None
    if override:
        pinned = _providers().get(override[0])
        if pinned:
            chain = [pinned] + chain
    if not chain:
        raise LLMUnavailable(
            "No LLM provider configured. Set GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY or OLLAMA_BASE_URL."
        )
    prices = _prices()
    errors: list[str] = []
    size_limits: list[int] = []
    async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
        for idx, prov in enumerate(chain):
            if pinned is not None and idx == 0:
                models = [override[1]]
            else:
                models = await candidate_models(client, prov, role)
            short_wait: list[float] = []  # per-minute limits seen on this provider that reset soon
            for mi, model in enumerate(models + models):
                if mi == len(models):  # second pass: only after every model hit a short rate limit
                    if not short_wait:
                        break
                    await asyncio.sleep(min(short_wait) * random.uniform(1.0, 1.2) + 0.5)
                    short_wait.clear()
                if mi >= len(models) and _is_exhausted(prov.name, model):
                    continue
                has_alternate = mi + 1 < len(models)
                body: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode and prov.name != "ollama":
                    body["response_format"] = {"type": "json_object"}
                if prov.name == "groq" and _is_reasoning_model(model):
                    body["reasoning_effort"] = "low"  # reasoning tokens count against max_tokens and TPM
                outcome = "next_model"
                for attempt in range(3):
                    try:
                        r = await client.post(
                            prov.base_url.rstrip("/") + "/chat/completions",
                            headers={"Authorization": f"Bearer {prov.api_key}"},
                            json=body,
                        )
                    except (httpx.TimeoutException, httpx.TransportError) as e:
                        errors.append(f"{prov.name}/{model}: {type(e).__name__}")
                        break
                    if r.status_code == 200:
                        data = r.json()
                        text = (data["choices"][0]["message"].get("content") or "").strip()
                        usage = data.get("usage") or {}
                        it, ot = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                        if not text:
                            errors.append(f"{prov.name}/{model}: empty completion")
                            break
                        pi, po = prices.get(prov.name, (0.0, 0.0))
                        return LLMResult(text, prov.name, model, it, ot, (it * pi + ot * po) / 1e6)
                    errors.append(f"{prov.name}/{model}: HTTP {r.status_code} {r.text[:200]}")
                    if r.status_code == 400 and "json_validate_failed" in r.text and "response_format" in body:
                        # Strict JSON mode sometimes rejects an otherwise usable answer; parse_json is tolerant.
                        body.pop("response_format")
                        continue
                    if "tokens per day" in r.text or "(TPD)" in r.text:
                        mark_exhausted(prov.name, model)  # daily quota: waiting minutes won't help
                        break
                    if "output tokens per minute" in r.text or "(OTPM)" in r.text:
                        cap = _token_limit(r.text)
                        if cap and cap - 50 >= MIN_OUTPUT_TOKENS and body["max_tokens"] > cap - 50:
                            body["max_tokens"] = cap - 50  # fits the model's output cap; enough for short JSON
                            continue
                        break  # this model's output cap is too small for the request; try another model
                    limit = _token_limit(r.text)
                    if r.status_code == 413 or (limit and "Request too large" in r.text):
                        if limit:
                            size_limits.append(limit)
                        outcome = "next_provider"  # same-provider models share the size cap
                        break
                    if r.status_code == 404 and "model" in r.text.lower():
                        _model_cache.pop(prov.name, None)  # re-resolve next time
                        break
                    if r.status_code in (429, 503, 529) and attempt < 2:
                        if has_alternate:
                            if _retry_after(r) <= 30:
                                short_wait.append(_retry_after(r))
                            break  # another model on this provider has its own rate-limit bucket
                        wait = _retry_after(r)
                        if wait <= 30 and mi < len(models) and len(models) > 1:
                            short_wait.append(wait)  # try the second pass after the shortest wait
                            break
                        if wait <= 30:  # per-minute token windows reset quickly on free tiers
                            await asyncio.sleep(wait * random.uniform(1.0, 1.2) + 0.5)
                            continue
                    break
                if outcome == "next_provider":
                    break
                log.warning("LLM %s/%s failed: %s", prov.name, model, errors[-1][:200])
    msg = "All LLM providers failed: " + " | ".join(errors)
    if size_limits and len(size_limits) == len(chain):
        raise ContextTooLarge(msg, min(size_limits))
    raise LLMUnavailable(msg)


def parse_json(text: str) -> Optional[dict]:
    """Tolerant JSON extraction (handles code fences and leading prose)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
