"""Runtime configuration and environment loading."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # --- Generative LLM providers (all OpenAI-compatible; any subset may be set) ---
    GEMINI_API_KEY: str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_FAST_MODEL: str = "gemini-3.5-flash-lite"
    # 2.5 models return 404 "no longer available to new users" (2026-09); 3.8-flash timed out at 60 s.
    GEMINI_SYNTH_MODEL: str = "gemini-3.5-flash"
    GEMINI_VERIFY_MODEL: str = "gemini-3.5-flash-lite"

    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_FAST_MODEL: str = "openai/gpt-oss-20b"
    GROQ_SYNTH_MODEL: str = "openai/gpt-oss-120b"
    # Each Groq model has its own tokens-per-minute bucket; verifying off the synthesis model avoids
    # waiting for its bucket to refill (measured 11-16 s stalls). gpt-oss-20b answered the verify
    # prompt in 0.8 s vs 1.6 s for qwen3.8-27b, which also hit a separate output-token limit.
    GROQ_VERIFY_MODEL: str = "openai/gpt-oss-20b"

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_FAST_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_SYNTH_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"

    OLLAMA_BASE_URL: str = ""  # e.g. http://localhost:11434/v1
    OLLAMA_FAST_MODEL: str = "llama3.1:8b"
    OLLAMA_SYNTH_MODEL: str = "llama3.1:8b"

    # Provider order per role (comma-separated; unconfigured providers are skipped)
    FAST_PROVIDER_ORDER: str = "groq,gemini,openrouter,ollama"
    SYNTH_PROVIDER_ORDER: str = "gemini,groq,openrouter,ollama"
    VERIFY_PROVIDER_ORDER: str = "groq,gemini,openrouter,ollama"
    # Optional USD per 1M tokens (input,output) if you ever route to a paid provider
    LLM_PRICE_PER_MTOK: str = ""  # e.g. "gemini=0.10/0.40,groq=0/0"

    # --- Laya (local, open-source System 1 decision model) ---
    LAYA_ENABLED: bool = True
    LAYA_MODEL: str = "convaiinnovations/laya"
    LAYA_SUBFOLDER: str = ""  # "" = English checkpoint; "multilingual" for non-English research
    LAYA_DEVICE: str = ""  # "" = auto (CUDA -> Apple MPS -> CPU); "cpu" to force

    # --- Search ---
    SEARCH_PROVIDER_ORDER: str = "searxng,ddg,tavily,brave"
    SEARXNG_URL: str = ""
    DDG_BACKENDS: str = "yahoo,bing"
    SOURCE_ROUTING: bool = True  # Laya-classified facets -> OpenAlex / Wikipedia / news / official-doc queries  # raced in parallel per query; "auto" is the fallback
    TAVILY_API_KEY: str = ""
    BRAVE_API_KEY: str = ""
    RESULTS_PER_QUERY: int = 5
    USE_JINA_READER: bool = True

    # --- Guardrails ---
    HTTP_TIMEOUT_SECONDS: float = 8.0
    LLM_TIMEOUT_SECONDS: float = 60.0
    FETCH_CONCURRENCY: int = 8
    COST_BUDGET_USD: float = 0.05
    MAX_RUN_SECONDS: float = 150.0
    SYNTH_EVIDENCE_CHAR_BUDGET: int = 48000  # ~12k tokens of evidence for synthesis
    CACHE_TTL_HOURS: float = 24.0

    # --- Web app ---
    DB_PATH: str = str(ROOT / "data" / "research.db")
    PUBLIC_BASE_URL: str = ""  # used when building share links; defaults to request host
    ACCESS_TOKEN: str = ""  # if set, starting research requires this token
    RUNS_PER_HOUR_PER_IP: int = 12

    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")


settings = Settings()
