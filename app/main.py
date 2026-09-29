"""FastAPI web app: research runs with live SSE progress, history, share links, and exports."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import exporters, laya_judge, llm, store
from app.agent import ResearchAgent
from app.config import settings
from app.schemas import ModelChoice, ResearchRequest
from app.search import enabled_providers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
STATIC = Path(__file__).parent / "static"


class RunChannel:
    """In-memory event log + fan-out for one active run (replayed to late SSE subscribers)."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.done = False

    async def publish(self, event: dict) -> None:
        self.events.append(event)
        for q in list(self.subscribers):
            q.put_nowait(event)


channels: dict[str, RunChannel] = {}
tasks: set[asyncio.Task] = set()
rate_log: dict[str, deque] = defaultdict(deque)


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.conn()
    store.mark_interrupted_runs()
    store.cache_prune()
    laya_judge.start_loading()  # background: the server accepts requests while weights load
    log.info("LLM providers: %s | search: %s | Laya: %s", llm.configured_providers(), enabled_providers(),
             "loading" if settings.LAYA_ENABLED else "disabled")
    yield


app = FastAPI(title="Agentic Research", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def _check_access(token: Optional[str]) -> None:
    if settings.ACCESS_TOKEN and not (token and secrets.compare_digest(token, settings.ACCESS_TOKEN)):
        raise HTTPException(401, "This instance requires an access token.")


def _rate_limit(ip: str) -> None:
    window = rate_log[ip]
    now = time.time()
    while window and now - window[0] > 3600:
        window.popleft()
    if len(window) >= settings.RUNS_PER_HOUR_PER_IP:
        raise HTTPException(429, "Hourly research limit reached for this address. Please try again later.")
    window.append(now)


def _base_url(request: Request) -> str:
    return settings.PUBLIC_BASE_URL.rstrip("/") or str(request.base_url).rstrip("/")


# ---------------------------------------------------------------- pages

@app.get("/", include_in_schema=False)
@app.get("/run/{run_id}", include_in_schema=False)
@app.get("/r/{slug}", include_in_schema=False)
async def index(run_id: str | None = None, slug: str | None = None):
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
async def config():
    return {
        "llm_providers": llm.configured_providers(),
        "search_providers": enabled_providers(),
        "laya": laya_judge.status(),
        "requires_token": bool(settings.ACCESS_TOKEN),
        "cost_budget_usd": settings.COST_BUDGET_USD,
    }


@app.get("/api/models")
async def models():
    """Selectable chat models per configured provider, for the per-run model picker."""
    return {"providers": await llm.list_models()}


async def _validate_pick(pick: Optional[ModelChoice], label: str) -> Optional[tuple[str, str]]:
    """Accept only models the server lists as selectable (e.g. never a paid OpenRouter model)."""
    if pick is None:
        return None
    available = await llm.list_models()
    if pick.model not in available.get(pick.provider, []):
        raise HTTPException(400, f"{label} model {pick.provider}/{pick.model} is not available on this server.")
    return pick.provider, pick.model


@app.get("/api/health")
async def health():
    return {"ok": True}


# ---------------------------------------------------------------- research runs

async def _execute(run_id: str, req: ResearchRequest, channel: RunChannel) -> None:
    agent = ResearchAgent(req.question, req.depth, emit=channel.publish, run_id=run_id,
                          writer=(req.writer.provider, req.writer.model) if req.writer else None,
                          helper=(req.helper.provider, req.helper.model) if req.helper else None)
    try:
        report = await agent.run()
        data = report.model_dump(mode="json")
        data["metrics"]["total_cost_usd"] = report.metrics.total_cost_usd
        store.finish_run(run_id, data, agent.trace)
        await channel.publish({"type": "done", "run_id": run_id})
    except Exception as e:
        log.exception("run %s failed", run_id)
        store.fail_run(run_id, str(e), agent.trace)
        await channel.publish({"type": "error", "message": str(e)})
    finally:
        channel.done = True
        for q in list(channel.subscribers):
            q.put_nowait(None)
        # Keep the channel briefly so reconnecting clients can replay, then free memory.
        await asyncio.sleep(120)
        channels.pop(run_id, None)


@app.post("/api/research")
async def start_research(req: ResearchRequest, request: Request, x_access_token: Optional[str] = Header(None)):
    _check_access(x_access_token)
    if not llm.configured_providers():
        raise HTTPException(503, "No LLM provider configured on the server (see .env.example).")
    await _validate_pick(req.writer, "Writer")
    await _validate_pick(req.helper, "Helper")
    _rate_limit(_client_ip(request))
    run_id = uuid.uuid4().hex[:12]
    store.create_run(run_id, req.question, req.depth)
    channel = channels[run_id] = RunChannel()
    task = asyncio.create_task(_execute(run_id, req, channel))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"run_id": run_id}


@app.get("/api/research/{run_id}/events")
async def events(run_id: str, request: Request):
    channel = channels.get(run_id)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "Unknown run")

    async def stream():
        if channel is None:  # finished long ago or server restarted: replay stored trace
            for ev in run["trace"]:
                yield f"data: {json.dumps(ev, default=str)}\n\n"
            final = {"type": "done", "run_id": run_id} if run["status"] == "done" else {
                "type": "error", "message": run.get("error") or "Run did not complete"}
            yield f"data: {json.dumps(final)}\n\n"
            return
        q: asyncio.Queue = asyncio.Queue()
        backlog = list(channel.events)
        channel.subscribers.add(q)
        try:
            for ev in backlog:
                yield f"data: {json.dumps(ev, default=str)}\n\n"
            if channel.done:
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    if await request.is_disconnected():
                        return
                    continue
                if ev is None:
                    return
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        finally:
            channel.subscribers.discard(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/research/{run_id}")
async def get_run(run_id: str):
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(404, "Unknown run")
    run.pop("trace", None)
    return run


@app.get("/api/runs")
async def runs(x_access_token: Optional[str] = Header(None)):
    _check_access(x_access_token)
    return store.list_runs()


@app.delete("/api/research/{run_id}")
async def delete_run(run_id: str, x_access_token: Optional[str] = Header(None)):
    _check_access(x_access_token)
    store.delete_run(run_id)
    return {"ok": True}


# ---------------------------------------------------------------- sharing

@app.post("/api/research/{run_id}/share")
async def share(run_id: str, request: Request, x_access_token: Optional[str] = Header(None)):
    _check_access(x_access_token)
    run = store.get_run(run_id)
    if not run or run["status"] != "done":
        raise HTTPException(400, "Only completed reports can be shared.")
    slug = store.create_share(run_id)
    return {"slug": slug, "url": f"{_base_url(request)}/r/{slug}"}


@app.delete("/api/research/{run_id}/share")
async def unshare(run_id: str, x_access_token: Optional[str] = Header(None)):
    _check_access(x_access_token)
    store.revoke_share(run_id)
    return {"ok": True}


@app.get("/api/shared/{slug}")
async def shared(slug: str):
    run_id = store.run_for_share(slug)
    run = store.get_run(run_id) if run_id else None
    if not run or not run["report"]:
        raise HTTPException(404, "This shared report does not exist or was revoked.")
    return {"question": run["question"], "created_at": run["created_at"], "report": run["report"], "slug": slug}


# ---------------------------------------------------------------- exports

EXPORTS = {
    "md": ("text/markdown; charset=utf-8", exporters.to_markdown),
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", exporters.to_docx),
    "pdf": ("application/pdf", exporters.to_pdf),
}


def _export(report: dict, fmt: str) -> Response:
    if fmt not in EXPORTS:
        raise HTTPException(400, "format must be one of md, docx, pdf")
    mime, fn = EXPORTS[fmt]
    try:
        body = fn(report)
    except Exception as e:
        log.exception("export failed")
        raise HTTPException(500, f"Export failed: {e}")
    name = f"{exporters.safe_filename(report.get('title', 'report'))}.{fmt}"
    return Response(body, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/research/{run_id}/export")
async def export_run(run_id: str, format: str = "pdf"):
    run = store.get_run(run_id)
    if not run or not run["report"]:
        raise HTTPException(404, "Report not found")
    return await asyncio.to_thread(_export, run["report"], format)


@app.get("/api/shared/{slug}/export")
async def export_shared(slug: str, format: str = "pdf"):
    run_id = store.run_for_share(slug)
    run = store.get_run(run_id) if run_id else None
    if not run or not run["report"]:
        raise HTTPException(404, "Report not found")
    return await asyncio.to_thread(_export, run["report"], format)


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
