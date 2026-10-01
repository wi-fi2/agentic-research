"""MCP server exposing the research agent as tools for Claude Desktop, Claude Code, Cursor, etc.

Run over stdio:  python mcp_server.py
Tools: research, list_runs, get_report. Runs are saved to the same SQLite store as the web app.
"""

import logging
import sys
import uuid

from mcp.server.fastmcp import FastMCP

from app import laya_judge, llm, store
from app.agent import ResearchAgent

logging.basicConfig(level=logging.INFO, stream=sys.stderr)  # stdout is the MCP transport

mcp = FastMCP("agentic-research")
_ready = False


def _init() -> None:
    global _ready
    if not _ready:
        store.conn()
        store.mark_interrupted_runs()
        _ready = True


@mcp.tool()
async def research(question: str, depth: str = "standard") -> str:
    """Run a research task and return a cited, fact-checked Markdown brief.

    Every factual sentence carries a quote that code verified against its source. Can take minutes.
    depth: "quick", "standard" or "deep".
    """
    if depth not in ("quick", "standard", "deep"):
        raise ValueError("depth must be quick, standard or deep")
    if not llm.configured_providers():
        raise RuntimeError("No LLM provider configured (see .env.example).")
    _init()
    laya_judge.start_loading()  # idempotent background load of the local decision model
    run_id = uuid.uuid4().hex[:12]
    store.create_run(run_id, question, depth)
    agent = ResearchAgent(question, depth, run_id=run_id)
    try:
        report = await agent.run()
    except Exception as e:
        store.fail_run(run_id, str(e), agent.trace)
        raise
    data = report.model_dump(mode="json")
    data["metrics"]["total_cost_usd"] = report.metrics.total_cost_usd
    store.finish_run(run_id, data, agent.trace)
    return f"{report.markdown}\n\n(run_id: {run_id})"


@mcp.tool()
def list_runs(limit: int = 20) -> list[dict]:
    """List recent research runs (id, question, depth, status, created_at)."""
    _init()
    return store.list_runs(max(1, min(limit, 100)))


@mcp.tool()
def get_report(run_id: str) -> str:
    """Fetch the Markdown brief of a finished run by its run_id."""
    _init()
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"No run with id {run_id}")
    if run["status"] != "done":
        return f"Run {run_id} is {run['status']}." + (f" Error: {run['error']}" if run.get("error") else "")
    return run["report"]["markdown"]


if __name__ == "__main__":
    mcp.run()
