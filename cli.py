"""Command-line entry point: python cli.py "your research question" [--depth quick|standard|deep] [--writer PROVIDER:MODEL] [--helper PROVIDER:MODEL]"""

import argparse
import asyncio
import json
import sys

from app.agent import ResearchAgent


async def main() -> None:
    ap = argparse.ArgumentParser(description="Run one research task and print the Markdown report.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--depth", default="standard", choices=["quick", "standard", "deep"])
    ap.add_argument("--json", action="store_true", help="print the full report JSON instead of Markdown")
    ap.add_argument("--writer", metavar="PROVIDER:MODEL", help="model for writing the brief, e.g. groq:openai/gpt-oss-120b")
    ap.add_argument("--helper", metavar="PROVIDER:MODEL", help="model for planning and claim checks")
    args = ap.parse_args()

    async def progress(ev: dict) -> None:
        if ev["type"] == "stage":
            print(f"[{ev['t']:6.1f}s] {ev['stage']:<12} {ev['status']:<8} {ev.get('detail', '')}", file=sys.stderr)
        elif ev["type"] == "log":
            print(f"[{ev['t']:6.1f}s] {ev['message']}", file=sys.stderr)

    def pick(value):
        if not value:
            return None
        provider, sep, model = value.partition(":")
        if not sep or not model:
            ap.error(f"expected PROVIDER:MODEL, got {value!r}")
        return provider, model

    report = await ResearchAgent(" ".join(args.question), args.depth, emit=progress,
                                 writer=pick(args.writer), helper=pick(args.helper)).run()
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), indent=2))
    else:
        print(report.markdown)
    m = report.metrics
    print(f"\n[cost ${m.total_cost_usd:.4f} | {m.llm_calls} LLM calls | {m.judge_calls} Laya batches ({m.laya_cache_hits} cached), {m.claims_escalated} claims escalated, quotes {m.quotes_exact}/{m.quotes_fuzzy}/{m.quotes_missing} exact/fuzzy/missing | {m.duration_seconds}s]",
          file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
