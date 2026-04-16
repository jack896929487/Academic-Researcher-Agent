"""Day 5 example: Multi-Agent Orchestrator + FastAPI deployment."""

import argparse
import asyncio
import os
import sys


def _add_src_to_path():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(repo_root, "src"))


_add_src_to_path()

from academic_researcher.agents.multi_agent_graph import run_day5_async


async def main_async(args):
    print("=" * 64)
    print("  MULTI-AGENT ACADEMIC RESEARCHER (Day 5)")
    print("=" * 64)
    print(f"  User  : {args.user_id}")
    print(f"  Topic : {args.topic}")
    print(f"  Goal  : {args.goal}")
    print(f"  Domain: {args.domain or 'auto'}")
    print()
    print("  Agent Pipeline:")
    print("    Planner → Researcher → Writer → Reviewer")
    print("    (if quality < 60 → Planner revises, up to 1 round)")
    print()
    print("  Running ...\n")

    result = await run_day5_async(
        topic=args.topic,
        goal=args.goal,
        user_id=args.user_id,
        domain=args.domain,
    )

    # ── Report ────────────────────────────────────────────────────
    print("=" * 64)
    print("  FINAL REPORT")
    print("=" * 64)
    print(result["report"])

    # ── Evaluation ────────────────────────────────────────────────
    ev = result.get("evaluation")
    if ev:
        print("\n" + "=" * 64)
        print(ev.summary())
    elif result.get("evaluation_error"):
        print("\n" + "=" * 64)
        print("  EVALUATION UNAVAILABLE")
        print("=" * 64)
        print(result["evaluation_error"])

    # ── A2A log ───────────────────────────────────────────────────
    if args.show_a2a:
        a2a = result.get("a2a_log", [])
        print("\n" + "=" * 64)
        print(f"  A2A MESSAGE LOG ({len(a2a)} messages)")
        print("=" * 64)
        for m in a2a:
            arrow = "→"
            print(
                f"  [{m.get('intent','?'):>10}] "
                f"{m.get('sender','?'):<14} {arrow} {m.get('receiver','?'):<14} "
                f"id={m.get('id','')[:8]}…"
            )

    # ── Trace ─────────────────────────────────────────────────────
    if args.show_trace:
        trace = result.get("trace", {})
        print("\n" + "=" * 64)
        print("  EXECUTION TRACE")
        print("=" * 64)
        print(f"  Total spans: {trace.get('total_spans', 0)}")
        print(f"  Total time : {trace.get('total_elapsed_ms', 0):.0f} ms")
        print()
        for span in trace.get("spans", []):
            icon = "✓" if span.get("status") == "ok" else "✗"
            extra = ""
            if "score" in span:
                extra += f"  score={span['score']}"
            if "tool_calls" in span:
                extra += f"  tools={span['tool_calls']}"
            print(
                f"  {icon} {span['span']:<30} "
                f"{span.get('elapsed_ms', 0):>8.0f} ms{extra}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--user-id", default="researcher_001")
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional domain skill, e.g. ai_algorithms or biomedicine",
    )
    parser.add_argument("--show-a2a", action="store_true", help="Show A2A message log")
    parser.add_argument("--show-trace", action="store_true", help="Show execution trace")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
