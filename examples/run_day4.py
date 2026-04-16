"""Day 4 example: Agent with Quality Evaluation and Observability."""

import argparse
import asyncio
import os
import sys


def _add_src_to_path():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(repo_root, "src"))


_add_src_to_path()

from academic_researcher.graphs.day4_quality_agent import run_day4_async


async def main_async(args):
    print("=== Academic Researcher Agent with Quality Evaluation (Day 4) ===")
    print(f"User ID : {args.user_id}")
    print(f"Topic   : {args.topic}")
    print(f"Goal    : {args.goal}")
    print()
    print("This agent will:")
    print("  1. Load context from memory")
    print("  2. Build a research plan")
    print("  3. Search ArXiv for relevant papers")
    print("  4. Write a research report")
    print("  5. Evaluate the report against a 6-criterion rubric")
    print("  6. Automatically improve if quality < 60 / 100 (up to 2 rounds)")
    print("  7. Save everything to memory")
    print("\nRunning ...\n")

    result = await run_day4_async(
        topic=args.topic,
        goal=args.goal,
        user_id=args.user_id,
    )

    # ── Report ────────────────────────────────────────────────────────
    print("=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)
    print(result["report"])

    # ── Evaluation ────────────────────────────────────────────────────
    ev = result.get("evaluation")
    if ev:
        print("\n" + "=" * 60)
        print(ev.summary())

    # ── Trace (optional) ─────────────────────────────────────────────
    if args.show_trace:
        trace = result.get("trace", {})
        print("\n" + "=" * 60)
        print("EXECUTION TRACE")
        print("=" * 60)
        print(f"Total spans   : {trace.get('total_spans', 0)}")
        print(f"Total time    : {trace.get('total_elapsed_ms', 0):.0f} ms")
        print()
        for span in trace.get("spans", []):
            status_icon = "✓" if span.get("status") == "ok" else "✗"
            print(
                f"  {status_icon} {span['span']:<35} "
                f"{span.get('elapsed_ms', 0):>6.0f} ms"
                + (f"  score={span['score']}" if "score" in span else "")
                + (f"  calls={span['tool_calls']}" if "tool_calls" in span else "")
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--user-id", default="researcher_001")
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print execution trace with timing per node",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
