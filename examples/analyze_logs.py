"""
Day 4 Bonus: Parse the JSONL log file produced by the agent and print
per-node performance statistics across all recorded runs.

Usage:
    python examples/analyze_logs.py
    python examples/analyze_logs.py --log-file logs/agent_spans.jsonl
    python examples/analyze_logs.py --user-id researcher_001
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_log_entries(log_path: str) -> list:
    """Load all JSON lines from the log file."""
    entries = []
    p = Path(log_path)
    if not p.exists():
        print(f"Log file not found: {p.absolute()}")
        print("Run Day 4 at least once to generate logs.")
        sys.exit(1)

    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip malformed lines
    return entries


def filter_span_events(entries: list, user_id: str | None = None) -> list:
    """Keep only span-completion events (span.*.done or span.*.error)."""
    spans = []
    for e in entries:
        event = e.get("event", "")
        if not event.startswith("span."):
            continue
        if user_id and e.get("user_id") != user_id:
            continue
        spans.append(e)
    return spans


def compute_stats(spans: list) -> dict:
    """Group by span name and compute min / avg / max / count."""
    by_name = defaultdict(list)
    for s in spans:
        name = s.get("span", "unknown")
        elapsed = s.get("elapsed_ms", 0)
        by_name[name].append({
            "elapsed_ms": elapsed,
            "status": s.get("status", "unknown"),
            "score": s.get("score"),
            "tool_calls": s.get("tool_calls"),
        })

    stats = {}
    for name, records in sorted(by_name.items()):
        times = [r["elapsed_ms"] for r in records]
        errors = sum(1 for r in records if r["status"] == "error")
        scores = [r["score"] for r in records if r["score"] is not None]
        tool_calls = [r["tool_calls"] for r in records if r["tool_calls"] is not None]

        stats[name] = {
            "count": len(records),
            "errors": errors,
            "min_ms": round(min(times), 1) if times else 0,
            "avg_ms": round(sum(times) / len(times), 1) if times else 0,
            "max_ms": round(max(times), 1) if times else 0,
            "total_ms": round(sum(times), 1),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
            "avg_tool_calls": round(sum(tool_calls) / len(tool_calls), 1) if tool_calls else None,
        }
    return stats


def compute_run_stats(entries: list, user_id: str | None = None) -> dict:
    """Compute per-run aggregates (total time, final score)."""
    runs = []
    for e in entries:
        if e.get("event") != "agent_run_complete":
            continue
        if user_id and e.get("user_id") != user_id:
            continue
        runs.append({
            "ts": e.get("ts"),
            "user_id": e.get("user_id"),
            "topic": e.get("topic"),
            "total_spans": e.get("total_spans"),
            "total_elapsed_ms": e.get("total_elapsed_ms"),
        })
    return runs


def print_report(stats: dict, runs: list):
    print("=" * 72)
    print("  AGENT PERFORMANCE ANALYSIS (from logs/agent_spans.jsonl)")
    print("=" * 72)

    # ── Per-node table ───────────────────────────────────────────────
    print()
    print(f"  {'Node':<30} {'Count':>5} {'Err':>4} {'Min ms':>9} {'Avg ms':>9} {'Max ms':>9}")
    print(f"  {'─'*30} {'─'*5} {'─'*4} {'─'*9} {'─'*9} {'─'*9}")

    total_avg = 0.0
    for name, s in stats.items():
        extra = ""
        if s["avg_score"] is not None:
            extra += f"  avg_score={s['avg_score']}"
        if s["avg_tool_calls"] is not None:
            extra += f"  avg_tools={s['avg_tool_calls']}"
        print(
            f"  {name:<30} {s['count']:>5} {s['errors']:>4} "
            f"{s['min_ms']:>9.1f} {s['avg_ms']:>9.1f} {s['max_ms']:>9.1f}"
            f"{extra}"
        )
        total_avg += s["avg_ms"]

    print(f"  {'─'*30} {'─'*5} {'─'*4} {'─'*9} {'─'*9} {'─'*9}")
    print(f"  {'TOTAL (avg per run)':<30} {'':>5} {'':>4} {'':>9} {total_avg:>9.1f}")

    # ── Per-run summary ──────────────────────────────────────────────
    if runs:
        print()
        print("  COMPLETED RUNS")
        print(f"  {'Timestamp':<28} {'User':<18} {'Topic':<30} {'Time (s)':>8}")
        print(f"  {'─'*28} {'─'*18} {'─'*30} {'─'*8}")
        for r in runs:
            topic = (r["topic"] or "")[:28]
            secs = r.get("total_elapsed_ms", 0) / 1000
            print(
                f"  {r.get('ts',''):<28} {r.get('user_id',''):<18} "
                f"{topic:<30} {secs:>7.1f}s"
            )
        total_runs = len(runs)
        avg_run_time = sum(r.get("total_elapsed_ms", 0) for r in runs) / total_runs / 1000
        print(f"\n  Total runs: {total_runs}   Avg run time: {avg_run_time:.1f}s")

    print()
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Analyze agent span logs")
    parser.add_argument(
        "--log-file",
        default="logs/agent_spans.jsonl",
        help="Path to the JSONL log file (default: logs/agent_spans.jsonl)",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Filter to a specific user ID",
    )
    args = parser.parse_args()

    # If run from examples/, the log file is one level up
    log_path = args.log_file
    if not Path(log_path).exists():
        alt = Path("..") / log_path
        if alt.exists():
            log_path = str(alt)

    entries = load_log_entries(log_path)
    spans = filter_span_events(entries, user_id=args.user_id)
    stats = compute_stats(spans)
    runs = compute_run_stats(entries, user_id=args.user_id)

    if not stats:
        print("No span data found in logs. Run Day 4 at least once first.")
        sys.exit(0)

    print_report(stats, runs)


if __name__ == "__main__":
    main()
