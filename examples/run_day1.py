import argparse
import os
import sys


def _add_src_to_path():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    sys.path.insert(0, src_path)


_add_src_to_path()

from academic_researcher.graphs.day1_basic_agent import run_day1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Research topic, e.g. 'graph neural networks for drug discovery'")
    parser.add_argument("--goal", required=True, help="What you want the report to accomplish")
    parser.add_argument("--user-id", default="default", help="Session/user id for memory (used later)")
    args = parser.parse_args()

    report = run_day1(topic=args.topic, goal=args.goal, user_id=args.user_id)
    print("\n=== Academic Researcher Agent Report (Day 1) ===\n")
    print(report)


if __name__ == "__main__":
    main()

