"""Day 2 example: Academic Researcher Agent with Tools."""

import argparse
import os
import sys


def _add_src_to_path():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    sys.path.insert(0, src_path)


_add_src_to_path()

from academic_researcher.graphs.day2_tools_agent import run_day2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--goal", required=True, help="Research goal")
    parser.add_argument("--user-id", default="default", help="User session ID")
    args = parser.parse_args()

    print("=== Academic Researcher Agent with Tools (Day 2) ===")
    print(f"Topic: {args.topic}")
    print(f"Goal: {args.goal}")
    print("\nThis agent will:")
    print("1. Create a research plan")
    print("2. Search ArXiv for relevant papers")
    print("3. Write a comprehensive report")
    print("\nRunning...\n")

    report = run_day2(topic=args.topic, goal=args.goal, user_id=args.user_id)
    
    print("=== Final Research Report ===\n")
    print(report)


if __name__ == "__main__":
    main()