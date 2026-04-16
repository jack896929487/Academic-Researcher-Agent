"""Day 3 example: Academic Researcher Agent with Memory."""

import argparse
import asyncio
import os
import sys


def _add_src_to_path():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, "src")
    sys.path.insert(0, src_path)


_add_src_to_path()

from academic_researcher.graphs.day3_memory_agent import run_day3_async, MemoryAwareAgent


async def set_user_preferences(user_id: str):
    """Set some example user preferences."""
    agent = MemoryAwareAgent()
    
    # Set research domain preference
    await agent.session_manager.store_user_preference(
        user_id=user_id,
        session_id="setup",
        preference_type="research_domain",
        preference_value="machine learning and artificial intelligence"
    )
    
    # Set report style preference
    await agent.session_manager.store_user_preference(
        user_id=user_id,
        session_id="setup",
        preference_type="report_style",
        preference_value="detailed with methodology focus and practical applications"
    )
    
    print(f"✓ Set user preferences for {user_id}")


async def show_user_stats(user_id: str):
    """Show user's memory statistics."""
    agent = MemoryAwareAgent()
    
    # Get user stats
    stats = agent.memory.get_user_stats(user_id)
    print(f"\n=== Memory Statistics for {user_id} ===")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Total sessions: {stats['total_sessions']}")
    print("Memory types:")
    for mem_type, count in stats['memory_types'].items():
        print(f"  - {mem_type}: {count}")
    
    # Get research history
    history = await agent.session_manager.get_research_history(user_id, limit=5)
    if history:
        print(f"\n=== Recent Research History ===")
        for item in history:
            print(f"- {item['topic']} (Goal: {item['goal']})")
            print(f"  Session: {item['session_id']}, Date: {item['created_at'][:19]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument("--goal", required=True, help="Research goal")
    parser.add_argument("--user-id", default="researcher_001", help="User ID for memory")
    parser.add_argument("--setup-preferences", action="store_true", 
                       help="Set up example user preferences first")
    parser.add_argument("--show-stats", action="store_true",
                       help="Show user memory statistics after research")
    args = parser.parse_args()

    async def run_async():
        print("=== Academic Researcher Agent with Memory (Day 3) ===")
        print(f"User ID: {args.user_id}")
        print(f"Topic: {args.topic}")
        print(f"Goal: {args.goal}")
        
        # Set up preferences if requested
        if args.setup_preferences:
            print("\n=== Setting up user preferences ===")
            await set_user_preferences(args.user_id)
        
        print("\nThis agent will:")
        print("1. Load context from your previous research")
        print("2. Create a personalized research plan")
        print("3. Search ArXiv for relevant papers")
        print("4. Write a report considering your preferences")
        print("5. Save the session to memory for future use")
        print("\nRunning...\n")

        # Run the research
        report = await run_day3_async(topic=args.topic, goal=args.goal, user_id=args.user_id)
        
        print("=== Final Research Report ===\n")
        print(report)
        
        # Show stats if requested
        if args.show_stats:
            await show_user_stats(args.user_id)

    # Run the async function
    asyncio.run(run_async())


if __name__ == "__main__":
    main()