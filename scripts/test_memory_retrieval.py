"""
Test memory retrieval from Qdrant.
Verify that uploaded profile data can be retrieved.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory.memory_manager import memory_manager  # noqa: E402


async def test_retrieval():
    """Test retrieving data from all memory systems."""
    print("=" * 60)
    print("MEMORY RETRIEVAL TEST")
    print("=" * 60)

    # Initialize memory systems
    print("\n[1/4] Initializing memory systems...")
    await memory_manager.initialize()
    print("   [OK] Memory systems initialized")

    # Test user ID (same as in upload script)
    user_id = "Vansh"
    session_id = "test_session"

    # Test queries
    test_queries = [
        "Tell me about my skills",
        "What is my name",
        "What projects have I worked on",
        "What is my education"
    ]

    for i, query in enumerate(test_queries, start=2):
        print(f"\n[{i}/4] Testing query: '{query}'")
        print("-" * 60)

        # Retrieve context
        context = await memory_manager.retrieve_context(
            user_id=user_id,
            session_id=session_id,
            query=query
        )

        # Display results
        print(f"\n   Chat History: {len(context.get('chat_history', []))} messages")
        print(f"   Preferences: {len(context.get('preferences', []))} items")

        long_term = context.get('long_term', {})
        print(f"   Long-term memory:")

        # Resume
        resume = long_term.get('resume', {})
        if resume and resume.get('content'):
            print(f"     - Resume: Found ({len(resume['content'])} chars)")
            print(f"       Preview: {resume['content'][:150]}...")
        else:
            print(f"     - Resume: NOT FOUND")

        # Skills
        skills = long_term.get('skills', [])
        if skills:
            print(f"     - Skills: Found {len(skills)} items")
            for skill in skills[:3]:
                print(f"       * {skill.get('content', '')[:100]}")
        else:
            print(f"     - Skills: NOT FOUND")

        # Projects
        projects = long_term.get('projects', [])
        if projects:
            print(f"     - Projects: Found {len(projects)} items")
            for project in projects[:2]:
                print(f"       * {project.get('content', '')[:100]}")
        else:
            print(f"     - Projects: NOT FOUND")

        # Format for prompt
        formatted = memory_manager.format_context_for_prompt(context)
        if formatted:
            print(f"\n   Formatted prompt context ({len(formatted)} chars):")
            print(f"   {formatted[:300]}...")
        else:
            print(f"\n   Formatted prompt: EMPTY")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_retrieval())
