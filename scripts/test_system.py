"""
Comprehensive system test for RAG backend.
Tests all components: Groq, Cohere, Qdrant, PostgreSQL, mem0
"""
import asyncio
import sys

async def test_system():
    print("=" * 70)
    print("🧪 COMPREHENSIVE SYSTEM TEST")
    print("=" * 70)

    results = {
        "groq": False,
        "cohere": False,
        "qdrant": False,
        "postgres": False,
        "memory": False,
        "chunking": False
    }

    # Test 1: Groq LLM
    print("\n[1/6] Testing Groq LLM...")
    try:
        from app.services.groq_service import groq_service
        results["groq"] = await groq_service.health_check()
        print(f"  {'✓' if results['groq'] else '✗'} Groq: {'OK' if results['groq'] else 'FAILED'}")
    except Exception as e:
        print(f"  ✗ Groq: {str(e)}")

    # Test 2: Cohere Embeddings
    print("\n[2/6] Testing Cohere Embeddings...")
    try:
        from app.services.cohere_service import cohere_service
        results["cohere"] = await cohere_service.health_check()
        if results["cohere"]:
            vec = await cohere_service.embed_text("test", "search_document")
            print(f"  ✓ Cohere: OK (dimension: {len(vec)})")
        else:
            print("  ✗ Cohere: FAILED")
    except Exception as e:
        print(f"  ✗ Cohere: {str(e)}")

    # Test 3: Qdrant Vector DB
    print("\n[3/6] Testing Qdrant Vector Database...")
    try:
        from app.services.qdrant_service import qdrant_service
        results["qdrant"] = await qdrant_service.health_check()
        print(f"  {'✓' if results['qdrant'] else '✗'} Qdrant: {'OK' if results['qdrant'] else 'FAILED'}")
    except Exception as e:
        print(f"  ✗ Qdrant: {str(e)}")

    # Test 4: PostgreSQL
    print("\n[4/6] Testing PostgreSQL Database...")
    try:
        from app.memory.short_term_memory import short_term_memory
        await short_term_memory.init_db()
        results["postgres"] = True
        print("  ✓ PostgreSQL: OK")
    except Exception as e:
        print(f"  ✗ PostgreSQL: {str(e)}")

    # Test 5: Chunking Service
    print("\n[5/6] Testing Text Chunking...")
    try:
        from app.services.chunking_service import chunking_service
        text = "This is a test. " * 100
        chunks = chunking_service.chunk_text(text, {"test": "metadata"})
        if chunks:
            results["chunking"] = True
            print(f"  ✓ Chunking: OK (created {len(chunks)} chunks)")
        else:
            print("  ✗ Chunking: No chunks created")
    except Exception as e:
        print(f"  ✗ Chunking: {str(e)}")

    # Test 6: Full Memory System
    print("\n[6/6] Testing Full Memory System (Qdrant + Cohere + Chunking)...")
    try:
        from app.memory.long_term_memory_qdrant import long_term_memory_qdrant
        await long_term_memory_qdrant.initialize()

        # Store test data
        doc_id = await long_term_memory_qdrant.store_resume(
            user_id="system_test",
            resume_text="Test resume content for verification",
            metadata={"test": True}
        )

        # Retrieve it
        result = await long_term_memory_qdrant.retrieve_resume("system_test")

        if result and "content" in result:
            results["memory"] = True
            print(f"  ✓ Memory System: OK (stored and retrieved)")
        else:
            print("  ✗ Memory System: Failed to retrieve")
    except Exception as e:
        print(f"  ✗ Memory System: {str(e)}")

    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {component.upper()}: {'PASSED' if status else 'FAILED'}")

    print("\n" + "=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL! Your RAG backend is ready!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} component(s) failed. Check configuration.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(test_system())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
