"""
Smoke test for resume pipeline end-to-end validation.
Tests: upload -> name retrieval -> skills retrieval -> projects retrieval

Usage:
    python scripts/smoke_test_pipeline.py --user-id vansh
"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.memory.memory_manager import memory_manager  # noqa: E402


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


async def run_smoke_test(user_id: str):
    print("=" * 60)
    print("RESUME PIPELINE SMOKE TEST")
    print("=" * 60)
    print(f"User ID: {user_id}\n")

    await memory_manager.initialize()
    lt = memory_manager.long_term
    all_passed = True

    # --- Test 1: Resume retrieval ---
    print("Test 1: Resume retrieval")
    resume = await lt.retrieve_resume(user_id)
    if resume and resume.get("content"):
        preview = resume["content"][:120].replace("\n", " ")
        print(f"  {PASS} Resume found. Preview: {preview}...")
    else:
        print(f"  {FAIL} No resume found for user_id='{user_id}'")
        print("        Run: python scripts/upload_pdf.py <your_resume.pdf> --user-id " + user_id)
        all_passed = False

    # --- Test 2: Name detection ---
    print("\nTest 2: Name detection from resume")
    if resume and resume.get("content"):
        lines = [ln.strip() for ln in resume["content"].split("\n") if ln.strip()]
        detected_name = lt._detect_name(lines)
        if detected_name:
            print(f"  {PASS} Name detected: '{detected_name}'")
        else:
            print(f"  {WARN} Name not auto-detected from top lines (resume may have unusual formatting)")
    else:
        print(f"  {WARN} Skipped (no resume)")

    # --- Test 3: Skills retrieval ---
    print("\nTest 3: Skills retrieval")
    skills = await lt.retrieve_skills(user_id, query="technical skills programming")
    if not skills:
        # Try fallback
        fallback = await lt._fallback_resume_search(user_id, "technical skills programming", "skills", 5)
        if fallback:
            print(f"  {WARN} skills_chunks empty — fallback returned {len(fallback)} result(s) from resume_chunks")
            print(f"        First result: {fallback[0]['content'][:100]}...")
        else:
            print(f"  {FAIL} No skills found in skills_chunks OR resume_chunks")
            print("        Re-upload resume; ensure it has a 'Skills' or 'Technical Skills' section")
            all_passed = False
    else:
        print(f"  {PASS} Found {len(skills)} skill result(s) in skills_chunks")
        print(f"        First result: {skills[0]['content'][:100]}...")

    # --- Test 4: Projects retrieval ---
    print("\nTest 4: Projects retrieval")
    projects = await lt.retrieve_projects(user_id, query="projects built developed")
    if not projects:
        # Try fallback
        fallback = await lt._fallback_resume_search(user_id, "projects built developed", "projects", 5)
        if fallback:
            print(f"  {WARN} projects_chunks empty — fallback returned {len(fallback)} result(s) from resume_chunks")
            print(f"        First result: {fallback[0]['content'][:100]}...")
        else:
            print(f"  {FAIL} No projects found in projects_chunks OR resume_chunks")
            print("        Re-upload resume; ensure it has a 'Projects' section")
            all_passed = False
    else:
        print(f"  {PASS} Found {len(projects)} project result(s) in projects_chunks")
        print(f"        First result: {projects[0]['content'][:100]}...")

    # --- Test 5: Semantic extraction preview ---
    print("\nTest 5: Semantic extraction from resume text")
    if resume and resume.get("content"):
        chunks = lt._extract_semantic_resume_chunks(resume["content"])
        type_counts = {}
        for c in chunks:
            t = c.get("type", "other")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  {PASS} Extracted {len(chunks)} semantic chunks: {type_counts}")
        missing = [t for t in ["skills", "projects", "experience", "education"] if t not in type_counts]
        if missing:
            print(f"  {WARN} Missing sections: {missing}")
            print("        These section headers may not match expected headings in your resume")
    else:
        print(f"  {WARN} Skipped (no resume)")

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_passed:
        print("[ALL TESTS PASSED] Pipeline is healthy.")
    else:
        print("[SOME TESTS FAILED] Fix the issues above before Phase 1.")
    print("=" * 60)

    await memory_manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test resume pipeline")
    parser.add_argument("--user-id", required=True, help="User ID to test (e.g. vansh)")
    args = parser.parse_args()

    asyncio.run(run_smoke_test(args.user_id.strip().lower()))
