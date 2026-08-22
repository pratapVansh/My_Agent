"""
Upload a timetable PDF to the academic agent's schedule.

Usage:
  python scripts/upload_timetable_pdf.py path/to/timetable.pdf --user-id vansh
  python scripts/upload_timetable_pdf.py path/to/timetable.pdf --user-id vansh --semester 7
  python scripts/upload_timetable_pdf.py path/to/timetable.pdf --user-id vansh --force

Always replaces: the previous timetable is atomically retired and the new one
activated in its place — see `/api/v1/agents/tools/timetable/upload-pdf`.
Uploading the exact same file twice is a no-op unless --force is passed.
"""
import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional

import httpx


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_api_base() -> str:
    """Read API base URL from frontend/.env.local or fall back to localhost."""
    env_path = Path(__file__).parent.parent / "frontend" / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("NEXT_PUBLIC_API_BASE"):
                _, _, value = line.partition("=")
                return value.strip().rstrip("/")
    return "http://localhost:8000"


def print_separator(char: str = "=", width: int = 60) -> None:
    print(char * width)


# ── Main upload function ─────────────────────────────────────────────────────

async def upload_timetable(
    pdf_path: str,
    user_id: str,
    semester: Optional[str] = None,
    force: bool = False,
    api_base: str = "http://localhost:8000",
) -> None:
    path = Path(pdf_path)

    if not path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    if not path.suffix.lower() == ".pdf":
        print(f"[ERROR] Only PDF files are supported. Got: {path.suffix}")
        sys.exit(1)

    file_size_kb = path.stat().st_size / 1024
    print(f"\n  File   : {path.name}")
    print(f"  Size   : {file_size_kb:.1f} KB")
    print(f"  User   : {user_id}")
    if semester:
        print(f"  Semester: {semester}")
    print()

    endpoint = f"{api_base}/api/v1/agents/tools/timetable/upload-pdf"

    print("Uploading and parsing (no model call is made)...")

    data = {"user_id": user_id, "force": str(force).lower()}
    if semester:
        data["semester"] = semester

    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(path, "rb") as f:
            response = await client.post(
                endpoint,
                data=data,
                files={"file": (path.name, f, "application/pdf")},
            )

    if response.status_code != 200:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        print(f"\n[ERROR] Upload failed (HTTP {response.status_code})")
        print(f"  Detail: {detail}")
        sys.exit(1)

    result = response.json()

    if not result.get("changed"):
        print_separator()
        print("  NO CHANGE — this file is already your active timetable")
        print_separator()
        print(f"  {result.get('message')}")
        print_separator()
        return

    print_separator()
    print("  TIMETABLE UPLOADED SUCCESSFULLY")
    print_separator()
    print(f"  File               : {result.get('filename')}")
    print(f"  Upload id          : {result.get('upload_id')}")
    print(f"  Pages in PDF       : {result.get('pages_in_pdf')}")
    print(f"  Entries parsed     : {result.get('entries_parsed')}")
    print(f"  Entries stored     : {result.get('entries_stored')}")
    if result.get("entries_skipped"):
        print(f"  Entries skipped    : {result.get('entries_skipped')} — {result.get('skipped')}")
    if result.get("old_entries_retired"):
        print(f"  Old entries retired: {result.get('old_entries_retired')}")
    if result.get("parse_notes"):
        print(f"  Notes              : {result.get('parse_notes')}")
    print()
    print("  Your timetable is ready. Try asking:")
    print('    "What classes do I have today?"')
    print('    "Show me my full schedule"')
    print('    "Which subjects need more attendance?"')
    print_separator()


# ── Entry point ──────────────────────────────────────────────────────────────

async def main() -> None:
    print_separator()
    print("  TIMETABLE PDF UPLOAD TOOL")
    print_separator()

    parser = argparse.ArgumentParser(
        description="Upload a semester timetable PDF to the academic agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Upload and activate as the current timetable:
    python scripts/upload_timetable_pdf.py sem7_timetable.pdf --user-id vansh --semester 7

  Reload the same file again even though nothing changed:
    python scripts/upload_timetable_pdf.py sem7_timetable.pdf --user-id vansh --force
        """,
    )
    parser.add_argument("pdf_path", help="Path to your timetable PDF file")
    parser.add_argument("--user-id", required=True, help="Your user ID (e.g. vansh)")
    parser.add_argument(
        "--semester", default=None, help="Semester label, e.g. '7' — not read from the PDF"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Reload even if this exact file is already the active timetable",
    )
    args = parser.parse_args()

    api_base = get_api_base()
    print(f"  API base: {api_base}")

    await upload_timetable(
        pdf_path=args.pdf_path,
        user_id=args.user_id.strip().lower(),
        semester=args.semester,
        force=args.force,
        api_base=api_base,
    )


if __name__ == "__main__":
    asyncio.run(main())
