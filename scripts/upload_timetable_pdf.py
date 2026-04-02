"""
Upload a timetable PDF to the academic agent's schedule.

Usage:
  python scripts/upload_timetable_pdf.py path/to/timetable.pdf --user-id vansh
  python scripts/upload_timetable_pdf.py path/to/timetable.pdf --user-id vansh --keep-existing

By default, any previously stored timetable is replaced by the new one.
Pass --keep-existing to append instead of replace.
"""
import asyncio
import sys
import argparse
from pathlib import Path

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
    clear_existing: bool = True,
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
    print(f"  Mode   : {'replace existing timetable' if clear_existing else 'append to existing timetable'}")
    print()

    endpoint = f"{api_base}/api/v1/agents/tools/timetable/upload-pdf"

    print("Uploading and parsing...")

    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(path, "rb") as f:
            response = await client.post(
                endpoint,
                data={
                    "user_id": user_id,
                    "clear_existing": str(clear_existing).lower(),
                },
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

    print_separator()
    print("  TIMETABLE UPLOADED SUCCESSFULLY")
    print_separator()
    print(f"  File              : {result.get('filename')}")
    print(f"  Pages in PDF      : {result.get('pages_in_pdf')}")
    print(f"  Entries parsed    : {result.get('entries_parsed')}")
    print(f"  Entries stored    : {result.get('entries_stored')}")
    if result.get("old_entries_cleared"):
        print(f"  Old entries cleared: {result.get('old_entries_cleared')}")
    if result.get("parse_notes"):
        print(f"  Notes             : {result.get('parse_notes')}")
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
  Upload and replace existing timetable (default):
    python scripts/upload_timetable_pdf.py sem5_timetable.pdf --user-id vansh

  Upload and keep existing entries too:
    python scripts/upload_timetable_pdf.py sem5_timetable.pdf --user-id vansh --keep-existing
        """,
    )
    parser.add_argument("pdf_path", help="Path to your timetable PDF file")
    parser.add_argument("--user-id", required=True, help="Your user ID (e.g. vansh)")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        default=False,
        help="Append new entries instead of replacing the existing timetable",
    )
    args = parser.parse_args()

    api_base = get_api_base()
    print(f"  API base: {api_base}")

    await upload_timetable(
        pdf_path=args.pdf_path,
        user_id=args.user_id.strip().lower(),
        clear_existing=not args.keep_existing,
        api_base=api_base,
    )


if __name__ == "__main__":
    asyncio.run(main())
