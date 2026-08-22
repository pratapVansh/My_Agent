"""
Upload a new timetable PDF and make it the active timetable.

Talks directly to the database — no server needs to be running and no login
is required. Deterministic only: no model call is made anywhere in this path
(see app/tools/timetable_pdf_parser.py). The PDF needs a text layer with one
class per line — a weekday and a time range on the same line, e.g.
"Monday 09:00 - 10:00 Data Structures Room: LT1 Instructor: Dr. Sharma". A
scanned/image PDF or a genuine grid-style table won't parse automatically;
see the note this script prints if that happens.

Usage:
    python scripts/upload_new_timetable.py path/to/new_timetable.pdf --user vansh
    python scripts/upload_new_timetable.py path/to/new_timetable.pdf --user vansh --semester 8
    python scripts/upload_new_timetable.py path/to/new_timetable.pdf --user vansh --dry-run
    python scripts/upload_new_timetable.py path/to/new_timetable.pdf --user vansh --force

The new timetable atomically replaces whatever is currently active — old
classes are retired, not mixed in — and uploading the exact same file twice
is a no-op unless --force is passed.
"""
import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import engine  # noqa: E402
from app.domain.schedule import schedule_repository  # noqa: E402
from app.tools.timetable_pdf_parser import timetable_pdf_parser  # noqa: E402
from app.tools.timetable_tool import validate_entries  # noqa: E402


async def main(args) -> int:
    pdf = Path(args.pdf).expanduser()
    if not pdf.exists():
        print(f"No such file: {pdf}")
        return 1

    pdf_bytes = pdf.read_bytes()
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    active = await schedule_repository.active_upload(args.user)
    if active and active["content_hash"] == content_hash and not args.force:
        print(
            f"'{pdf.name}' is already your active timetable "
            f"(uploaded {active['created_at']}). Nothing to do.\n"
            "Pass --force to reload it anyway."
        )
        return 0

    print(f"Reading {pdf.name} ...")
    parse_result = await timetable_pdf_parser.parse(pdf_bytes, filename=pdf.name)

    if not parse_result["success"]:
        print("\nCould not read this as a timetable.")
        print(f"Reason: {parse_result['parse_notes']}")
        if parse_result["raw_text_preview"]:
            print(f"\nFirst text extracted from the PDF:\n{parse_result['raw_text_preview']}")
        print(
            "\nIf this is a scanned image or a grid-style table PDF (rows/columns, "
            "not one line of text per class), it needs a manual verified "
            "transcription instead — see app/tools/timetable_source.py for how the "
            "current one was done, and scripts/load_timetable.py to load it."
        )
        return 1

    print(f"Parsed {parse_result['entry_count']} class(es) from {parse_result['pages']} page(s):")
    for e in parse_result["entries"]:
        extras = "  ".join(x for x in (e.location, e.instructor) if x)
        print(f"  day={e.day_of_week}  {e.start_time}-{e.end_time}  {e.subject}"
              + (f"  {extras}" if extras else ""))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    valid, skipped = validate_entries(parse_result["entries"])
    result = await schedule_repository.replace_active_timetable(
        args.user,
        filename=pdf.name,
        content_hash=content_hash,
        valid_entries=valid,
        skipped=skipped,
        semester=args.semester,
        page_count=parse_result["pages"],
        parse_notes=parse_result["parse_notes"],
        ingest_method="pdf_parser",
    )

    if not result["success"]:
        print(f"\nNothing written — no entries passed validation: {result['skipped']}")
        return 1

    print(
        f"\nRetired {result['deactivated_rows']} old class(es) / "
        f"{result['deactivated_uploads']} old upload(s)."
    )
    print(f"Stored  {result['stored_count']} new class(es). Upload id: {result['upload_id']}")
    if result["skipped_count"]:
        print(f"Skipped {result['skipped_count']}: {result['skipped']}")

    print("\nDone. Try asking the assistant:")
    print('  "What classes do I have today?"')
    print('  "What is my next class?"')
    print('  "Who is my professor for <subject>?"')
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", help="Path to the new timetable PDF")
    parser.add_argument("--user", required=True, help="Your user id (e.g. vansh)")
    parser.add_argument(
        "--semester", default=None, help="Semester label, e.g. '8' — not read from the PDF"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reload even if this exact file is already the active timetable",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse and print only — write nothing"
    )
    args = parser.parse_args()
    try:
        code = asyncio.run(main(args))
    finally:
        asyncio.run(engine.dispose())
    sys.exit(code)
