"""
Load the 7th-semester timetable into the store, from the verified transcription.

`Time_Table_IT.pdf` has no text layer — PyPDF2 extracts zero characters, so
`timetable_pdf_parser` correctly reports that it cannot read it and the upload
endpoint returns 422. The grid was therefore read from the document and written
down in `app/tools/timetable_source.py`, which is what this script loads.

The hash check is the point. This will refuse to run against a PDF that is not
byte-for-byte the one the transcription was made from, so a new or edited
timetable cannot quietly inherit the old transcription — it fails loudly and
asks for a new one.

Usage:
    python scripts/load_timetable.py --user you@example.com --pdf /path/Time_Table_IT.pdf
    python scripts/load_timetable.py --user you@example.com --pdf ... --dry-run
    python scripts/load_timetable.py --user you@example.com --pdf ... --semester 7
"""
import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import engine  # noqa: E402
from app.domain.schedule import schedule_repository  # noqa: E402
from app.tools import timetable_source  # noqa: E402
from app.tools.timetable_tool import TimetableInput, validate_entries  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_entries() -> list:
    """The transcription, as storable rows. Pure — safe to run in a dry run."""
    return [
        TimetableInput(
            day_of_week=entry.day_of_week,
            start_time=entry.start_time,
            end_time=entry.end_time,
            # The full name is what the user hears; the code lives in metadata
            # so "when is GAI" still matches.
            subject=entry.subject_name,
            location=entry.room,
            instructor=entry.faculty,
            metadata=entry.as_metadata(),
        )
        for entry in timetable_source.entries()
    ]


async def main(args) -> int:
    pdf = Path(args.pdf).expanduser()
    if not pdf.exists():
        print(f"No such file: {pdf}")
        return 1

    digest = _hash(pdf)
    if digest != timetable_source.SOURCE_PDF_SHA256:
        print(
            "REFUSING TO LOAD: this PDF is not the document the stored\n"
            "transcription was read from.\n\n"
            f"  file   : {digest}\n"
            f"  expected: {timetable_source.SOURCE_PDF_SHA256}\n\n"
            "A different timetable needs a new transcription in\n"
            "app/tools/timetable_source.py — reusing this one would file\n"
            "last semester's classes under this semester's document."
        )
        return 1

    entries = build_entries()
    print(f"Source     : {timetable_source.SOURCE_TITLE}")
    print(f"File       : {pdf.name}  ({digest[:12]}…)")
    print(f"Entries    : {len(entries)} classes "
          f"(lunch column excluded — it spells LUNCH vertically)")
    print(f"Semester   : {args.semester or '(not stated in the document)'}")
    print()
    for e in timetable_source.entries():
        room = f"  {e.room}" if e.room else ""
        note = f"  [{e.note}]" if e.note else ""
        print(f"  {timetable_source.WEEKDAYS[e.day_of_week]:<10} "
              f"{e.start_time}-{e.end_time}  {e.subject_code:<5} "
              f"{e.subject_name:<40}{room}{note}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    existing = await schedule_repository.find_by_hash(args.user, digest)
    if existing and not args.force:
        print(
            f"\nThis file was already loaded on {existing['created_at']} "
            f"(upload {existing['id'][:8]}). Use --force to load it again."
        )
        return 0

    valid, skipped = validate_entries(entries)

    # Retire the previous timetable and activate this one in a single
    # transaction — see `ScheduleRepository.replace_active_timetable`. No
    # window exists where the old rows are gone and the new ones aren't in
    # yet, and a failure partway through leaves the previous timetable
    # untouched rather than half-replaced.
    result = await schedule_repository.replace_active_timetable(
        args.user,
        filename=pdf.name,
        content_hash=digest,
        valid_entries=valid,
        skipped=skipped,
        source_title=timetable_source.SOURCE_TITLE,
        semester=args.semester,
        page_count=1,
        parse_notes=timetable_source.TRANSCRIPTION_NOTES,
        ingest_method="verified_transcription",
    )

    if not result["success"]:
        print(f"\nNothing written — no entries passed validation: {result['skipped']}")
        return 1

    print(
        f"\nRetired {result['deactivated_rows']} old row(s) / "
        f"{result['deactivated_uploads']} old upload(s)."
    )
    print(f"Upload id : {result['upload_id']}")
    print(f"Stored    : {result['stored_count']} classes")
    if result["skipped_count"]:
        print(f"SKIPPED   : {result['skipped_count']} — {result['skipped']}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="user_id to load it for")
    parser.add_argument("--pdf", required=True, help="path to Time_Table_IT.pdf")
    parser.add_argument("--semester", default=None,
                        help="semester label; the document does not state one")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="load again even if this file was loaded before")
    args = parser.parse_args()
    try:
        code = asyncio.run(main(args))
    finally:
        asyncio.run(engine.dispose())
    sys.exit(code)
