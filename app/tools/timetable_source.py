"""
The 7th-semester timetable, transcribed from `Time_Table_IT.pdf`.

**Why this file exists, and why it is not a parser.**

`timetable_pdf_parser` extracts text with PyPDF2 and parses it deterministically
— a weekday plus a time range, one class per line, no model call involved. Run
against this PDF it extracts *zero characters*: the document has no text layer
at all — the grid is an image. The parser reports that honestly and the upload
endpoint returns 422, which is the correct behaviour and also means the generic
path cannot ingest this particular document, regardless of how the text is
parsed. OCR would be the general answer; it is a system dependency, a new
failure mode, and it would not fix the harder problem below.

The harder problem is that this grid contains a trap. Reading the 12.30–01.30
column downwards gives:

    Monday      L
    Tuesday     U
    Wednesday   N
    Thursday    C
    Friday      H

That is the word LUNCH written vertically down the break column. Any parser
that reads cells row-by-row — an LLM included — sees five one-letter subjects
and stores "L" as Monday's 12.30 class. So this file records `LUNCH_SLOT_INDEX`
explicitly and excludes it, rather than hoping a model notices.

So the grid was read from the document directly and written down here, and the
transcription is the artifact: a flat, reviewable table anyone can diff against
the PDF in about a minute. It is checked into the repository for exactly that
reason. `SOURCE_PDF_SHA256` pins *which* document it was read from, so a
different or edited PDF cannot silently keep using this transcription.

**Nothing here is inferred.** Every subject name and every faculty name is
copied from the legend printed beneath the grid. Where the PDF gives a venue it
is recorded; where it does not, `room` is None rather than a guess. The one
annotation whose meaning is not stated in the document — `(HUK)` against
Organizational Psychology — is preserved verbatim in `note` and not promoted to
a room, because the PDF never says it is one.

The semester is **not in this document**. It is supplied at upload time and
stored on the `schedule_upload` row, not invented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ── Provenance ───────────────────────────────────────────────────────────────

SOURCE_FILENAME = "Time_Table_IT.pdf"

SOURCE_PDF_SHA256 = (
    "3723215ce284664b18a0ba280cd253926a21c1ae9cc7aaf8293acc8c5750b500"
)
"""SHA-256 of the exact PDF this grid was read from.

The ingestion script refuses to attach this transcription to a file that hashes
differently. A new timetable is a new transcription, not a silent reuse of this
one."""

SOURCE_TITLE = "Time-Table for (IT+IDD23-85)-AB012"
"""The heading printed above the grid, verbatim. Carries the section/batch
identifier; it is stored as-is rather than split, because the document does not
say which part is the section and which is the room."""

SOURCE_PAGE = 1


# ── The time columns ─────────────────────────────────────────────────────────
#
# Eight slots, exactly as printed. Times are normalised from the PDF's "9.30 am
# to 10.30 am" style into 24-hour HH:MM; nothing else about them is changed.

TIME_SLOTS: Tuple[Tuple[str, str], ...] = (
    ("09:30", "10:30"),
    ("10:30", "11:30"),
    ("11:30", "12:30"),
    ("12:30", "13:30"),   # LUNCH — see LUNCH_SLOT_INDEX
    ("13:30", "14:30"),
    ("14:30", "15:30"),
    ("15:30", "16:30"),
    ("16:30", "17:30"),
)

LUNCH_SLOT_INDEX = 3
"""The 12.30–13.30 column. Its cells spell L-U-N-C-H downwards; they are a break
label, not five classes. Excluded from every generated entry."""


# ── The legend ───────────────────────────────────────────────────────────────
#
# Copied from the six lines printed under the grid, plus the single venue note.
# The code→name pairing is by elimination: the grid uses exactly six codes and
# the legend lists exactly six subjects. "OS" ↔ "Organizational Psychology" is
# the one pair whose letters do not line up; it is recorded as printed and
# flagged in `TRANSCRIPTION_NOTES` rather than corrected.

@dataclass(frozen=True)
class Subject:
    code: str
    name: str
    faculty: str
    room: Optional[str] = None


SUBJECTS: Dict[str, Subject] = {
    "SCC": Subject("SCC", "Sustainability & Climate Change", "Dr. Roopa Manjunatha"),
    "GAI": Subject("GAI", "Generative AI", "Dr. Pallabi Saikia"),
    "OS": Subject("OS", "Organizational Psychology", "Dr. Kavita Srivatsava"),
    "DIP": Subject("DIP", "Digital Image Processing", "Dr. Susham Biswas"),
    "POE": Subject("POE", "Principles of Economics", "Dr. Debasish Jena"),
    # The only venue the document states.
    "ESIR": Subject(
        "ESIR", "Employability Skills & Industry Readiness", "Dr. Arjun Deo",
        room="Seminar Hall",
    ),
}

TRANSCRIPTION_NOTES = (
    "The 12.30-13.30 column spells LUNCH vertically and is excluded as a break. "
    "Subject codes are paired to the printed legend by elimination; 'OS' maps to "
    "'Organizational Psychology' as the remaining unmatched entry. '(OE)' and "
    "'(LAB)' are preserved as session markers. '(HUK)' against OS is preserved "
    "verbatim as a note because the document does not say what it denotes. Only "
    "ESIR has a stated venue (Seminar Hall)."
)


# ── The grid ─────────────────────────────────────────────────────────────────
#
# One tuple per weekday, eight cells each, positionally matching TIME_SLOTS.
# `None` is an empty cell in the PDF. Cell text is verbatim.

WEEKDAYS: Tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

GRID: Dict[int, Tuple[Optional[str], ...]] = {
    # 0 = Monday … 4 = Friday, matching `Timetable.day_of_week`.
    0: ("SCC (OE)", "SCC (OE)", None, "L", "POE", "DIP", "DIP", None),
    1: (None, None, "DIP", "U", "POE", None, None, None),
    2: (None, "SCC (OE)", None, "N", "OS (HUK)", "OS (HUK)", "GAI (LAB)", "GAI (LAB)"),
    3: (None, "GAI", "GAI", "C", None, None, "DIP (LAB)", "DIP (LAB)"),
    4: (None, None, None, "H", None, None, "ESIR", "ESIR"),
}


# ── Derivation ───────────────────────────────────────────────────────────────

def _split_cell(cell: str) -> Tuple[str, Optional[str]]:
    """`"GAI (LAB)"` → `("GAI", "LAB")`. `"DIP"` → `("DIP", None)`."""
    text = cell.strip()
    if "(" in text and text.endswith(")"):
        code, _, rest = text.partition("(")
        return code.strip(), rest[:-1].strip() or None
    return text, None


@dataclass(frozen=True)
class SourceEntry:
    """One class, as the PDF states it."""

    day_of_week: int
    start_time: str
    end_time: str
    subject_code: str
    subject_name: str
    faculty: str
    room: Optional[str]
    session_type: str          # "LAB" or "LECTURE"
    note: Optional[str]        # unrecognised annotation, verbatim ("OE", "HUK")
    raw_cell: str
    slot_index: int

    def as_metadata(self) -> Dict[str, object]:
        """The JSONB payload stored beside the row, for provenance and filtering."""
        return {
            "subject_code": self.subject_code,
            "session_type": self.session_type,
            "note": self.note,
            "raw_cell": self.raw_cell,
            "slot_index": self.slot_index,
            "source_title": SOURCE_TITLE,
            "source_page": SOURCE_PAGE,
        }


def entries() -> List[SourceEntry]:
    """
    Every class in the grid, flattened. Deterministic and side-effect free.

    Ordered by day then slot, so the output reads down the printed table.
    Unknown subject codes are skipped rather than stored under their code — a
    code with no legend entry is a transcription error, and storing it would put
    an unexplained abbreviation in front of the user.
    """
    out: List[SourceEntry] = []
    for day in sorted(GRID):
        cells = GRID[day]
        for slot, cell in enumerate(cells):
            if slot == LUNCH_SLOT_INDEX or not cell:
                continue
            code, annotation = _split_cell(cell)
            subject = SUBJECTS.get(code)
            if subject is None:
                continue
            start, end = TIME_SLOTS[slot]
            session_type = "LAB" if (annotation or "").upper() == "LAB" else "LECTURE"
            note = None if session_type == "LAB" else annotation
            out.append(SourceEntry(
                day_of_week=day,
                start_time=start,
                end_time=end,
                subject_code=code,
                subject_name=subject.name,
                faculty=subject.faculty,
                room=subject.room,
                session_type=session_type,
                note=note,
                raw_cell=cell,
                slot_index=slot,
            ))
    return out


__all__ = [
    "GRID",
    "LUNCH_SLOT_INDEX",
    "SOURCE_FILENAME",
    "SOURCE_PAGE",
    "SOURCE_PDF_SHA256",
    "SOURCE_TITLE",
    "SUBJECTS",
    "SourceEntry",
    "Subject",
    "TIME_SLOTS",
    "TRANSCRIPTION_NOTES",
    "WEEKDAYS",
    "entries",
]
