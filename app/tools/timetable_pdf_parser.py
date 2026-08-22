"""
Timetable PDF parser — deterministic, no model call.

Extracts text with PyPDF2, then parses it with fixed rules: a line counts as
one class only if it names a weekday *and* a time range. Everything else in
that line — after the day and the time range are removed — becomes the
subject, unless it also matches an explicit `Room:`/`Instructor:`-style label,
in which case that piece is split out instead of left in the subject text.

This is deliberately narrower than an LLM parse would be. It handles the
common case of a timetable exported as one line of text per class
("Monday 09:00-10:00 Data Structures Room 204 Dr. Sharma"), and it refuses
anything that doesn't fit that shape rather than guessing at a layout it
cannot actually read — a two-dimensional grid table, in particular, does not
survive PyPDF2's text extraction in reading order and is exactly the kind of
document `app/tools/timetable_source.py` exists to handle instead, by hand,
with the source PDF's hash pinned so a different document cannot silently
reuse the transcription.

No entry here is inferred. A line that names a day and a time but nothing
recognisable as a subject is skipped, not guessed at; a room or instructor is
recorded only when the line labels it explicitly. The same discipline the
memory layer applies to personal facts applies here to classes: absence of a
clean parse is reported, never papered over.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PyPDF2 import PdfReader

from app.tools.timetable_tool import TimetableInput

_DAY_MAP: Dict[str, int] = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tues": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thurs": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_DAY_RE = re.compile(
    r"\b(monday|mon|tuesday|tues|tue|wednesday|wed|thursday|thurs|thu"
    r"|friday|fri|saturday|sat|sunday|sun)\b",
    re.IGNORECASE,
)

# "9:00 - 10:00", "9 to 10 am", "09:00am-10:30am". Both sides are captured
# whole (digits, optional minutes, optional meridiem) so `_normalize_time`
# gets exactly what a person would read out loud, not a pre-split fragment.
_TIME_RANGE_RE = re.compile(
    # The optional whitespace before a meridiem lives *inside* that optional
    # group, not next to it — otherwise `\s*` has nothing to backtrack out of
    # when there is no "am"/"pm" to match, and "09:00 " (trailing space) ends
    # up as the captured time instead of "09:00".
    r"(\d{1,2}(?::\d{2})?(?:\s*(?:am|pm|a\.m\.|p\.m\.))?)"
    r"\s*(?:-|to|–|—)\s*"
    r"(\d{1,2}(?::\d{2})?(?:\s*(?:am|pm|a\.m\.|p\.m\.))?)",
    re.IGNORECASE,
)

# Explicit labels only. "Room 204" inside a subject name is not the same
# claim as "Room: 204" — the latter is the document telling the parser what a
# field is; the former is the parser guessing, which this module does not do.
_ROOM_RE = re.compile(
    r"\b(?:room|location|venue|hall)\s*[:#]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9 \-]{0,30}?)"
    r"(?=$|,|;|\.|\bInstructor\b|\bFaculty\b|\bTaught\b|\bDr\b|\bProf)",
    re.IGNORECASE,
)
_INSTRUCTOR_RE = re.compile(
    # No bare "." in the labelled branch's stop condition: a name captured
    # here routinely contains one itself ("Instructor: Dr. Sharma"), and a
    # period-terminated lookahead would cut the title off mid-name.
    r"\b(?:instructor|faculty|taught\s+by)\s*[:#]?\s*"
    r"([A-Za-z][A-Za-z.\-\' ]{1,50}?)(?=$|,|;)"
    r"|(\bDr\.?\s+[A-Za-z][A-Za-z.\-\' ]{1,50}|\bProf\.?\s+[A-Za-z][A-Za-z.\-\' ]{1,50})",
    re.IGNORECASE,
)


class TimetablePDFParser:
    """
    Two-step parser:
      1. PyPDF2 → raw text.
      2. Fixed rules → structured entries → `List[TimetableInput]`.

    No step here makes a network or model call — parsing a timetable never
    blocks on an LLM being reachable, rate-limited, or right.
    """

    # ── Public entry point ────────────────────────────────────────────────────

    async def parse(self, pdf_bytes: bytes, filename: str = "timetable.pdf") -> Dict[str, Any]:
        """
        Parse a timetable PDF into structured TimetableInput entries.

        Returns:
            {
                "success": bool,
                "entries": List[TimetableInput],
                "entry_count": int,
                "pages": int,
                "raw_text_preview": str,   # first 500 chars of extracted text
                "parse_notes": str,
                "filename": str,
            }

        Still `async` even though nothing inside awaits: the signature is the
        contract callers hold, and every caller already awaits it.
        """
        raw_text, num_pages = self._extract_text(pdf_bytes)

        if not raw_text.strip():
            return {
                "success": False,
                "entries": [],
                "entry_count": 0,
                "pages": num_pages,
                "raw_text_preview": "",
                "parse_notes": (
                    "No text could be extracted from the PDF. Make sure it is a "
                    "text-based PDF, not a scanned image — a scanned or "
                    "image-only grid has no text layer for anything, model or "
                    "not, to read."
                ),
                "filename": filename,
            }

        raw_entries, parse_notes = self._parse_deterministic(raw_text)
        timetable_inputs = self._to_timetable_inputs(raw_entries)

        return {
            "success": len(timetable_inputs) > 0,
            "entries": timetable_inputs,
            "entry_count": len(timetable_inputs),
            "pages": num_pages,
            "raw_text_preview": raw_text[:500],
            "parse_notes": parse_notes,
            "filename": filename,
        }

    # ── Step 1: PDF text extraction ───────────────────────────────────────────

    def _extract_text(self, pdf_bytes: bytes) -> Tuple[str, int]:
        """Return (full_text, page_count). Empty string on failure."""
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages: List[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            return "\n\n".join(pages), len(reader.pages)
        except Exception:
            return "", 0

    # ── Step 2: deterministic, line-based parsing ─────────────────────────────

    def _parse_deterministic(self, raw_text: str) -> Tuple[List[Dict], str]:
        """
        One class per line: a weekday, a time range, and whatever remains.

        A line missing either signal is not a class this parser can read and
        is skipped outright — there is no partial-credit guess for "probably a
        continuation of the class above" or "probably a header row".
        """
        entries: List[Dict] = []
        non_blank_lines = 0
        unmatched_lines = 0

        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            non_blank_lines += 1

            day_match = _DAY_RE.search(line)
            time_match = _TIME_RANGE_RE.search(line)
            if not day_match or not time_match:
                unmatched_lines += 1
                continue

            # Remove the matched day and time-range spans, in reverse order of
            # position, so removing one does not shift the other's indices.
            spans = sorted([day_match.span(), time_match.span()], reverse=True)
            remainder = line
            for start, end in spans:
                remainder = remainder[:start] + " " + remainder[end:]

            room = None
            m = _ROOM_RE.search(remainder)
            if m:
                room = (m.group(1) or "").strip(" ,;:-.") or None
                remainder = remainder[:m.start()] + " " + remainder[m.end():]

            instructor = None
            m = _INSTRUCTOR_RE.search(remainder)
            if m:
                instructor = (m.group(1) or m.group(2) or "").strip(" ,;:-.") or None
                remainder = remainder[:m.start()] + " " + remainder[m.end():]

            subject = re.sub(r"\s+", " ", remainder).strip(" ,;:-.")
            if not subject:
                unmatched_lines += 1
                continue

            entries.append({
                "subject": subject,
                "day": day_match.group(1),
                "start_time": time_match.group(1),
                "end_time": time_match.group(2),
                "location": room,
                "instructor": instructor,
            })

        notes = (
            f"Parsed deterministically (no model call): {len(entries)} of "
            f"{non_blank_lines} non-blank line(s) named a weekday and a time "
            "range."
        )
        if unmatched_lines:
            notes += (
                f" {unmatched_lines} line(s) did not and were skipped — this "
                "parser reads one class per line and cannot reconstruct a "
                "two-dimensional grid table."
            )
        return entries, notes

    # ── Step 3: Convert to TimetableInput ─────────────────────────────────────

    def _to_timetable_inputs(self, entries: List[Dict]) -> List[TimetableInput]:
        """Validate and convert raw dicts to TimetableInput objects."""
        result: List[TimetableInput] = []

        for entry in entries:
            subject = str(entry.get("subject") or "").strip()
            if not subject:
                continue

            day_raw = str(entry.get("day") or "").strip().lower()
            day_int = _DAY_MAP.get(day_raw, -1)
            if day_int == -1:
                continue

            start_time = self._normalize_time(str(entry.get("start_time") or ""))
            end_time = self._normalize_time(str(entry.get("end_time") or ""))
            if not start_time or not end_time:
                continue

            location = (str(entry.get("location") or "")).strip() or None
            instructor = (str(entry.get("instructor") or "")).strip() or None

            result.append(TimetableInput(
                day_of_week=day_int,
                start_time=start_time,
                end_time=end_time,
                subject=subject,
                location=location,
                instructor=instructor,
            ))

        return result

    def _normalize_time(self, value: str) -> Optional[str]:
        """Parse any time string and return HH:MM (24-hour). Returns None on failure."""
        if not value:
            return None
        v = value.strip().lower().replace(" ", "")
        for fmt in ("%H:%M", "%H:%M:%S", "%I:%M%p", "%I%p"):
            try:
                return datetime.strptime(v, fmt).strftime("%H:%M")
            except Exception:
                continue
        return None


# Singleton
timetable_pdf_parser = TimetablePDFParser()
