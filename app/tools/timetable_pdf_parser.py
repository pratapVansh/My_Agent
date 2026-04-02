"""
Timetable PDF parser.
Extracts text from a PDF using PyPDF2, then uses Groq LLM to parse the
timetable (any layout — grid, list, weekly view) into structured entries.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from PyPDF2 import PdfReader

from app.services.groq_service import groq_service
from app.tools.timetable_tool import TimetableInput


class TimetablePDFParser:
    """
    Two-step parser:
      1. PyPDF2  → raw text
      2. Groq LLM → structured JSON entries → List[TimetableInput]
    """

    _DAY_MAP: Dict[str, int] = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }

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
                    "No text could be extracted from the PDF. "
                    "Make sure it is a text-based PDF, not a scanned image."
                ),
                "filename": filename,
            }

        raw_entries, parse_notes = await self._parse_with_llm(raw_text)
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

    # ── Step 2: LLM parsing ───────────────────────────────────────────────────

    _SYSTEM_PROMPT = """You are a timetable parser. Extract every class/lecture from the raw text.

Output ONLY a JSON object — no explanation, no markdown, just the JSON:
{
  "entries": [
    {
      "subject": "Data Structures",
      "day": "Monday",
      "start_time": "09:00",
      "end_time": "10:00",
      "location": "Room 204",
      "instructor": "Dr. Sharma"
    }
  ],
  "notes": "optional notes about the parse"
}

Rules:
- "day" must be one of: Monday Tuesday Wednesday Thursday Friday Saturday Sunday
- "start_time" and "end_time" must be 24-hour HH:MM format (e.g. "14:30")
- If end_time is missing, add 1 hour to start_time
- "location" and "instructor" are null when not mentioned
- For grid timetables (days as columns, time-slots as rows): read each non-empty cell
- Skip blank cells, breaks, lunch, free periods
- Do NOT invent data — only extract what is written in the text
- Output ONLY the raw JSON, nothing else"""

    async def _parse_with_llm(self, raw_text: str) -> Tuple[List[Dict], str]:
        """
        Call Groq with the extracted PDF text and return (entries, notes).
        Truncates input to 4000 chars to avoid token limits.
        """
        text_to_send = raw_text[:4000]

        try:
            response = await groq_service.chat_completion(
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": f"Parse this timetable:\n\n{text_to_send}"},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            content = response.get("content", "")
            return self._extract_json(content)
        except Exception as e:
            return [], f"LLM call failed: {str(e)}"

    # ── JSON extraction (4-strategy fallback) ─────────────────────────────────

    def _extract_json(self, text: str) -> Tuple[List[Dict], str]:
        """Try 4 strategies to parse JSON from LLM response."""
        # 1. Raw text
        try:
            data = json.loads(text.strip())
            return data.get("entries", []), data.get("notes", "Parsed successfully")
        except Exception:
            pass

        # 2. ```json ... ``` block
        m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if m:
            try:
                data = json.loads(m.group(1))
                return data.get("entries", []), data.get("notes", "Parsed from json block")
            except Exception:
                pass

        # 3. ``` ... ``` block
        m = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if m:
            try:
                data = json.loads(m.group(1))
                return data.get("entries", []), data.get("notes", "Parsed from code block")
            except Exception:
                pass

        # 4. First { ... } in text
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                data = json.loads(m.group(0))
                return data.get("entries", []), data.get("notes", "Parsed via substring fallback")
            except Exception:
                pass

        return [], "Could not extract JSON from LLM response"

    # ── Step 3: Convert to TimetableInput ─────────────────────────────────────

    def _to_timetable_inputs(self, entries: List[Dict]) -> List[TimetableInput]:
        """Validate and convert raw dicts to TimetableInput objects."""
        result: List[TimetableInput] = []

        for entry in entries:
            subject = str(entry.get("subject") or "").strip()
            if not subject:
                continue

            day_raw = str(entry.get("day") or "").strip().lower()
            day_int = self._DAY_MAP.get(day_raw, -1)
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
