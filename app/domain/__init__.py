"""
Application records — data the app stores, as opposed to what the assistant
remembers about the user.

Attendance, timetables, exams, plans, job bookmarks, and email drafts all used
to live under `app/memory/`. That conflation is what produced a 1,145-line
`ShortTermMemory` owning twelve unrelated entity types and a `MemoryManager`
carrying thirty pass-through methods to reach them.

Memory lives in `app/memory/`. See docs/MEMORY_ARCHITECTURE.md §1.2.
"""
from app.domain.academic import academic_repository
from app.domain.email import email_repository
from app.domain.jobs import jobs_repository

__all__ = [
    "academic_repository",
    "email_repository",
    "jobs_repository",
]
