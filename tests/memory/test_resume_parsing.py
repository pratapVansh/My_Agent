"""
Résumé section parsing — regression tests.

Every case here comes from a real failure observed on a real PDF, not from an
invented résumé. The originating defects, all in `_extract_semantic_resume_chunks`:

1. **Substring heading matching deleted content.** A line was treated as a
   heading if any known heading appeared *anywhere* in it, so
   "Programming Languages: C/C++, Python, SQL" and "Tools: Git, Linux" were
   consumed as delimiters. Both were absent from Qdrant entirely — the two most
   valuable lines of the skills section, silently dropped at ingest.

2. **Entry segmentation never fired.** Splitting required `^[-*•]\\s+`, but
   PyPDF2 renders Wingdings bullets as U+FFFD with no trailing space, so three
   projects were stored as one 1451-character blob.

3. **Unknown headings contaminated the previous section.** "Relevant Coursework"
   was not in the vocabulary, so coursework accreted onto Projects.

4. **The stored résumé did not round-trip.** Headings were consumed as
   delimiters and never persisted; `retrieve_resume` rebuilds the résumé by
   joining stored chunks, so re-parsing it found *no* sections at all. This is
   what made the smoke test report "missing sections: skills, projects,
   experience, education" for a résumé whose headings were perfectly clean —
   the parser was failing on its own output, not on the PDF.

5. **`_detect_name` had no heading guard**, storing "SKILLS" as the user's
   identity at high importance.
"""
import pytest

from app.memory.long_term_memory_qdrant import (
    classify_heading,
    long_term_memory_qdrant,
)

extract_chunks = long_term_memory_qdrant._extract_semantic_resume_chunks
detect_name = long_term_memory_qdrant._detect_name


def types_of(chunks):
    return {c["type"] for c in chunks}


def content_of(chunks, kind):
    return [c["content"] for c in chunks if c["type"] == kind]


def joined(chunks, kind):
    return "\n".join(content_of(chunks, kind))


# The bullet glyph PyPDF2 actually emitted for this résumé's Wingdings bullets.
PDF_BULLET = "�"

# A faithful reproduction of the extractor's output: glued words, U+FFFD
# bullets, no space after the bullet, dates fused to the preceding token.
REAL_PDF_RESUME = f"""Vansh Pratap Singh+91-6392306428
Roll No.:23IT3048 vanshprataps2004@gmail.com
Information Technology Github
Rajiv Gandhi Institute of Petroleum Technology LinkedIn
Education
Rajiv Gandhi Institute of Petroleum Technology (Institution of National Importance)Jais, Amethi
B.Tech. in Information Technology (CGPA: 8.80 / 10)Aug 2023-Present
Experience
DrUpsc (EdTech) May 2026 {chr(8211)} June 2026
Full Stack Developer Intern Remote
{PDF_BULLET}Built5+ production-ready featuresfor profile customization.
{PDF_BULLET}Implemented anAWS S3-based image upload system.
Department of Computer Science, Institute of Science, BHUJune 2025 {chr(8211)} July 2025
Research Intern {chr(8211)} Under Prof. Manjari Gupta Varanasi, India
{PDF_BULLET}Conductedquantitativemeta-analysisof29+EEG/MRIstudiesusingPython, Pandas, andScikit-learnformachinelearning-
based schizophrenia classification.
{PDF_BULLET}Evaluated preprocessing techniques, improving classification accuracy by up
to15%.
Projects
Multi-Agent AI Voice Assistant (My_Agent) | GitHubJuly 2026 {chr(8211)} Present
Low-latency AI voice assistant with streaming speech.
{PDF_BULLET}Built a multi-agent AI voice assistant usingFastAPIandLangGraph.
{PDF_BULLET}Built a hybrid memory system usingPostgreSQL,Qdrant, andMem0.
Campus Placement Management System | GitHub | LiveJan 2026 {chr(8211)} March 2026
Full-stack platform for end-to-end campus placement management.
{PDF_BULLET}Built a scalable CPMS usingNext.js,Node.js,PostgreSQL, andPrisma.
TRACE | GitHub June 2026 {chr(8211)} July 2026
AI-powered platform for intelligent document processing.
{PDF_BULLET}Built a document intelligence platform withFastAPI.
Relevant Coursework
Data Structures and Algorithms, Object-Oriented Programming, Database Management Systems (DBMS),
Operating Systems, Web Technology, Discrete Mathematics, Computer Networks
Technical Skills
Programming Languages:C/C++, Python, JavaScript (ES6+), TypeScript, SQL
Frontend:React.js, Next.js, HTML5, CSS3, Tailwind CSS
Backend:FastAPI, Node.js, Express.js, REST APIs, WebSockets
Databases & ORM:PostgreSQL, MongoDB, Redis, Qdrant, Neo4j, Prisma ORM
Cloud & DevOps:AWS (EC2, S3), Docker, GitHub Actions, CI/CD
AI:LangGraph, RAG, Groq API
Authentication & Security:JWT, OAuth 2.0, RBAC, Rate Limiting
Tools:Git, Linux, Shell Scripting, Postman, Vercel, Render
Achievements
Awarded theRGIPT Merit-cum-Means Scholarshipfor ranking among the top10%of B.Tech (IT) students.
"""


# ─────────────────────────────────────────────────────────────────────────
# Defect 1 — labelled content lines are content, not headings
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "line",
    [
        "Programming Languages:C/C++, Python, JavaScript (ES6+), TypeScript, SQL",
        "Tools:Git, Linux, Shell Scripting, Postman, Vercel, Render",
    ],
)
def test_labelled_skill_lines_survive_ingestion(line):
    """
    These two lines were deleted outright by the old substring matcher. The
    label may legitimately *be* a heading word, so the line must be kept as
    content whichever way it is classified.
    """
    chunks = extract_chunks(f"Technical Skills\n{line}\n")
    assert line.split(":", 1)[1].split(",")[0].strip() in joined(chunks, "skills")


def test_every_skill_label_reaches_the_skills_section():
    skills = joined(extract_chunks(REAL_PDF_RESUME), "skills")
    for expected in ["C/C++", "React.js", "FastAPI", "PostgreSQL", "AWS",
                     "LangGraph", "JWT", "Shell Scripting"]:
        assert expected in skills, f"{expected!r} lost during ingestion"


@pytest.mark.parametrize(
    "line",
    [
        "Frontend:React.js, Next.js, HTML5",
        "AI:LangGraph, RAG, Groq API",
        "Databases & ORM:PostgreSQL, MongoDB",
        "Roll No.:23IT3048 vanshprataps2004@gmail.com",
        "B.Tech. in Information Technology (CGPA: 8.80 / 10)Aug 2023-Present",
        "Rajiv Gandhi Institute of Petroleum Technology (Institution of National Importance)Jais, Amethi",
        "Full Stack Developer Intern Remote",
        "Information Technology Github",
    ],
)
def test_content_lines_are_never_read_as_headings(line):
    assert classify_heading(line) is None


def test_a_bullet_is_always_content_even_when_it_names_a_section():
    assert classify_heading(f"{PDF_BULLET}Led the migration, gaining experience") is None
    assert classify_heading("• Skills") is None


# ─────────────────────────────────────────────────────────────────────────
# Heading recognition — real-world variants
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "line,expected",
    [
        ("SKILLS", "skills"),
        ("Technical Skills", "skills"),
        ("TECHNICAL SKILLS:", "skills"),
        ("Skills & Tools", "skills"),
        ("TECHNICAL SKILLS (Python, Java)", "skills"),
        ("  technologies   ", "skills"),
        ("Core Competencies", "skills"),
        ("PROJECTS", "projects"),
        ("Academic Projects", "projects"),
        ("Key Projects", "projects"),
        ("EXPERIENCE", "experience"),
        ("Work Experience", "experience"),
        ("Professional Experience", "experience"),
        ("Work History", "experience"),
        ("Internships", "experience"),
        ("EDUCATION", "education"),
        ("Academic Qualifications", "education"),
        ("Educational Background", "education"),
        ("ACHIEVEMENTS", "achievements"),
        ("Honors & Awards", "achievements"),
        ("Certifications", "achievements"),
        ("Relevant Coursework", "other"),
        ("Positions of Responsibility", "other"),
        ("Professional Summary", "other"),
        ("Career Objective", "other"),
    ],
)
def test_heading_variants_classify(line, expected):
    match = classify_heading(line)
    assert match is not None, f"{line!r} was not recognised as a heading"
    assert match[0] == expected


def test_inline_heading_keeps_both_meanings():
    """"SKILLS: Python, Java" is a boundary and a content line at once."""
    section, inline = classify_heading("SKILLS: Python, Java")
    assert section == "skills"
    assert inline == "Python, Java"

    chunks = extract_chunks("Jane Doe\n\nSKILLS: Python, Java\n\nEDUCATION\nB.Tech\n")
    assert "Python" in joined(chunks, "skills")
    assert "education" in types_of(chunks)


# ─────────────────────────────────────────────────────────────────────────
# Defect 2 — entry segmentation across PDF bullet glyphs
# ─────────────────────────────────────────────────────────────────────────

def test_each_project_becomes_its_own_chunk():
    projects = content_of(extract_chunks(REAL_PDF_RESUME), "projects")
    assert len(projects) == 3, f"expected 3 project entries, got {len(projects)}"
    blob = "\n".join(projects)
    for title in ["My_Agent", "Campus Placement", "TRACE"]:
        assert title in blob


def test_each_employer_becomes_its_own_chunk():
    experience = content_of(extract_chunks(REAL_PDF_RESUME), "experience")
    assert len(experience) == 2, f"expected 2 employers, got {len(experience)}"
    assert "DrUpsc" in experience[0]
    assert "BHU" in experience[1]


@pytest.mark.parametrize("bullet", ["-", "*", "•", "●", "▪", "‣", "�", ""])
def test_bullet_glyph_variants_all_segment(bullet):
    text = (
        "PROJECTS\n"
        "First Project | GitHub\n"
        f"{bullet}Did a thing.\n"
        "Second Project | GitHub\n"
        f"{bullet}Did another thing.\n"
    )
    projects = content_of(extract_chunks(text), "projects")
    assert len(projects) == 2
    assert not any(p.lstrip().startswith(bullet) for p in projects)


def test_wrapped_bullet_does_not_start_a_new_entry():
    """
    PDF line wrapping splits one bullet across two lines. The continuation is
    not a new employer — this produced 4 experience chunks where 2 were right.
    """
    experience = content_of(extract_chunks(REAL_PDF_RESUME), "experience")
    assert "based schizophrenia classification." in experience[1]
    assert "to15%." in experience[1]


# ─────────────────────────────────────────────────────────────────────────
# Defect 3 — unknown sections must not contaminate the previous one
# ─────────────────────────────────────────────────────────────────────────

def test_coursework_does_not_leak_into_projects():
    projects = joined(extract_chunks(REAL_PDF_RESUME), "projects")
    assert "Discrete Mathematics" not in projects
    assert "Operating Systems" not in projects


def test_an_unknown_section_still_ends_the_previous_one():
    chunks = extract_chunks(
        "PROJECTS\nBuilt a thing | GitHub\n• Shipped it.\n"
        "Positions of Responsibility\nClass representative for two years.\n"
    )
    assert "Class representative" not in joined(chunks, "projects")


# ─────────────────────────────────────────────────────────────────────────
# Defect 4 — the stored résumé must re-parse into the same sections
# ─────────────────────────────────────────────────────────────────────────

def test_stored_resume_round_trips_through_reassembly():
    """
    `store_resume` persists chunk text; `retrieve_resume` rebuilds the résumé by
    joining those chunks with a blank line. Re-parsing that reconstruction must
    still find every section, or the stored résumé is structurally lossy — which
    is exactly what the smoke test was reporting.
    """
    first_pass = extract_chunks(REAL_PDF_RESUME)
    reassembled = "\n\n".join(c["content"] for c in first_pass)
    second_pass = extract_chunks(reassembled)

    for section in ["skills", "projects", "experience", "education"]:
        assert section in types_of(second_pass), (
            f"{section!r} lost on round-trip through storage"
        )


def test_round_trip_preserves_the_detected_name():
    first_pass = extract_chunks(REAL_PDF_RESUME)
    reassembled = "\n\n".join(c["content"] for c in first_pass)
    assert content_of(extract_chunks(reassembled), "name") == ["Vansh Pratap Singh"]


# ─────────────────────────────────────────────────────────────────────────
# Defect 5 — a heading is not a name
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("heading", ["SKILLS", "EXPERIENCE", "EDUCATION", "Technical Skills"])
def test_a_heading_is_never_stored_as_the_users_name(heading):
    assert detect_name([heading, "Python, FastAPI"]) is None


def test_a_real_name_is_still_detected_ahead_of_headings():
    chunks = extract_chunks(REAL_PDF_RESUME)
    assert content_of(chunks, "name") == ["Vansh Pratap Singh"]


# ─────────────────────────────────────────────────────────────────────────
# Whole-document classification
# ─────────────────────────────────────────────────────────────────────────

def test_all_four_sections_are_classified_from_the_real_pdf():
    types = types_of(extract_chunks(REAL_PDF_RESUME))
    assert {"skills", "projects", "experience", "education", "name"} <= types


def test_education_content_lands_in_education():
    education = joined(extract_chunks(REAL_PDF_RESUME), "education")
    assert "CGPA: 8.80" in education
    assert "Rajiv Gandhi Institute" in education


def test_achievements_are_stored_as_other_and_tagged():
    chunks = extract_chunks(REAL_PDF_RESUME)
    achievement = next(c for c in chunks if "Merit-cum-Means" in c["content"])
    assert achievement["type"] == "other"
    assert "achievements" in achievement["tags"]


def test_chunk_shape_is_unchanged():
    for chunk in extract_chunks(REAL_PDF_RESUME):
        assert set(chunk) == {"type", "content", "tags", "importance"}
        assert chunk["content"].strip()
        assert chunk["importance"] in {"high", "medium"}


def test_unstructured_text_still_falls_back_to_chunks():
    """A résumé with no recognisable headings must still produce usable chunks."""
    prose = "\n\n".join(
        f"This is paragraph number {i} and it contains more than eight words."
        for i in range(4)
    )
    chunks = extract_chunks(prose)
    assert len(chunks) >= 3
    assert all(c["content"].strip() for c in chunks)


def test_empty_input_is_still_empty():
    assert extract_chunks("") == []
    assert extract_chunks("   \n\n  ") == []
