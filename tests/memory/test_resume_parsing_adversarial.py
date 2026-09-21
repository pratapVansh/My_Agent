"""
Adversarial résumé parsing — layout and extraction-artifact robustness.

`test_resume_parsing.py` pins the defects found on one real PDF. This file
attacks the parser with the layouts and artifacts it will meet in the wild:
capitalisation and punctuation variants, wrapped lines, Unicode damage,
one- and two-column extraction, missing and reordered sections, and résumés
with no bullets or nothing but bullets.

The standard applied throughout is *graceful degradation*: an unparseable
layout must still yield usable chunks with nothing silently deleted. A résumé
whose structure we cannot read is a worse answer, never a lost one.
"""
import pytest

from app.memory.long_term_memory_qdrant import (
    classify_heading,
    derive_entry_title,
    _segment_entries,
    long_term_memory_qdrant,
)

extract_chunks = long_term_memory_qdrant._extract_semantic_resume_chunks


def types_of(chunks):
    return {c["type"] for c in chunks}


def content_of(chunks, kind):
    return [c["content"] for c in chunks if c["type"] == kind]


def all_text(chunks):
    return "\n".join(c["content"] for c in chunks)


# ─────────────────────────────────────────────────────────────────────────
# Capitalisation and punctuation
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "heading",
    [
        "SKILLS", "Skills", "skills", "SkIlLs", "  Skills  ",
        "Skills:", "SKILLS :", "Skills.", "~ SKILLS ~", "— Skills —",
        "**Skills**", "[ SKILLS ]", "SKILLS…",
    ],
)
def test_heading_survives_capitalisation_and_decoration(heading):
    match = classify_heading(heading)
    assert match is not None, f"{heading!r} not recognised"
    assert match[0] == "skills"


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("WORK EXPERIENCE", "experience"),
        ("Work-Experience", "experience"),
        ("EDUCATION & TRAINING", "education"),
        ("Education / Qualifications", "education"),
        ("PROJECTS:", "projects"),
        ("Technical Skills / Tools", "skills"),
        ("AWARDS AND HONORS", "achievements"),
    ],
)
def test_punctuation_between_heading_words(heading, expected):
    match = classify_heading(heading)
    assert match is not None, f"{heading!r} not recognised"
    assert match[0] == expected


def test_heading_with_and_without_colon_are_equivalent():
    assert classify_heading("EDUCATION")[0] == "education"
    assert classify_heading("EDUCATION:")[0] == "education"
    with_inline = classify_heading("EDUCATION: B.Tech, 2023")
    assert with_inline[0] == "education"
    assert with_inline[1] == "B.Tech, 2023"


def test_a_sentence_mentioning_a_section_is_not_a_heading():
    for line in [
        "Extensive experience with distributed systems and cloud tooling.",
        "Skills were developed over four years of professional work.",
        "Responsible for education outreach across three universities.",
    ]:
        assert classify_heading(line) is None


# ─────────────────────────────────────────────────────────────────────────
# Unicode / PDF extraction artifacts
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "artifact",
    ["�", "", "•", "●", "▪", "‣", "·"],
)
def test_private_use_and_replacement_bullets_are_treated_as_bullets(artifact):
    text = f"PROJECTS\nAlpha | GitHub\n{artifact}Shipped it.\nBeta | GitHub\n{artifact}Shipped it too.\n"
    projects = content_of(extract_chunks(text), "projects")
    assert len(projects) == 2
    # Bullet glyphs are stripped from the typed entries. (The paragraph fallback
    # that guarantees a minimum chunk count quotes raw text verbatim, so this
    # asserts on the typed chunks rather than on every chunk.)
    assert artifact not in "\n".join(projects)
    assert "Shipped it." in projects[0]


def test_non_breaking_spaces_and_smart_quotes_do_not_break_headings():
    assert classify_heading("TECHNICAL SKILLS")[0] == "skills"
    assert classify_heading("Employee’s Experience") is None  # not a heading
    chunks = extract_chunks("Jane Roe\n\nSKILLS \nPython’s ecosystem, FastAPI\n")
    assert "skills" in types_of(chunks)


def test_zero_width_and_control_characters_are_tolerated():
    text = "Jane Roe\n\nSKILLS​\nPython, FastAPI\n\nEDUCATION\nB.Tech\n"
    chunks = extract_chunks(text)
    assert {"skills", "education"} <= types_of(chunks)


def test_a_resume_that_is_one_long_line_still_produces_chunks():
    chunks = extract_chunks("Jane Roe SKILLS Python FastAPI EDUCATION B.Tech " * 20)
    assert chunks
    assert all(c["content"].strip() for c in chunks)


# ─────────────────────────────────────────────────────────────────────────
# Wrapped lines
# ─────────────────────────────────────────────────────────────────────────

def test_lowercase_wrap_continues_the_previous_entry():
    text = (
        "PROJECTS\n"
        "Alpha Engine | GitHub\n"
        "• Built a pipeline processing several million records per day using\n"
        "distributed workers and a queue.\n"
        "Beta Service | GitHub\n"
        "• Did other things.\n"
    )
    projects = content_of(extract_chunks(text), "projects")
    assert len(projects) == 2
    assert "distributed workers" in projects[0]


def test_wrap_beginning_with_a_capitalised_word_still_continues():
    """The previous line ends on a dangling preposition, so it cannot have ended."""
    text = (
        "EXPERIENCE\n"
        "Acme Corp | Remote  Jan 2020 - Dec 2022\n"
        "• Migrated the billing platform to\n"
        "Kubernetes across three regions.\n"
        "Globex | Onsite  Jan 2023 - Present\n"
        "• Led the platform team.\n"
    )
    experience = content_of(extract_chunks(text), "experience")
    assert len(experience) == 2
    assert "Kubernetes" in experience[0]


def test_hyphen_broken_word_continues():
    text = (
        "PROJECTS\n"
        "Alpha | GitHub\n"
        "• Implemented a machine-\n"
        "learning classifier.\n"
    )
    projects = content_of(extract_chunks(text), "projects")
    assert len(projects) == 1


# ─────────────────────────────────────────────────────────────────────────
# Column layouts
# ─────────────────────────────────────────────────────────────────────────

def test_single_column_with_right_aligned_dates():
    text = (
        "EXPERIENCE\n"
        "Acme Corp                                   Jan 2020 - Dec 2022\n"
        "Software Engineer                           Bangalore, India\n"
        "• Built the billing system.\n"
        "Globex                                      Jan 2023 - Present\n"
        "Senior Engineer                             Remote\n"
        "• Led the platform team.\n"
    )
    experience = content_of(extract_chunks(text), "experience")
    assert len(experience) == 2
    assert "Acme Corp" in experience[0] and "Globex" in experience[1]


def test_two_column_extraction_keeps_sections_and_loses_nothing():
    """
    Two-column PDFs extract with the right column fused onto the left. Headings
    on their own line still classify; fused lines stay content rather than being
    mistaken for headings, and no text is dropped.
    """
    text = (
        "Jane Roe\n"
        "Bachelor of Technology Github\n"
        "State Institute of Technology LinkedIn\n"
        "EDUCATION\n"
        "State Institute of Technology (Autonomous)Springfield\n"
        "B.Tech. in Computer Science (CGPA: 9.10 / 10)Aug 2021-Present\n"
        "TECHNICAL SKILLS\n"
        "Programming Languages:Python, Go\n"
        "Tools:Docker, Kubernetes\n"
    )
    chunks = extract_chunks(text)
    assert {"education", "skills"} <= types_of(chunks)
    joined = all_text(chunks)
    for fragment in ["CGPA: 9.10", "Python, Go", "Docker, Kubernetes", "LinkedIn"]:
        assert fragment in joined, f"{fragment!r} lost"


def test_a_fused_heading_line_does_not_swallow_the_column_beside_it():
    # "EDUCATION Springfield, IL" — heading fused with right-column text. It is
    # not classified as a heading, but the text must survive as content.
    chunks = extract_chunks("Jane Roe\n\nEDUCATION Springfield, IL\nB.Tech 2021\n")
    assert "Springfield, IL" in all_text(chunks)


# ─────────────────────────────────────────────────────────────────────────
# Missing / reordered / duplicated sections
# ─────────────────────────────────────────────────────────────────────────

def test_missing_sections_do_not_fabricate_chunks():
    chunks = extract_chunks("Jane Roe\n\nSKILLS\nPython, FastAPI\n")
    assert "skills" in types_of(chunks)
    assert "projects" not in types_of(chunks)
    assert "experience" not in types_of(chunks)


@pytest.mark.parametrize(
    "order",
    [
        ["SKILLS", "PROJECTS", "EXPERIENCE", "EDUCATION"],
        ["EDUCATION", "EXPERIENCE", "PROJECTS", "SKILLS"],
        ["PROJECTS", "EDUCATION", "SKILLS", "EXPERIENCE"],
        ["EXPERIENCE", "SKILLS", "EDUCATION", "PROJECTS"],
    ],
)
def test_section_order_does_not_change_classification(order):
    body = {
        "SKILLS": "Python, FastAPI, PostgreSQL",
        "PROJECTS": "Alpha | GitHub\n• Built a thing.",
        "EXPERIENCE": "Acme Corp | Remote  Jan 2020 - Dec 2022\n• Did the work.",
        "EDUCATION": "B.Tech in Computer Science",
    }
    text = "Jane Roe\n\n" + "\n\n".join(f"{h}\n{body[h]}" for h in order)
    assert {"skills", "projects", "experience", "education"} <= types_of(extract_chunks(text))


def test_a_section_opened_twice_keeps_both_blocks():
    text = (
        "Jane Roe\n\n"
        "PROJECTS\nAlpha | GitHub\n• Built alpha.\n\n"
        "EXPERIENCE\nAcme | Remote  Jan 2020 - Dec 2022\n• Worked.\n\n"
        "PROJECTS\nBeta | GitHub\n• Built beta.\n"
    )
    projects = content_of(extract_chunks(text), "projects")
    assert any("Alpha" in p for p in projects)
    assert any("Beta" in p for p in projects)


def test_content_before_any_heading_is_never_dropped():
    chunks = extract_chunks(
        "Jane Roe\njane@example.com | +1 555 0100\nSenior Engineer, distributed systems\n\n"
        "SKILLS\nPython\n"
    )
    assert "distributed systems" in all_text(chunks)


# ─────────────────────────────────────────────────────────────────────────
# Bullet-free and bullet-heavy layouts
# ─────────────────────────────────────────────────────────────────────────

def test_bullet_free_projects_split_on_title_lines():
    text = (
        "PROJECTS\n"
        "Portfolio Website | GitHub  Jan 2024 - Mar 2024\n"
        "Built with React and deployed on Vercel in 2024.\n"
        "Chat App | GitHub  Apr 2024 - Jun 2024\n"
        "Real-time messaging with WebSockets.\n"
    )
    projects = content_of(extract_chunks(text), "projects")
    assert len(projects) == 2
    assert "Portfolio Website" in projects[0] and "Chat App" in projects[1]


def test_bullet_free_projects_split_on_blank_lines():
    text = (
        "PROJECTS\n\n"
        "Portfolio Website\nBuilt with React.\n\n"
        "Chat App\nReal-time messaging.\n\n"
        "Task Manager\nA Kanban board.\n"
    )
    assert len(content_of(extract_chunks(text), "projects")) == 3


def test_one_project_per_line_with_no_bullets_or_dates():
    text = "PROJECTS\nPortfolio Website | GitHub\nChat App | GitHub\nTask Manager | GitHub\n"
    assert len(content_of(extract_chunks(text), "projects")) == 3


def test_a_description_containing_a_year_does_not_split_a_project():
    text = (
        "PROJECTS\n"
        "Portfolio Website | GitHub\n"
        "Built with React and deployed on Vercel in 2024.\n"
    )
    assert len(content_of(extract_chunks(text), "projects")) == 1


def test_double_spaced_extraction_does_not_split_every_line():
    """A blank between *every* line carries no information and must be ignored."""
    text = (
        "PROJECTS\n\n"
        "Portfolio Website | GitHub\n\n"
        "Built with React.\n\n"
        "Chat App | GitHub\n\n"
        "Real-time messaging.\n"
    )
    assert len(content_of(extract_chunks(text), "projects")) == 2


def test_a_section_of_nothing_but_bullets_splits_per_bullet():
    text = (
        "PROJECTS\n"
        "• Portfolio Website - React and Vite\n"
        "• Chat App - WebSockets and Redis\n"
        "• Task Manager - Kanban board\n"
    )
    projects = content_of(extract_chunks(text), "projects")
    assert len(projects) == 3
    assert not any(p.startswith("•") for p in projects)


def test_excessive_bullets_under_titles_stay_with_their_title():
    text = "PROJECTS\nAlpha | GitHub\n" + "".join(
        f"• Bullet number {i} describing the work.\n" for i in range(12)
    ) + "Beta | GitHub\n• Only one bullet.\n"
    projects = content_of(extract_chunks(text), "projects")
    assert len(projects) == 2
    assert projects[0].count("Bullet number") == 12


def test_company_and_role_on_separate_lines_stay_one_entry():
    text = (
        "EXPERIENCE\n"
        "Acme Corp - Remote  Jan 2020 - Dec 2022\n"
        "Software Engineer - Platform team\n"
        "• Built billing.\n"
    )
    assert len(content_of(extract_chunks(text), "experience")) == 1


def test_segmenting_an_empty_or_blank_section_is_safe():
    assert _segment_entries([]) == []
    assert _segment_entries(["", "  ", ""]) == []


# ─────────────────────────────────────────────────────────────────────────
# Entry titles
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "content,expected",
    [
        ("Projects\nTRACE | GitHub June 2026 - July 2026\nAI platform.", "TRACE"),
        ("Alpha Engine | GitHub\n• Built it.", "Alpha Engine"),
        ("Portfolio Website  Jan 2024 - Mar 2024", "Portfolio Website"),
        ("Campus Placement Management System | GitHub | Live", "Campus Placement Management System"),
    ],
)
def test_entry_titles_are_derived_without_the_section_heading(content, expected):
    assert derive_entry_title(content) == expected


def test_entry_title_of_empty_content_is_empty():
    assert derive_entry_title("") == ""
    assert derive_entry_title("\n\n") == ""


# ─────────────────────────────────────────────────────────────────────────
# Nothing is ever silently deleted
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text",
    [
        "SKILLS\nPython, Go\nPROJECTS\nAlpha | GitHub\n",
        "Jane Roe\nSKILLS: Python\nTools: Docker, Kubernetes\n",
        "PROJECTS\n• One\n• Two\n• Three\n",
        "EXPERIENCE\nAcme  2020-2022\nEngineer\nEDUCATION\nB.Tech  2016-2020\n",
        "no headings at all, just a paragraph of prose about a career in software",
    ],
)
def test_distinctive_tokens_always_survive_ingestion(text):
    joined = all_text(extract_chunks(text)).lower()
    for token in ["python", "go", "alpha", "docker", "kubernetes", "acme",
                  "b.tech", "three", "career"]:
        if token in text.lower():
            assert token in joined, f"{token!r} lost from {text!r}"
