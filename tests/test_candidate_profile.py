"""
The structured profile, and the guarantee that it never invents a qualification.

The property under test throughout is narrow and absolute:

    every skill in a CandidateProfile traces to text the user supplied

Ranking a job wrongly is recoverable. Telling a recruiter the user knows
Kubernetes because the word appeared in the posting is not — so most of what
follows is about what the profile *refuses* to contain.

The fixtures are not invented. They are the real stored résumé text from
`test_resume_parsing.py`, glued words and all:

    Built a multi-agent AI voice assistant usingFastAPIandLangGraph.

There is no word boundary before `FastAPI` there. A profile that cannot find it
would under-claim on every project the user has actually built, and the
adversarial half of that — finding `Go` inside `going` — is what the
`boundary_only` rule exists to prevent. Both are pinned below.
"""
from __future__ import annotations

import pytest

from app.candidate import (
    CandidateProfile,
    CandidateProfileBuilder,
    Evidence,
    EvidenceKind,
    ProfileSource,
    SkillClaim,
)
from app.candidate.vocabulary import (
    canonical_for,
    find_skills,
    normalise_surface,
    split_skill_line,
)
from app.memory.retrieval_result import RetrievalResult

OWNER = "owner@example.com"


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — real stored résumé text
# ═══════════════════════════════════════════════════════════════════════════

SKILLS_CHUNK = """Technical Skills
Programming Languages:C/C++, Python, JavaScript (ES6+), TypeScript, SQL
Frontend:React.js, Next.js, HTML5, CSS3, Tailwind CSS
Backend:FastAPI, Node.js, Express.js, REST APIs, WebSockets
Databases & ORM:PostgreSQL, MongoDB, Redis, Qdrant, Neo4j, Prisma ORM
Cloud & DevOps:AWS (EC2, S3), Docker, GitHub Actions, CI/CD
AI:LangGraph, RAG, Groq API
Authentication & Security:JWT, OAuth 2.0, RBAC, Rate Limiting
Tools:Git, Linux, Shell Scripting, Postman, Vercel, Render"""

# Note the glued words — exactly as PyPDF2 renders this résumé.
PROJECT_CHUNK = (
    "Multi-Agent AI Voice Assistant (My_Agent) | GitHubJuly 2026 - Present\n"
    "Low-latency AI voice assistant with streaming speech.\n"
    "Built a multi-agent AI voice assistant usingFastAPIandLangGraph.\n"
    "Built a hybrid memory system usingPostgreSQL,Qdrant, andMem0."
)

EXPERIENCE_CHUNK = (
    "Experience\n"
    "Department of Computer Science, Institute of Science, BHU June 2025 - July 2025\n"
    "Research Intern - Under Prof. Manjari Gupta Varanasi, India\n"
    "Conducted quantitative meta-analysis of 29+ EEG/MRI studies using Python, "
    "Pandas, and Scikit-learn for machine learning-based classification."
)

EDUCATION_CHUNK = (
    "Education\n"
    "Rajiv Gandhi Institute of Petroleum Technology Jais, Amethi\n"
    "B.Tech. in Information Technology (CGPA: 8.80 / 10) Aug 2023-Present"
)

ACHIEVEMENT_CHUNK = (
    "Achievements\n"
    "Awarded the RGIPT Merit-cum-Means Scholarship for ranking among the top 10%."
)


def _item(content, *, string_id="chunk_1", entity_id=None, title=None):
    metadata = {"string_id": string_id, "parent_id": "resume_1"}
    if entity_id:
        metadata["entity_id"] = entity_id
    out = {"content": content, "metadata": metadata}
    if title:
        out["title"] = title
    return out


class FakeMemory:
    """
    Stands in for the Qdrant store, returning the same `RetrievalResult` shapes.

    Per-source failure is configurable because that is the case the profile has
    to get right: an errored lookup and an empty section produce identical data
    and license opposite statements.
    """

    def __init__(self, *, skills=None, projects=None, experience=None,
                 education=None, achievements=None, name=None, fail=()):
        self._data = {
            "skills": skills if skills is not None else [_item(SKILLS_CHUNK)],
            "projects": projects if projects is not None else [
                _item(PROJECT_CHUNK, string_id="chunk_5",
                      entity_id="resume_1_project_1",
                      title="Multi-Agent AI Voice Assistant (My_Agent)")
            ],
            "experience": experience if experience is not None else [
                _item(EXPERIENCE_CHUNK, string_id="chunk_3")
            ],
            "education": education if education is not None else [
                _item(EDUCATION_CHUNK, string_id="chunk_2")
            ],
            "achievements": achievements if achievements is not None else [
                _item(ACHIEVEMENT_CHUNK, string_id="chunk_9")
            ],
            "name": name if name is not None else [_item("Vansh Pratap Singh")],
        }
        self._fail = set(fail)

    def _result(self, key):
        if key in self._fail:
            return RetrievalResult.error()
        items = self._data.get(key) or []
        return RetrievalResult.ok(items) if items else RetrievalResult.no_data()

    async def retrieve_skills(self, user_id, query=None, limit=10):
        return self._result("skills")

    async def retrieve_projects(self, user_id, query=None, limit=10):
        return self._result("projects")

    async def retrieve_section(self, user_id, section, limit=5):
        return self._result(section)


async def build(**kwargs) -> CandidateProfile:
    return await CandidateProfileBuilder(memory=FakeMemory(**kwargs)).build(OWNER)


# ═══════════════════════════════════════════════════════════════════════════
# 1. A claim cannot exist without evidence
# ═══════════════════════════════════════════════════════════════════════════

def test_a_skill_claim_requires_evidence():
    """
    The type refuses to represent an unsupported qualification. This is the
    anti-invention guarantee at its narrowest point — everything else in the
    package depends on it holding.
    """
    with pytest.raises(ValueError) as exc:
        SkillClaim(name="Kubernetes", evidence=())
    assert "no evidence" in str(exc.value)


def test_a_claim_with_evidence_is_fine():
    claim = SkillClaim(
        name="Python",
        evidence=(Evidence(EvidenceKind.RESUME_SKILLS, "chunk_1", "Python, SQL"),),
    )
    assert claim.strength > 0


def test_using_a_skill_outranks_listing_it():
    """A reviewer weights shipped work above a list. So does this."""
    listed = SkillClaim(
        "Docker", evidence=(Evidence(EvidenceKind.RESUME_SKILLS, "c1", "Docker"),)
    )
    used = SkillClaim(
        "Docker", evidence=(Evidence(EvidenceKind.RESUME_EXPERIENCE, "c2", "..."),)
    )
    assert used.strength > listed.strength
    assert used.demonstrated is True
    assert listed.demonstrated is False


def test_corroboration_saturates():
    """Three projects beat one; thirty do not beat three by ten times."""
    one = SkillClaim("Python", evidence=(
        Evidence(EvidenceKind.RESUME_PROJECT, "p1", "..."),
    ))
    many = SkillClaim("Python", evidence=tuple(
        Evidence(EvidenceKind.RESUME_PROJECT, f"p{i}", "...") for i in range(30)
    ))
    assert many.strength > one.strength
    assert many.strength <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Parsing real résumé skill lines
# ═══════════════════════════════════════════════════════════════════════════

def test_a_labelled_skill_line_splits_into_category_and_skills():
    category, entries = split_skill_line(
        "Programming Languages:C/C++, Python, JavaScript (ES6+), TypeScript, SQL"
    )
    assert category == "Programming Languages"
    assert entries == ["C/C++", "Python", "JavaScript (ES6+)", "TypeScript", "SQL"]


def test_a_parenthetical_is_not_split_on_its_inner_comma():
    """
    `AWS (EC2, S3)` is one skill. The previous splitter produced `AWS (EC2` and
    `S3)` — two tokens, neither of them a skill.
    """
    _, entries = split_skill_line("Cloud & DevOps:AWS (EC2, S3), Docker, CI/CD")
    assert entries == ["AWS (EC2, S3)", "Docker", "CI/CD"]


def test_slashed_names_stay_whole():
    """`C/C++` and `CI/CD` are single names, not two each."""
    _, entries = split_skill_line("Languages:C/C++, Python")
    assert "C/C++" in entries


def test_the_category_label_is_never_treated_as_a_skill():
    category, entries = split_skill_line("Tools:Git, Linux, Postman")
    assert category == "Tools"
    assert "Tools" not in entries


def test_a_parenthetical_is_dropped_when_normalising():
    assert normalise_surface("AWS (EC2, S3)") == "AWS"
    assert normalise_surface("JavaScript (ES6+)") == "JavaScript"


@pytest.mark.parametrize("surface,expected", [
    ("react.js", "React"), ("ReactJS", "React"), ("React", "React"),
    ("postgres", "PostgreSQL"), ("PostgreSQL", "PostgreSQL"),
    ("nodejs", "Node.js"), ("py", "Python"), ("k8s", "Kubernetes"),
    ("sklearn", "Scikit-learn"), ("golang", "Go"),
])
def test_aliases_resolve_to_one_canonical_name(surface, expected):
    """
    A résumé says React.js and a posting says ReactJS. Matching them as strings
    invents a gap, and a candidate who looks unqualified for a job they can do.
    """
    skill = canonical_for(surface)
    assert skill is not None and skill.canonical == expected


def test_an_unknown_token_resolves_to_nothing():
    assert canonical_for("Blockchain Wizardry") is None
    assert canonical_for("") is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. Detection in mangled PDF text
# ═══════════════════════════════════════════════════════════════════════════

def test_skills_are_found_in_glued_pdf_text():
    """
    The constraint that shaped the matcher. There is no word boundary before
    `FastAPI` in this line, and tokenising finds nothing at all.
    """
    text = "Built a multi-agent AI voice assistant usingFastAPIandLangGraph."
    found = {s.canonical for s, _ in find_skills(text)}
    assert "FastAPI" in found
    assert "LangGraph" in found


def test_skills_are_found_in_ordinary_prose():
    found = {s.canonical for s, _ in find_skills("Built the API with React and Docker")}
    assert {"React", "Docker"} <= found


@pytest.mark.parametrize("text", [
    "the team was going to refactor",      # Go
    "she reacted badly to the news",       # React
    "a rusty old algorithm",               # Rust
    "he said it would work",               # AI inside "said"
    "the car park",                        # R
])
def test_ordinary_words_do_not_produce_skills(text):
    """
    The adversarial half. Substring matching is what finds `FastAPI` in glued
    text and also what would find `Go` in `going` — short and word-like names
    are boundary-only for exactly this reason.
    """
    assert find_skills(text) == []


def test_a_lowercase_mention_inside_a_word_is_not_a_match():
    """
    Exact case is what makes the substring rule safe: `reacted` contains
    `react` but not `React`.
    """
    assert find_skills("reacted") == []


def test_longer_names_win_over_shorter_ones_they_contain():
    found = {s.canonical for s, _ in find_skills("We use Machine Learning daily")}
    assert "Machine Learning" in found


# ═══════════════════════════════════════════════════════════════════════════
# 4. Building the profile
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_profile_is_built_from_every_source():
    profile = await build()

    assert profile.owner_id == OWNER
    assert profile.name == "Vansh Pratap Singh"
    assert profile.skills
    assert profile.projects and profile.experience and profile.education
    assert profile.degraded is False
    assert profile.may_assert_gaps is True


async def test_listed_skills_become_claims_with_provenance():
    profile = await build()

    claim = profile.claim("PostgreSQL")
    assert claim is not None
    assert claim.name == "PostgreSQL"
    # Points at a real stored chunk, not an invented identifier.
    assert any(e.source_id == "chunk_1" for e in claim.evidence)
    assert any(e.kind is EvidenceKind.RESUME_SKILLS for e in claim.evidence)


async def test_project_usage_becomes_stronger_evidence_than_the_list():
    """
    FastAPI is both listed and used. The claim should carry both, and the
    project evidence should be what makes it `demonstrated`.
    """
    profile = await build()
    claim = profile.claim("FastAPI")

    kinds = {e.kind for e in claim.evidence}
    assert EvidenceKind.RESUME_SKILLS in kinds
    assert EvidenceKind.RESUME_PROJECT in kinds
    assert claim.demonstrated is True
    assert any(e.detail.startswith("Multi-Agent") for e in claim.evidence)


async def test_experience_evidence_points_at_the_role():
    profile = await build()
    claim = profile.claim("Scikit-learn")

    assert claim is not None
    assert any(e.kind is EvidenceKind.RESUME_EXPERIENCE for e in claim.evidence)
    assert claim.demonstrated is True


async def test_project_entries_carry_their_skills_and_identity():
    profile = await build()
    project = profile.projects[0]

    assert "My_Agent" in project.title
    assert project.entity_id == "resume_1_project_1"
    assert "FastAPI" in project.skills
    assert "PostgreSQL" in project.skills


async def test_education_is_parsed_including_cgpa():
    profile = await build()
    education = profile.education[0]

    assert "Rajiv Gandhi" in education.institution
    assert education.cgpa == "8.80"
    assert "B.Tech" in education.qualification


async def test_achievements_are_captured():
    profile = await build()
    assert any("Scholarship" in a for a in profile.achievements)


async def test_evidence_can_be_traced_back_for_any_claim():
    """Every claim must be answerable to 'why do you think I know this?'."""
    profile = await build()
    for claim in profile.skills.values():
        assert claim.evidence, claim.name
        for item in claim.evidence:
            assert item.source_id and item.source_id != "unknown"
            assert item.excerpt


# ═══════════════════════════════════════════════════════════════════════════
# 5. Never inventing
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_skill_absent_from_the_resume_is_absent_from_the_profile():
    profile = await build()

    for absent in ("Kubernetes", "Terraform", "Azure", "GraphQL", "Rust"):
        assert profile.has_skill(absent) is False, absent
        assert profile.evidence_for(absent) == ()


async def test_an_unknown_token_in_the_resume_does_not_become_a_skill():
    """
    The vocabulary is closed. A résumé listing something the system has no name
    for yields no claim, rather than a claim it cannot describe.
    """
    # Every other source emptied, so the only claims possible come from this
    # one line — otherwise the projects and roles contribute their own.
    profile = await build(
        skills=[_item("Skills\nTools:Blockchain Wizardry, Quantum Alchemy, Python")],
        projects=[], experience=[], education=[], achievements=[],
    )

    assert profile.has_skill("Python") is True
    assert profile.has_skill("Blockchain Wizardry") is False
    assert profile.has_skill("Quantum Alchemy") is False
    assert set(profile.skills) == {"python"}, "an unknown token became a claim"


async def test_an_empty_resume_produces_an_empty_profile_not_an_error():
    profile = await build(
        skills=[], projects=[], experience=[], education=[],
        achievements=[], name=[],
    )
    assert profile.is_empty is True
    assert profile.skills == {}
    # Nothing failed, so absence here is real and may be reported.
    assert profile.degraded is False
    assert profile.may_assert_gaps is True


# ═══════════════════════════════════════════════════════════════════════════
# 6. A failed lookup is not an empty résumé
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_failed_source_degrades_the_profile():
    """
    The answerability distinction, at profile level. After a broken lookup,
    "you don't know Docker" is a statement about the network.
    """
    profile = await build(fail={"skills"})

    assert profile.degraded is True
    assert profile.may_assert_gaps is False
    assert ProfileSource.SKILLS in profile.sources_failed
    assert ProfileSource.SKILLS not in profile.sources_loaded


async def test_a_failed_source_does_not_abort_the_rest():
    """A partial profile is useful; a missing one is not."""
    profile = await build(fail={"skills"})

    assert profile.projects, "projects should still have loaded"
    assert profile.education, "education should still have loaded"
    # And project usage still yields claims, from the sources that worked.
    assert profile.has_skill("FastAPI") is True


async def test_an_empty_source_is_not_a_failed_one():
    profile = await build(skills=[])

    assert profile.degraded is False
    assert profile.may_assert_gaps is True
    assert ProfileSource.SKILLS in profile.sources_loaded


async def test_every_source_failing_still_returns_a_profile():
    profile = await build(
        fail={"skills", "projects", "experience", "education", "achievements", "name"}
    )
    assert profile.is_empty is True
    assert profile.may_assert_gaps is False
    assert len(profile.sources_failed) == 6


async def test_a_raising_source_is_treated_as_a_failure_not_as_emptiness():
    class Exploding(FakeMemory):
        async def retrieve_skills(self, user_id, query=None, limit=10):
            raise RuntimeError("qdrant unreachable")

    profile = await CandidateProfileBuilder(memory=Exploding()).build(OWNER)

    assert profile.degraded is True
    assert profile.may_assert_gaps is False


# ═══════════════════════════════════════════════════════════════════════════
# 7. Queries used by the job agent
# ═══════════════════════════════════════════════════════════════════════════

async def test_demonstrated_skills_are_those_actually_used():
    profile = await build()
    demonstrated = {c.name for c in profile.demonstrated_skills()}

    assert "FastAPI" in demonstrated          # used in a project
    assert "Python" in demonstrated           # used in a role
    # Listed but never used in any project or role.
    assert "Postman" not in demonstrated


async def test_top_skills_are_ordered_by_strength():
    profile = await build()
    top = profile.top_skills(limit=5)
    strengths = [c.strength for c in top]

    assert strengths == sorted(strengths, reverse=True)
    assert len(top) <= 5


async def test_where_names_the_places_a_skill_appears():
    profile = await build()
    places = profile.claim("FastAPI").where()

    assert places
    assert any("My_Agent" in p for p in places)


async def test_the_summary_never_leaks_resume_content():
    profile = await build()
    rendered = str(profile.summary())

    assert "Vansh" not in rendered
    assert "FastAPI" not in rendered
    assert "skills" in rendered and "degraded" in rendered
