"""
Requirement extraction, evidence-based matching, and the refusal to invent.

The property under test is the one `app/candidate` established, carried across
the join where it would actually be spoken aloud:

    the assistant never claims a qualification the resume does not evidence

A wrong score is recoverable. Telling a recruiter the candidate has three years
of Kubernetes because the posting asked for three years of Kubernetes is not, so
most of what follows is about what the matcher *refuses* to say.

Three things are pinned throughout:

**Every positive verdict carries a real stored id.** Not a synthesised one —
the same `string_id` / `entity_id` the résumé ingestion wrote, so a claim can be
traced back to the chunk it came from.

**MISSING and UNKNOWN never collapse.** They both mean "found nothing" and they
license opposite sentences. The tests below drive a failed source, an empty
profile and a healthy-but-silent résumé through the same requirement and assert
three different verdicts.

**The score is arithmetic, not opinion.** It is asserted to exact values,
because a score nobody can reproduce is a score nobody can argue with.

The candidate fixtures are the real stored résumé text reused from
`test_candidate_profile.py` — glued PDF words and all — so the matcher is
exercised against the same strings production reads.
"""
from __future__ import annotations

import pytest

from app.candidate import CandidateProfileBuilder
from app.candidate.models import (
    CandidateProfile,
    Evidence,
    EvidenceKind,
)
from app.matching import (
    Band,
    MatchStatus,
    Necessity,
    RequirementKind,
    extract_requirements,
    match_requirements,
    render,
    render_summary,
)
from app.matching.models import Requirement, RequirementMatch
from app.matching.scoring import compute_score
from tests.support.fake_llm import final, tool_call
from tests.support.harness import OWNER, drive, state, stub_services
from tests.test_candidate_profile import FakeMemory, _item

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

JD = """Senior AI Engineer

Requirements:
- 3+ years of experience building production ML systems
- Strong Python and FastAPI
- Experience with PostgreSQL
- Bachelor's degree in Computer Science
- Kubernetes

Nice to have:
- ReactJS
- Terraform preferred
- Familiarity with LangGraph

Benefits:
- Free lunch, Docker-themed socks, and a Kubernetes cluster to play with
"""


async def profile(**kwargs) -> CandidateProfile:
    """The real builder over the real résumé fixtures."""
    return await CandidateProfileBuilder(memory=FakeMemory(**kwargs)).build(OWNER)


async def report(text: str = JD, *, title: str = "Senior AI Engineer", **kwargs):
    """Extract, match, and return the report — the whole pure pipeline."""
    return match_requirements(
        extract_requirements(text, title=title), await profile(**kwargs)
    )


def verdict(rep, name: str) -> RequirementMatch:
    """The match for one requirement, by canonical name."""
    found = next(
        (m for m in rep.matches if m.requirement.canonical.lower() == name.lower()),
        None,
    )
    assert found is not None, (
        f"{name!r} not in report: "
        f"{[m.requirement.canonical for m in rep.matches]}"
    )
    return found


# ═══════════════════════════════════════════════════════════════════════════
# 1. Requirement extraction — required vs preferred
# ═══════════════════════════════════════════════════════════════════════════

def test_a_requirements_heading_makes_its_items_required():
    reqs = extract_requirements(JD)
    required = {r.canonical for r in reqs.required}

    assert {"Python", "FastAPI", "PostgreSQL", "Kubernetes"} <= required


def test_a_nice_to_have_heading_makes_its_items_preferred():
    reqs = extract_requirements(JD)
    preferred = {r.canonical for r in reqs.preferred}

    assert "React" in preferred
    assert "LangGraph" in preferred
    assert "React" not in {r.canonical for r in reqs.required}


def test_a_line_marker_overrides_its_section():
    """`Terraform preferred` sits under "Nice to have" and also says so."""
    reqs = extract_requirements(JD)
    terraform = next(r for r in reqs if r.canonical == "Terraform")
    assert terraform.necessity is Necessity.PREFERRED


def test_preferred_beats_required_inside_one_line():
    """
    "Strong Kubernetes preferred" contains both markers. Reading it as a bar
    would invent a gap the posting did not create.
    """
    reqs = extract_requirements("Strong Kubernetes experience preferred")
    assert next(iter(reqs)).necessity is Necessity.PREFERRED


def test_an_unlabelled_line_is_required_by_default():
    """The conservative reading: an unmarked bar is still a bar."""
    reqs = extract_requirements("We are looking for someone who writes Python.")
    assert next(r for r in reqs if r.canonical == "Python").necessity is (
        Necessity.REQUIRED
    )


def test_a_benefits_section_produces_no_requirements():
    """
    A perks list mentioning Kubernetes is not asking for Kubernetes. Without
    the suppression this posting would demand Docker because of the socks.
    """
    reqs = extract_requirements(JD)
    line_sources = " ".join(r.source_line for r in reqs).lower()
    assert "socks" not in line_sources
    assert "Docker" not in {r.canonical for r in reqs}


def test_the_strictest_necessity_wins_a_duplicate():
    text = (
        "Requirements:\n- Python\n\nNice to have:\n- Python at scale\n"
    )
    reqs = extract_requirements(text)
    pythons = [r for r in reqs if r.canonical == "Python"]
    assert len(pythons) == 1
    assert pythons[0].necessity is Necessity.REQUIRED


# ═══════════════════════════════════════════════════════════════════════════
# 2. Extraction — the canonical vocabulary is shared with the résumé
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("written,canonical", [
    ("ReactJS", "React"), ("react.js", "React"),
    ("postgres", "PostgreSQL"), ("k8s", "Kubernetes"),
    ("nodejs", "Node.js"), ("sklearn", "Scikit-learn"),
])
def test_posting_aliases_resolve_to_the_same_canonical_name(written, canonical):
    """
    The reason a résumé saying React.js matches a posting saying ReactJS. Both
    sides go through `app.candidate.vocabulary`, so there is one spelling.
    """
    reqs = extract_requirements(f"Requirements:\n- Experience with {written}")
    assert canonical in {r.canonical for r in reqs}


def test_an_unknown_technology_does_not_become_a_requirement():
    """
    The vocabulary is closed on the posting side too. Inventing a requirement
    invents a gap, which is the same lie pointed the other way.
    """
    reqs = extract_requirements(
        "Requirements:\n- Deep expertise in Blockchain Wizardry and Python"
    )
    names = {r.canonical for r in reqs if r.kind is RequirementKind.SKILL}
    assert names == {"Python"}


def test_years_of_experience_is_extracted_with_its_minimum():
    reqs = extract_requirements("Requirements:\n- 5+ years of backend experience")
    years = next(r for r in reqs if r.kind is RequirementKind.EXPERIENCE_YEARS)
    assert years.min_years == 5


@pytest.mark.parametrize("text,expected", [
    ("at least 2 years of experience", 2),
    ("3+ years building systems", 3),
    ("minimum 4 years in the field", 4),
    ("2-4 years of relevant work", 2),
])
def test_experience_phrasings_all_yield_a_minimum(text, expected):
    reqs = extract_requirements(f"Requirements:\n- {text}")
    years = next(r for r in reqs if r.kind is RequirementKind.EXPERIENCE_YEARS)
    assert years.min_years == expected


def test_a_degree_requirement_carries_its_level_and_field():
    reqs = extract_requirements(
        "Requirements:\n- Bachelor's degree in Computer Science"
    )
    degree = next(r for r in reqs if r.kind is RequirementKind.EDUCATION)
    assert degree.degree_level == 2
    assert "Computer Science" in degree.field_of_study


def test_a_masters_requirement_outranks_a_bachelors():
    reqs = extract_requirements("Requirements:\n- Master's degree required")
    degree = next(r for r in reqs if r.kind is RequirementKind.EDUCATION)
    assert degree.degree_level == 3


def test_a_domain_phrase_is_extracted():
    reqs = extract_requirements(
        "Requirements:\n- Experience running services in production"
    )
    assert "Production systems" in {
        r.canonical for r in reqs if r.kind is RequirementKind.DOMAIN
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Malformed and empty postings
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", None])
def test_an_empty_posting_yields_no_requirements(text):
    reqs = extract_requirements(text or "")
    assert len(reqs) == 0
    assert bool(reqs) is False


def test_a_posting_with_no_recognisable_content_yields_nothing():
    reqs = extract_requirements("!!! ??? ... ***")
    assert len(reqs) == 0


async def test_an_empty_posting_produces_an_unknown_report_not_a_zero():
    """
    Reading nothing out of a posting is not a finding about the candidate.
    Scoring it 0% would say the opposite.
    """
    rep = await report("")
    assert rep.matches == ()
    assert rep.band is Band.UNKNOWN
    assert "No requirements could be read" in render(rep)


async def test_a_run_on_snippet_still_splits_into_requirements():
    """
    Job search returns a 300-character snippet with no newlines. Sentence
    splitting is the only structure available there.
    """
    rep = await report(
        "AI Engineer role. Requires Python and FastAPI. Kubernetes is a plus.",
        title="AI Engineer",
    )
    assert verdict(rep, "Python").requirement.necessity is Necessity.REQUIRED
    assert verdict(rep, "Kubernetes").requirement.necessity is Necessity.PREFERRED


# ═══════════════════════════════════════════════════════════════════════════
# 4. Matching — evidence is mandatory for any positive claim
# ═══════════════════════════════════════════════════════════════════════════

def test_a_positive_verdict_without_evidence_cannot_be_constructed():
    """
    The anti-invention guarantee at its narrowest point. Everything else in
    this file depends on this holding.
    """
    requirement = Requirement(
        kind=RequirementKind.SKILL, canonical="Kubernetes",
        necessity=Necessity.REQUIRED,
    )
    for status in (MatchStatus.MATCHED, MatchStatus.PARTIAL):
        with pytest.raises(ValueError) as exc:
            RequirementMatch(requirement=requirement, status=status)
        assert "no evidence" in str(exc.value)


def test_a_claimed_absence_carrying_evidence_cannot_be_constructed():
    """The contradiction pointed the other way: 'you lack X, here is your X'."""
    requirement = Requirement(
        kind=RequirementKind.SKILL, canonical="Python",
        necessity=Necessity.REQUIRED,
    )
    with pytest.raises(ValueError):
        RequirementMatch(
            requirement=requirement,
            status=MatchStatus.MISSING,
            evidence=(Evidence(EvidenceKind.RESUME_SKILLS, "chunk_1", "Python"),),
        )


async def test_every_positive_verdict_carries_a_real_stored_id():
    """Any claim must be answerable to 'which chunk of my résumé says that?'."""
    rep = await report()
    for match in rep.matches:
        if not match.status.is_positive:
            continue
        assert match.evidence, match.requirement.canonical
        for item in match.evidence:
            assert item.source_id and item.source_id != "unknown"
            assert item.excerpt


async def test_a_gap_never_carries_evidence():
    rep = await report()
    for match in rep.missing:
        assert match.evidence == ()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Matched, partial, missing
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_skill_used_in_a_project_is_matched():
    rep = await report()
    fastapi = verdict(rep, "FastAPI")

    assert fastapi.status is MatchStatus.MATCHED
    assert any(e.source_id == "resume_1_project_1" for e in fastapi.evidence)
    assert "My_Agent" in fastapi.rationale


async def test_a_skill_used_in_a_role_is_matched():
    rep = await report()
    python = verdict(rep, "Python")

    assert python.status is MatchStatus.MATCHED
    assert any(e.kind is EvidenceKind.RESUME_EXPERIENCE for e in python.evidence)


async def test_a_skill_only_listed_is_partial_not_matched():
    """
    "Not demonstrated" is a different fact from "not found", and a reviewer
    treats them differently. Postman is on the résumé and in no project.
    """
    rep = await report("Requirements:\n- Postman")
    postman = verdict(rep, "Postman")

    assert postman.status is MatchStatus.PARTIAL
    assert postman.evidence
    assert "not evidenced in a project or role" in postman.rationale


async def test_a_skill_absent_from_the_resume_is_missing():
    rep = await report()
    kubernetes = verdict(rep, "Kubernetes")

    assert kubernetes.status is MatchStatus.MISSING
    assert kubernetes.evidence == ()
    assert "no skill, project or role" in kubernetes.rationale


async def test_an_alias_in_the_posting_matches_the_resume_spelling():
    """A posting saying `postgres` meets a résumé saying `PostgreSQL`."""
    rep = await report("Requirements:\n- Deep experience with postgres")
    assert verdict(rep, "PostgreSQL").status is MatchStatus.MATCHED


# ═══════════════════════════════════════════════════════════════════════════
# 6. Experience and education
# ═══════════════════════════════════════════════════════════════════════════

async def test_years_of_experience_is_never_claimed_as_matched():
    """
    The builder says outright that dates in parsed PDFs are unreliable, so no
    duration is stored. Answering "yes, 3 years" would invent the one number
    the requirement is about.
    """
    rep = await report()
    years = next(
        m for m in rep.matches
        if m.requirement.kind is RequirementKind.EXPERIENCE_YEARS
    )
    assert years.status is MatchStatus.PARTIAL
    assert "does not record durations" in years.rationale
    assert years.evidence


async def test_years_of_experience_with_no_roles_on_file_is_missing():
    rep = await report(
        "Requirements:\n- 3+ years of experience", experience=[]
    )
    years = next(
        m for m in rep.matches
        if m.requirement.kind is RequirementKind.EXPERIENCE_YEARS
    )
    assert years.status is MatchStatus.MISSING


async def test_a_degree_at_or_above_the_required_level_is_matched():
    """B.Tech is level 2; a Bachelor's requirement is level 2."""
    rep = await report("Requirements:\n- Bachelor's degree required")
    degree = next(
        m for m in rep.matches if m.requirement.kind is RequirementKind.EDUCATION
    )
    assert degree.status is MatchStatus.MATCHED
    assert any(e.source_id == "chunk_2" for e in degree.evidence)


async def test_a_degree_below_the_required_level_is_partial():
    rep = await report("Requirements:\n- PhD required")
    degree = next(
        m for m in rep.matches if m.requirement.kind is RequirementKind.EDUCATION
    )
    assert degree.status is MatchStatus.PARTIAL
    assert "below the level" in degree.rationale


async def test_a_field_mismatch_is_noted_without_denying_the_degree():
    rep = await report("Requirements:\n- Bachelor's degree in Mechanical Engineering")
    degree = next(
        m for m in rep.matches if m.requirement.kind is RequirementKind.EDUCATION
    )
    assert degree.status is MatchStatus.MATCHED
    assert "different field" in degree.rationale


async def test_a_domain_requirement_is_capped_at_partial():
    """
    A word in a project description is real evidence and weaker than the
    résumé naming a skill. It can support a partial claim and never a full one.
    """
    rep = await report("Requirements:\n- Experience with research")
    domain = next(
        m for m in rep.matches if m.requirement.kind is RequirementKind.DOMAIN
    )
    assert domain.status is MatchStatus.PARTIAL
    assert domain.evidence


async def test_a_responsibility_is_reported_as_unchecked_not_guessed():
    rep = await report(
        "Requirements:\n- Ability to collaborate with cross functional partners"
    )
    duty = next(
        m for m in rep.matches
        if m.requirement.kind is RequirementKind.RESPONSIBILITY
    )
    assert duty.status is MatchStatus.UNKNOWN
    assert duty.scorable is False


# ═══════════════════════════════════════════════════════════════════════════
# 7. A failed lookup is not a gap — the answerability rule
# ═══════════════════════════════════════════════════════════════════════════

async def test_a_failed_source_turns_gaps_into_unknowns():
    """
    After a broken lookup, "you don't know Kubernetes" is a statement about
    the network. This is `may_assert_gaps` reaching the match table.
    """
    rep = await report(fail={"skills", "projects", "experience"})

    assert rep.profile_degraded is True
    assert rep.may_assert_gaps is False
    assert rep.missing == (), "a degraded profile must not assert any gap"
    kubernetes = verdict(rep, "Kubernetes")
    assert kubernetes.status is MatchStatus.UNKNOWN
    assert "lookup failed" in kubernetes.rationale


async def test_a_healthy_but_silent_resume_does_assert_the_gap():
    """The other side of the same rule: real absence is reportable."""
    rep = await report()
    assert rep.may_assert_gaps is True
    assert verdict(rep, "Kubernetes").status is MatchStatus.MISSING


async def test_no_profile_on_file_is_unknown_not_missing():
    """
    A user who never uploaded a résumé produces an empty profile through a
    *healthy* lookup. Reporting every requirement as missing would tell them
    they lack skills nothing ever looked for.
    """
    rep = await report(
        skills=[], projects=[], experience=[], education=[],
        achievements=[], name=[],
    )
    assert rep.profile_empty is True
    assert rep.missing == ()
    assert all(m.status is MatchStatus.UNKNOWN for m in rep.matches)


async def test_the_three_no_evidence_cases_are_three_different_verdicts():
    """
    Failed lookup, no profile, and a genuinely silent résumé all produce an
    empty search and must not produce the same sentence.
    """
    text = "Requirements:\n- Kubernetes"

    healthy = verdict(await report(text), "Kubernetes")
    degraded = verdict(await report(text, fail={"skills"}), "Kubernetes")
    empty = verdict(
        await report(
            text, skills=[], projects=[], experience=[], education=[],
            achievements=[], name=[],
        ),
        "Kubernetes",
    )

    assert healthy.status is MatchStatus.MISSING
    assert degraded.status is MatchStatus.UNKNOWN
    assert empty.status is MatchStatus.UNKNOWN
    assert healthy.rationale != degraded.rationale != empty.rationale


async def test_a_degraded_report_says_so_in_its_explanation():
    rendered = render(await report(fail={"skills"}))
    assert "could not be loaded" in rendered
    assert "statement that you lack a skill" in rendered


async def test_a_partial_outage_never_produces_a_flattering_verdict():
    """
    The regression this rule was written for. Excluding UNKNOWN from the score
    is correct, but it also *deletes the gaps* — so the mean is taken over the
    survivors, which are disproportionately the things that matched. With the
    skills lookup failing, a profile that scores 61% healthy came back 93% and
    STRONG, because every gap had turned into an unknown and dropped out.

    An upper bound presented as a verdict is the flattering direction of a
    wrong answer, which is the one direction this package must never fail in.
    """
    healthy = await report()
    degraded = await report(fail={"skills"})

    assert healthy.band is not Band.UNKNOWN
    assert degraded.band is Band.UNKNOWN
    assert degraded.score >= healthy.score, (
        "the arithmetic genuinely does inflate — the band is what must refuse"
    )
    assert "failed to load" in degraded.unscorable_reason
    # And the number is never presented as a verdict.
    assert "%" not in render_summary(degraded)
    assert f"{degraded.score:.0%}" not in render(degraded).splitlines()[0]


async def test_an_outage_that_cost_no_check_still_yields_a_verdict():
    """
    Degradation only suppresses the band when it actually removed a check. A
    failed achievements lookup that changed nothing must not erase an
    otherwise sound answer.
    """
    rep = await report("Requirements:\n- Python\n- FastAPI", fail={"achievements"})

    assert rep.profile_degraded is True
    assert rep.unknown == ()
    assert rep.band is not Band.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# 8. Scoring — deterministic and required-weighted
# ═══════════════════════════════════════════════════════════════════════════

def _match(name, necessity, status, *, kind=RequirementKind.SKILL):
    """A synthetic verdict, for scoring arithmetic in isolation."""
    requirement = Requirement(kind=kind, canonical=name, necessity=necessity)
    evidence = (
        (Evidence(EvidenceKind.RESUME_PROJECT, f"src_{name}", "..."),)
        if status.is_positive else ()
    )
    return RequirementMatch(
        requirement=requirement, status=status, evidence=evidence
    )


def test_the_score_is_a_weighted_average_of_the_two_coverages():
    matches = [
        _match("A", Necessity.REQUIRED, MatchStatus.MATCHED),
        _match("B", Necessity.REQUIRED, MatchStatus.MISSING),
        _match("C", Necessity.PREFERRED, MatchStatus.MATCHED),
    ]
    score, required, preferred = compute_score(matches)

    assert required == 0.5
    assert preferred == 1.0
    # 0.8 * 0.5 + 0.2 * 1.0
    assert score == 0.6


def test_a_missing_required_costs_four_times_a_missing_preferred():
    """Required outranks preferred by construction, not by prompt."""
    required_gap = [
        _match("A", Necessity.REQUIRED, MatchStatus.MISSING),
        _match("B", Necessity.PREFERRED, MatchStatus.MATCHED),
    ]
    preferred_gap = [
        _match("A", Necessity.REQUIRED, MatchStatus.MATCHED),
        _match("B", Necessity.PREFERRED, MatchStatus.MISSING),
    ]
    assert compute_score(required_gap)[0] == 0.2
    assert compute_score(preferred_gap)[0] == 0.8


def test_a_partial_is_worth_exactly_half_a_match():
    full = [_match("A", Necessity.REQUIRED, MatchStatus.MATCHED)]
    half = [_match("A", Necessity.REQUIRED, MatchStatus.PARTIAL)]
    assert compute_score(full)[0] == 1.0
    assert compute_score(half)[0] == 0.5


def test_an_unknown_is_excluded_rather_than_scored_as_zero():
    """
    The answerability rule in arithmetic form. Scoring an unchecked requirement
    as zero would make a Qdrant outage look like an unqualified candidate.
    """
    without = [_match("A", Necessity.REQUIRED, MatchStatus.MATCHED)]
    with_unknown = [
        _match("A", Necessity.REQUIRED, MatchStatus.MATCHED),
        _match("B", Necessity.REQUIRED, MatchStatus.UNKNOWN),
    ]
    assert compute_score(without)[0] == compute_score(with_unknown)[0] == 1.0


def test_weights_renormalise_when_a_posting_has_only_required_items():
    matches = [_match("A", Necessity.REQUIRED, MatchStatus.MATCHED)]
    score, required, preferred = compute_score(matches)
    assert score == 1.0 and required == 1.0 and preferred is None


def test_the_score_is_reproducible():
    matches = [
        _match("A", Necessity.REQUIRED, MatchStatus.MATCHED),
        _match("B", Necessity.PREFERRED, MatchStatus.PARTIAL),
    ]
    assert compute_score(matches) == compute_score(list(reversed(matches)))


async def test_the_same_inputs_always_produce_the_same_report():
    first, second = await report(), await report()
    assert first.score == second.score
    assert first.band is second.band
    assert [m.summary() for m in first.matches] == [
        m.summary() for m in second.matches
    ]
    assert render(first) == render(second)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Bands — a failed bar caps the verdict
# ═══════════════════════════════════════════════════════════════════════════

def test_a_missing_required_requirement_caps_the_band_below_strong():
    """
    Nine matches and one failed bar is not a strong match, whatever the
    weighted average works out to.
    """
    from app.matching.scoring import compute_band

    matches = [
        _match(f"S{i}", Necessity.REQUIRED, MatchStatus.MATCHED)
        for i in range(9)
    ] + [_match("Gap", Necessity.REQUIRED, MatchStatus.MISSING)]

    score, required, _ = compute_score(matches)
    band, reason = compute_band(matches, score, required_coverage=required)

    assert score >= 0.80
    assert band is Band.GOOD
    assert "capped" in reason


def test_a_clean_sweep_is_strong():
    from app.matching.scoring import compute_band

    matches = [_match("A", Necessity.REQUIRED, MatchStatus.MATCHED)]
    score, required, _ = compute_score(matches)
    assert compute_band(matches, score, required_coverage=required)[0] is Band.STRONG


def test_mostly_unknown_required_items_yield_no_verdict_at_all():
    from app.matching.scoring import compute_band

    matches = [
        _match("A", Necessity.REQUIRED, MatchStatus.MATCHED),
        _match("B", Necessity.REQUIRED, MatchStatus.UNKNOWN),
        _match("C", Necessity.REQUIRED, MatchStatus.UNKNOWN),
    ]
    score, required, _ = compute_score(matches)
    band, reason = compute_band(matches, score, required_coverage=required)

    assert band is Band.UNKNOWN
    assert "could not be checked" in reason


async def test_a_degraded_profile_does_not_produce_a_confident_band():
    rep = await report(fail={"skills", "projects", "experience", "education"})
    assert rep.band is Band.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# 10. Explanation — rendered from the table, never generated
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_explanation_cites_a_source_for_every_positive_row():
    rendered = render(await report())
    for line in rendered.splitlines():
        if line.strip().startswith("- ") and "used in" in line:
            assert "(source:" in line, line


async def test_the_explanation_separates_not_found_from_not_demonstrated():
    rep = await report("Requirements:\n- Postman\n- Kubernetes")
    rendered = render(rep)

    assert "Partial evidence:" in rendered
    assert "No evidence found:" in rendered
    assert "not evidenced in a project or role" in rendered
    assert "no skill, project or role" in rendered


async def test_the_explanation_names_the_strongest_evidence():
    rendered = render(await report())
    assert "Strongest evidence:" in rendered


async def test_a_cited_source_id_is_never_truncated():
    """
    Real ingestion ids are longer than the display cap used to be, so
    `skills_resume_vansh_1a36dcef_chunk_1` rendered as
    `skills_resume_vansh_1a36dcef_chu` — an identifier that looks real, cannot
    be looked up, and is indistinguishable from a genuine one. A live run
    against the real store caught it.
    """
    long_id = "skills_resume_vansh_1a36dcef_chunk_1"
    memory = FakeMemory(
        projects=[_item(
            "TRACE\nBuilt with FastAPI.",
            string_id=long_id, entity_id=long_id, title="TRACE",
        )],
        skills=[], experience=[], education=[], achievements=[],
    )
    rep = match_requirements(
        extract_requirements("Requirements:\n- FastAPI"),
        await CandidateProfileBuilder(memory=memory).build(OWNER),
    )
    rendered = render(rep)

    assert long_id in rendered
    assert "chunk_1" in rendered
    # Every id the report leans on appears whole in the prose.
    for source_id in rep.evidence_ids():
        assert source_id in rendered


async def test_the_explanation_states_unmet_required_items_explicitly():
    rendered = render(await report())
    assert "Required but not evidenced:" in rendered
    assert "Kubernetes" in rendered
    assert "I have not claimed otherwise" in rendered


async def test_a_degraded_report_never_lists_a_requirement_as_not_found():
    rendered = render(await report(fail={"skills", "projects"}))
    assert "No evidence found:" not in rendered
    assert "Required but not evidenced:" not in rendered


async def test_truncation_drops_rows_not_characters():
    """
    A cut mid-string can sever a source id and emit an identifier that looks
    real. Truncation is row-granular and says how much it dropped.
    """
    rendered = render(await report(), limit=600)

    assert len(rendered) <= 600
    assert "further requirement(s) not shown" in rendered
    # The conclusions survive: a truncated report must never read as a better
    # match than it is.
    assert "Required but not evidenced:" in rendered


async def test_truncation_never_leaves_an_empty_section_heading():
    for limit in range(300, 2400, 100):
        rendered = render(await report(), limit=limit)
        lines = rendered.splitlines()
        for index, line in enumerate(lines):
            if line.endswith(":") and line in (
                "Evidenced:", "Partial evidence:", "No evidence found:",
                "Not checked:",
            ):
                following = lines[index + 1] if index + 1 < len(lines) else ""
                assert following.strip().startswith("- "), (
                    f"orphan heading {line!r} at limit={limit}"
                )


async def test_the_rendered_report_is_plain_ascii():
    """
    It travels through `ToolResult.observation`, which escapes non-ASCII to
    six characters apiece, and it is read aloud on the voice path.
    """
    rendered = render(await report())
    non_ascii = {c for c in rendered if ord(c) > 127}
    # Résumé excerpts may legitimately carry any character; the template must
    # not add its own.
    assert not (non_ascii - set("–—’é")), non_ascii


async def test_the_summary_reports_coverage_rather_than_judgement():
    summary = render_summary(await report())
    assert "%" in summary
    assert "evidenced in your profile" in summary
    assert "Not evidenced, and required:" in summary


async def test_the_summary_of_an_empty_profile_blames_nothing_on_the_user():
    summary = render_summary(await report(
        skills=[], projects=[], experience=[], education=[],
        achievements=[], name=[],
    ))
    assert "no resume or profile data on file" in summary.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 11. The job agent, end to end
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def agent():
    from app.agents.job_agent import job_agent
    return job_agent


@pytest.fixture
def services(monkeypatch):
    return stub_services(monkeypatch)


def _resume_memory(monkeypatch, **kwargs):
    """Point the profile builder at the real résumé fixtures."""
    from app.candidate import builder as builder_module

    memory = FakeMemory(**kwargs)
    monkeypatch.setattr(
        builder_module.candidate_profile_builder, "_memory", memory, raising=False
    )
    return memory


async def test_match_job_is_a_read_tool(agent, services):
    from tests.support.harness import capture_registry
    from app.tools.contract import Effect

    tools = await capture_registry(agent)
    assert tools["match_job"]["effect"] is Effect.READ
    assert "preview" not in tools["match_job"], (
        "a READ tool must not be gated — it changes nothing"
    )


async def test_the_agent_answers_a_match_question_from_the_report(
    agent, services, monkeypatch
):
    """
    The end-to-end path: model decides to call match_job, the matcher produces
    the verdict, and the *rendered* report becomes the answer.
    """
    _resume_memory(monkeypatch)

    result, llm = await drive(
        agent,
        [
            tool_call("match_job", job_description=JD, title="Senior AI Engineer"),
            final("Here is a lovely summary."),
        ],
        state("How well do I match this Senior AI Engineer job?"),
    )

    envelope = result["task_result"]
    content = envelope["result"]["content"]

    assert envelope["status"] == "success"
    assert "match_job" in envelope["evidence"]
    # The model's prose is discarded in favour of the rendered table.
    assert "lovely summary" not in content
    assert "Strongest evidence:" in content
    assert "(source: chunk_1" in content or "(source: chunk_" in content


async def test_the_envelope_carries_the_structured_verdict(
    agent, services, monkeypatch
):
    _resume_memory(monkeypatch)

    result, _ = await drive(
        agent,
        [tool_call("match_job", job_description=JD), final("done")],
        state("do I match this job"),
    )

    match = result["task_result"]["result"]["match"]
    assert 0.0 <= match["score"] <= 1.0
    assert match["band"] in {b.value for b in Band}
    assert match["evidence_ids"], "the verdict must name the chunks it used"
    assert any(
        r["status"] == "missing" and r["requirement"] == "Kubernetes"
        for r in match["requirements"]
    )


async def test_the_agent_never_claims_a_skill_the_resume_lacks(
    agent, services, monkeypatch
):
    """
    The adversarial case, and the reason the answer is rendered rather than
    generated. The model here does its best to assert Kubernetes; the answer
    the user receives says the opposite, because the model is not the author.
    """
    _resume_memory(monkeypatch)

    result, _ = await drive(
        agent,
        [
            tool_call("match_job", job_description=JD),
            final(
                "You are a perfect fit! You have 5 years of Kubernetes and "
                "Terraform in production."
            ),
        ],
        state("am I a good fit for this role"),
    )

    content = result["task_result"]["result"]["content"]
    assert "perfect fit" not in content
    assert "5 years of Kubernetes" not in content
    assert "Kubernetes" in content  # named, as a gap
    assert "Required but not evidenced" in content


async def test_candidate_data_in_the_tool_arguments_is_ignored(
    agent, services, monkeypatch
):
    """
    No second source of truth. A model that puts a skill list in its arguments
    cannot thereby introduce a qualification — the tool reads the profile.
    """
    _resume_memory(monkeypatch)

    result, _ = await drive(
        agent,
        [
            tool_call(
                "match_job",
                job_description="Requirements:\n- Kubernetes",
                user_skills=["Kubernetes", "Terraform"],
                candidate_has=["Kubernetes"],
            ),
            final("done"),
        ],
        state("do I match this"),
    )

    match = result["task_result"]["result"]["match"]
    kubernetes = next(
        r for r in match["requirements"] if r["requirement"] == "Kubernetes"
    )
    assert kubernetes["status"] == "missing"


async def test_a_match_question_without_a_posting_asks_for_one(
    agent, services, monkeypatch
):
    _resume_memory(monkeypatch)

    result, _ = await drive(
        agent,
        [
            tool_call("match_job"),
            final("I need the job description to compare against."),
        ],
        state("how well do I match this job"),
    )

    envelope = result["task_result"]
    # No report was produced, so the model's own reply stands.
    assert "match" not in envelope["result"]
    assert result["answerability"] == "NO_DATA"


async def test_a_job_from_an_earlier_search_can_be_matched_by_index(
    agent, monkeypatch
):
    """
    "How well do I match the second one?" — the posting is reused from the
    search results rather than re-typed by the model.
    """
    async def _search(*a, **kw):
        return {
            "tool": "job_search", "success": True, "user_skills": [],
            "total_candidates": 2, "total_filtered": 2,
            "results": [
                {"title": "Frontend Engineer", "url": "https://x/1",
                 "snippet": "Requirements: React and TypeScript."},
                {"title": "AI Engineer", "url": "https://x/2",
                 "snippet": "Requirements: Python, FastAPI and Kubernetes."},
            ],
        }

    stub_services(monkeypatch, search_jobs=_search)
    _resume_memory(monkeypatch)

    result, _ = await drive(
        agent,
        [
            tool_call("job_search", query="engineer"),
            tool_call("match_job", job_index=2),
            final("done"),
        ],
        state("how well do I match the second one"),
    )

    match = result["task_result"]["result"]["match"]
    assert match["title"] == "AI Engineer"
    statuses = {r["requirement"]: r["status"] for r in match["requirements"]}
    assert statuses["Python"] == "matched"
    assert statuses["Kubernetes"] == "missing"


async def test_several_results_are_never_guessed_between(agent, monkeypatch):
    """
    Silently matching against the wrong posting produces a confident,
    thoroughly evidenced answer to a question nobody asked.
    """
    async def _search(*a, **kw):
        return {
            "tool": "job_search", "success": True, "user_skills": [],
            "total_candidates": 2, "total_filtered": 2,
            "results": [
                {"title": "A", "url": "https://x/1", "snippet": "Python."},
                {"title": "B", "url": "https://x/2", "snippet": "Rust."},
            ],
        }

    stub_services(monkeypatch, search_jobs=_search)
    _resume_memory(monkeypatch)

    result, _ = await drive(
        agent,
        [
            tool_call("job_search", query="engineer"),
            tool_call("match_job"),          # no index, two candidates
            final("Which posting did you mean?"),
        ],
        state("do I match it"),
    )

    assert "match" not in result["task_result"]["result"]


async def test_a_degraded_profile_reaches_the_agents_answer(
    agent, services, monkeypatch
):
    """The honesty rule survives the whole stack, not just the matcher."""
    _resume_memory(monkeypatch, fail={"skills", "projects", "experience"})

    result, _ = await drive(
        agent,
        [tool_call("match_job", job_description=JD), final("done")],
        state("how well do I match this job"),
    )

    content = result["task_result"]["result"]["content"]
    assert "could not be loaded" in content
    assert "Required but not evidenced" not in content
    assert result["task_result"]["result"]["match"]["degraded"] is True


async def test_the_observation_the_model_sees_survives_truncation(
    agent, services, monkeypatch
):
    """
    `ToolResult.observation` truncates at 1200 characters and escapes
    non-ASCII. The report is fitted below that so a cut cannot sever a source
    id and emit an identifier that looks real.
    """
    from app.tools.contract import Effect, coerce
    from tests.support.harness import capture_registry

    _resume_memory(monkeypatch)
    tools = await capture_registry(agent)

    raw = await tools["match_job"]["callable"]({"job_description": JD})
    observation = coerce(raw, tool="match_job",
                         declared_effect=Effect.READ).observation()

    assert not observation.endswith("...")
    assert len(observation) <= 1200


# ═══════════════════════════════════════════════════════════════════════════
# 12. Routing — the question reaches the agent that can answer it
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query", [
    "How well do I match this AI Engineer job?",
    "am I a good fit for this role",
    "do I match this job description",
    "do I qualify for this position",
    "what are my chances for this opening",
    "should I apply for this one",
])
def test_a_match_question_reaches_the_job_agent(query):
    """
    Grammatically these are questions about the user, so `profile_intent`
    labels them PROFILE_EXPERIENCE and routing sent them to an agent holding
    résumé tools and no posting — leaving the model to supply the comparison.
    """
    from app.agents import query_intent
    from app.memory.sources import QueryCategory

    decision = query_intent.classify(query, has_context=True)
    assert decision.category is QueryCategory.JOB_MATCH
    assert query_intent.agent_for(decision, "profile") == "job"


@pytest.mark.parametrize("query,expected", [
    ("what is my CGPA", "PROFILE_EDUCATION"),
    ("tell me about my projects", "PROFILE_PROJECTS"),
    ("what skills do I have", "PROFILE_SKILLS"),
    ("where did I intern", "PROFILE_EXPERIENCE"),
    ("what did I just tell you", "CONVERSATION_CURRENT"),
    ("does this fit in my schedule", "SCHEDULE_TEMPORAL"),
    ("what is my timetable today", "SCHEDULE_TEMPORAL"),
])
def test_existing_routing_is_unchanged(query, expected):
    """Regression cover: the new category must claim only what it should."""
    from app.agents import query_intent

    decision = query_intent.classify(query, has_context=True)
    assert decision.category.value == expected


def test_a_match_question_is_never_a_clarification_case():
    """
    "Which job do you mean?" in place of an answer is the failure the
    answer-first policy exists to prevent.
    """
    from app.agents import query_intent
    from app.memory.sources import QueryCategory, may_clarify

    decision = query_intent.classify("am I a good fit for this role")
    assert decision.may_clarify is False
    assert may_clarify(QueryCategory.JOB_MATCH) is False


# ═══════════════════════════════════════════════════════════════════════════
# 13. The rendered answer survives the rest of the workflow
# ═══════════════════════════════════════════════════════════════════════════

def _reflect_state(category, plan):
    from tests.support.harness import state as base_state

    s = base_state("how well do I match this job")
    s.update({
        "query_category": category,
        "execution_plan": plan,
        "current_step_index": 0,
        "selected_agent": plan[0]["agent"] if plan else "job",
        "task_result": {
            "agent": "job", "status": "success", "confidence": 0.9,
            "result": {"content": "Moderate match: 59% ..."},
            "evidence": ["match_job"], "next_actions": [],
            "goal": "g", "inputs": {}, "constraints": {}, "task_id": "t",
        },
    })
    return s


async def test_a_match_turn_is_never_replanned_into_a_second_step():
    """
    The regression a live run found. Routing sends a JOB_MATCH turn to the job
    agent, which renders the evidence-backed report — and then the planner's
    two-step plan advanced to the profile agent, whose envelope *replaced* it
    with free prose. The rendered answer, its source ids and its structured
    verdict were all discarded.

    The planner cannot prevent this: it never learns that routing overrode its
    agent choice, so it keeps proposing a follow-up step.
    """
    from app.agents.workflow import reflect_node

    plan = [
        {"step": 1, "agent": "job", "goal": "match the posting"},
        {"step": 2, "agent": "profile", "goal": "summarise the comparison"},
    ]
    result = await reflect_node(_reflect_state("JOB_MATCH", plan))

    assert result["reflect_outcome"] == "done"
    assert result["current_step_index"] == 0
    assert result["selected_agent"] == "job"
    assert result["task_result"]["evidence"] == ["match_job"]


async def test_ordinary_multi_step_plans_still_advance():
    """Regression cover: the guard is narrow and must not disarm planning."""
    from app.agents.workflow import reflect_node

    plan = [
        {"step": 1, "agent": "academic", "goal": "check attendance"},
        {"step": 2, "agent": "email", "goal": "draft a note to the professor"},
    ]
    result = await reflect_node(_reflect_state("SCHEDULE_TEMPORAL", plan))

    assert result["reflect_outcome"] == "next_step"
    assert result["selected_agent"] == "email"


def test_only_job_match_is_marked_single_step():
    """
    Deliberately not "every category that owns its agent" — SCHEDULE_TEMPORAL
    owns `academic`, and "check my attendance then email my professor" is a
    legitimate two-step plan.
    """
    from app.agents.query_intent import SINGLE_STEP_CATEGORIES, is_single_step
    from app.memory.sources import QueryCategory

    assert SINGLE_STEP_CATEGORIES == {QueryCategory.JOB_MATCH}
    assert is_single_step("JOB_MATCH") is True
    assert is_single_step("SCHEDULE_TEMPORAL") is False
    assert is_single_step("") is False
    assert is_single_step("NOT_A_CATEGORY") is False


def test_a_spoken_match_question_takes_the_tool_path():
    """
    The streaming path runs without tools. A spoken match question routed
    there would be answered by a model reading a memory blob — the one way
    this capability can still produce an unevidenced claim.
    """
    from app.agents.hybrid_router import ROUTE_TOOL, classify_heuristically

    for spoken in (
        "how well do I match this job",
        "am I a good fit for this role",
        "do I qualify for this position",
        "should I apply for this one",
    ):
        assert classify_heuristically(spoken) == ROUTE_TOOL, spoken
