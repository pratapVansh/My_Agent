"""
Running the lookup the turn was already required to make, before asking a model
whether to make it.

`app.agents.grounding` decides deterministically, at the routing edge, which
tools a category may not answer without — and then enforces afterwards that one
of them ran. Between those two deterministic points sat a language model being
asked a question whose only acceptable answer was already known: *should I call
`get_identity`?* Yes. It was always yes. The requirement had been computed, it
was printed into the prompt as a ROUTING DECISION the model was told not to
re-derive, and the answer was checked afterwards.

That question cost a full completion. "What is my name?" therefore spent one
call deciding to call `get_identity`, and a second reading the result back — on
an 8,000 TPM account where one call of that shape is ~4,000 tokens. And when
the model got the predetermined question *wrong* and answered without the tool,
`grounding.enforce` replaced its answer, `reflect_node` retried, and the whole
specialist ran again: the cheapest possible failure made the most expensive
possible response.

So the lookup runs first, and the model is handed the observation instead of
the choice. One call instead of two, `grounding` satisfied by construction
rather than by persuasion, and the retry loop left in place for the failures it
was actually written for.

Deliberately narrow, and the narrowness is the whole safety argument. A tool
belongs here only when calling it with **no arguments** is the complete lookup
for its category. That is a property of the tool, not a guess about the
sentence:

* `get_identity`, `get_education`, `get_skills` … read one section of the
  user's own record. There is no argument that could make them read a
  different one, so running them early cannot answer a different question than
  the model would have asked.
* `get_schedule` is excluded even though `SCHEDULE_TEMPORAL` requires it. It
  takes `when` and defaults to *today*, so a no-argument call on "what classes
  do I have tomorrow" would return the wrong day and — worse — would satisfy
  the grounding check with it. A prefetch that can seed a confidently wrong
  observation is worse than the call it saves.
* `match_job` and `recall_explicit_memory` are excluded for the same reason:
  their argument comes out of the utterance, and choosing it is genuine
  reasoning rather than a foregone conclusion.
* `job_search` *was* excluded on that reading, and the reading was wrong. Its
  wrapper already defaults `query` to the user's own utterance
  (`job_agent.tool_job_search`), so the zero-argument call is not a guess at
  the question — it is literally the sentence the model would have passed,
  enriched with the user's skills by the tool itself. It qualifies on exactly
  the stated property. Observed live before this change: only 1 JOB_SEARCH turn
  in 5 delivered listings, because the model kept answering without emitting
  the call and `grounding` — correctly — discarded the answer every time.

Nothing here decides *whether* a tool was required — that stays in `grounding`,
which is the one place that knowledge lives.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# Required tools whose zero-argument call is the whole lookup. Order within a
# category's requirement list still decides which one runs — see `plan`.
ZERO_ARGUMENT_LOOKUPS: frozenset[str] = frozenset({
    "get_identity",
    "get_profile_summary",
    "get_education",
    "get_experience",
    "get_achievements",
    "get_skills",
    "get_projects",
    "get_resume",
    "list_my_memories",
    # Zero-argument by construction: the wrapper substitutes the user's
    # utterance when no `query` is given. See the module docstring.
    "job_search",
})


def _arguments_for(tool: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    What to call the tool with. Empty for all but the one that narrows.

    `get_projects` accepts an optional `query`, and a follow-up that named a
    project ("tell me more about TRACE") has already had that name resolved
    deterministically into `followup_subject` at the routing edge. Passing it
    retrieves the one project rather than the portfolio — which is both a
    better observation and a much smaller one.
    """
    if tool == "get_projects":
        subject = (state.get("followup_subject") or "").strip()
        if subject:
            return {"query": subject}
    return {}


def plan(
    state: Dict[str, Any],
    available_tools: Iterable[str],
    *,
    already_used: Sequence[str] = (),
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    The lookup to run before the first model call, or None.

    Returns `(tool_name, arguments)`. None means the turn's requirement cannot
    be discharged without reasoning — either nothing was required, the agent
    holds none of the required tools, or the required tool needs an argument
    only the model can choose. In every one of those cases the loop behaves
    exactly as it did before this module existed.

    `already_used` lets a retry skip a tool that has already run this turn, so
    a second pass tries the requirement's *next* candidate rather than
    repeating the one that produced nothing.
    """
    required = [str(t) for t in (state.get("required_tools") or []) if t]
    if not required:
        return None

    held = set(available_tools)
    used = set(already_used)

    for tool in required:
        if tool in used or tool not in held:
            continue
        if tool not in ZERO_ARGUMENT_LOOKUPS:
            # Required, held, and genuinely parameterised. The model chooses.
            continue
        return tool, _arguments_for(tool, state)

    return None


__all__ = ["ZERO_ARGUMENT_LOOKUPS", "plan"]
