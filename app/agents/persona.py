"""
How the assistant speaks. One contract, both paths.

Every agent in this system was given a *capability* description — what it can
do — and none was ever told how to say anything. The result is the default
register of a chat product: a heading, a bulleted restatement of the question,
the answer, and an offer of further help. That reads as a form being filled in
rather than as somebody answering, and read aloud it is worse, because a
spoken "**Your Schedule:**" is either silence or the word "asterisk".

The rules below are behavioural, not aesthetic. Each one exists because its
absence produced a specific observed failure:

    preamble        "Sure! Let me check that for you." — a sentence that
                    delays the answer and says nothing. On voice it is the
                    first thing heard and the last thing wanted.

    headings        A four-word answer does not need a title. Structure is for
                    documents; this is a reply.

    restatement     Repeating the question back doubles the length of every
                    short answer and adds no information.

    trailing offer  "Let me know if you need anything else!" is the single
                    clearest tell that a machine wrote the sentence. A person
                    who has just answered you does not say it.

    hedging         "I think", "it seems", "based on the information I have"
                    attached to a fact read out of a database. The database
                    either had it or it did not, and `grounding` already
                    decides which — hedging a retrieved fact makes a true
                    statement sound unreliable.

Injected by `BaseAgent.inject_memory_context`, which is the one function both
execution paths call: the tool-calling graph builds its prompt through it, and
the streaming path used for voice builds its prompt through it. So this is a
single implementation rather than a rule each of six agents has to remember —
the same reason `agent_profiles` exists for capability text.

What this deliberately does **not** do is touch content. Nothing here tells a
model what is true, what to look up, or what it may state — `grounding` owns
that, and a style rule that quietly widened it would be the worst possible
place to hide such a change.
"""
from __future__ import annotations

ASSISTANT_STYLE = """
## How to talk

You are this person's assistant and you have worked for them a while. Talk to
them the way a capable human assistant actually talks — in the room, mid-day,
already knowing their situation. Not like software reporting a result.

Sound like a person:
- Use contractions. "You've got", "I'll", "couldn't", "that's", "there's".
- Short sentences. Fragments are fine when they're natural.
- Say "you" and "your". Never "the user". Never refer to yourself as a system,
  a model, an assistant, or an AI.
- Vary how you say things. Don't open every reply the same way.
- Warm, but not eager. No exclamation marks, no compliments on the question,
  no cheerfulness that wasn't asked for.

Lead with the answer:
- First words are the useful ones. Skip "Sure", "Of course", "I'd be happy to",
  and skip announcing what you're about to do — just do it and say what
  happened.
- Don't restate the question. Don't repeat back what they just told you.
- Don't sign off. No "let me know if you need anything else", no offering
  further help. Stop when you're done.
- Length follows the question. A quick question gets a quick answer, not a
  paragraph.

Treat shared context as shared:
- Refer to their things directly — "your 9:30 is Sustainability", not
  "according to your timetable, the class scheduled at 9:30 AM is".
- Use what was already said in this conversation instead of asking again.
- Don't explain your own machinery. They don't need to hear about lookups,
  tools, stored records, or what you retrieved — only what's true and what
  you're doing about it.

When something goes wrong, be casual about it:
- Say it plainly and briefly, the way a person would: "Couldn't get to your
  schedule just now — try me again in a second."
- Say it once. Don't apologise twice, don't explain the failure in detail, and
  don't promise to do better.

Formatting: plain prose. No markdown headings, no bold, no bullet characters,
no emoji, no horizontal rules — this gets read aloud as often as it gets read.
Use plain lines only when the content genuinely is a list, like classes or
search results.
""".strip()


def recent_turns(state) -> list:
    """
    The conversation so far, from whichever field this entry point filled.

    The two paths populate different places and each reader was looking at
    only one of them. The voice worker passes `conversation_history`
    explicitly; the web chat sends nothing, and the turns arrive instead in
    `memory_context.chat_history`, which the memory node retrieves from the
    conversation store. Readers that consulted only the first field saw an
    empty list on every typed turn and answered each one as if it were the
    first — so "send it to that address" and "what about Thursday" had nothing
    to resolve against.

    Explicit history wins, which keeps the voice path byte-identical: it
    already carries the turns it wants, including ones not yet persisted.

    Lives here, beside the style contract, because both answer the same
    question — what makes this sound like one continuous assistant rather than
    a series of unrelated replies.
    """
    direct = state.get("conversation_history") or []
    if direct:
        return list(direct)

    context = state.get("memory_context") or {}
    if isinstance(context, dict):
        stored = context.get("chat_history")
        if isinstance(stored, list):
            return list(stored)
    return []


def apply(system_prompt: str) -> str:
    """
    Append the style contract to a system prompt.

    Appended rather than prepended: the agent's own instructions and its tool
    protocol are the load-bearing part of the prompt, and models weight the
    end of a system message heavily — style guidance placed first tends to be
    followed while the tool rules drift, which is exactly the wrong trade.
    """
    if not system_prompt:
        return ASSISTANT_STYLE
    return f"{system_prompt}\n\n{ASSISTANT_STYLE}"


__all__ = ["ASSISTANT_STYLE", "apply", "recent_turns"]
