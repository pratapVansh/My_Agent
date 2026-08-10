"""
Recognising "stop" — and why it cannot be an ordinary turn.

Barge-in in this system was gated on two conditions, and every word a person
actually uses to interrupt fails at least one of them:

    len(transcript.split()) >= voice_bargein_min_words   # default 2
    state.speaking                                        # TTS already playing

"Stop." is one word, so it never cleared the first. And the whole point of
saying it is usually that the assistant is *thinking* — grinding through an LLM
call or a tool round-trip — at which point nothing is being spoken and the
second gate is closed too. The result was an interruption mechanism that worked
only for long sentences said over the top of speech already in progress, which
is the one case where the user would probably have just waited.

Both gates exist for a real reason: raw VAD barges in on coughs, keyboard
noise, and the assistant's own voice echoing back from the speakers, and a
word-count floor is a cheap defence against that. So this module does not
remove the floor — it carves out an exemption for the small, closed set of
words whose *only* conversational function is to stop what is happening. A
one-word utterance of "stop" is not noise; it is the most deliberate thing a
user can say.

The second property that matters here is that a stop command is not a question.
Cancelling the turn and then sending "stop" to the language model produces a
reply to the word "stop" — the previous task ends and a new one immediately
takes its place, which from the user's side is indistinguishable from not
having been interrupted at all. `is_pure_stop` is what lets the caller cancel
and then say nothing.
"""
from __future__ import annotations

import re
from typing import Set

# Words whose only function is to halt what is happening. Deliberately closed
# and short: every addition is a word that will sometimes cancel a turn the
# user wanted, and the cost of a false stop is a lost answer.
#
# "no" earns its place despite being a normal answer to a yes/no question,
# because a bare "no" spoken over a reply in progress is a correction, not an
# answer — and the recovery ("ask again") is cheap, while being unable to stop
# a wrong answer mid-flight is the complaint this exists to fix.
STOP_WORDS: Set[str] = {
    "stop", "wait", "no", "cancel", "nope", "nevermind", "abort", "quit",
    "halt", "shush", "enough", "pause", "hold",
}

# Multi-word forms that are still nothing but a stop command.
_STOP_PHRASES: Set[str] = {
    "hold on", "hang on", "never mind", "stop it", "stop stop", "wait wait",
    "shut up", "be quiet", "stop talking", "stop please", "please stop",
    "wait a second", "wait a minute", "hold up", "cancel that", "forget it",
    "that is enough", "thats enough", "no no", "no wait", "stop right there",
}

# Leading filler that can precede a stop command without changing that it is
# one: "uh, stop", "ok stop", "hey wait".
_LEADING_FILLER = re.compile(
    r"^(uh|um|er|ah|oh|ok|okay|hey|yo|so|well|hmm+|like)\s+", re.IGNORECASE
)


def _normalise(text: str) -> str:
    """Lowercased, punctuation-stripped, filler-trimmed."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = _LEADING_FILLER.sub("", cleaned, count=1).strip()
    return cleaned


def is_stop_command(text: str) -> bool:
    """
    Whether this utterance is an instruction to stop.

    True for a bare stop word or stop phrase, and for one carrying leading
    filler. False for a sentence that merely *contains* one — "no, tell me
    about my projects" is a correction with a request attached, and cancelling
    without answering it would lose the request.
    """
    cleaned = _normalise(text)
    if not cleaned:
        return False
    if cleaned in _STOP_PHRASES:
        return True
    words = cleaned.split()
    if len(words) == 1:
        return words[0] in STOP_WORDS
    # Two words that are both stop words ("no stop", "wait no").
    if len(words) == 2 and all(word in STOP_WORDS for word in words):
        return True
    return False


def is_pure_stop(text: str) -> bool:
    """
    Whether this is *only* a stop, with no request attached.

    The caller cancels either way; this decides whether anything should be said
    or generated afterwards. "Stop" alone wants silence. "Stop, what's my CGPA"
    wants the current work abandoned and the new question answered — treating
    that as a bare stop would drop a question the user did ask.
    """
    return is_stop_command(text)


def carries_new_request(text: str) -> bool:
    """
    Whether an interrupting utterance also asks for something new.

    "No, tell me about my projects" both cancels and requests. The cancel is
    handled by the barge-in path; this reports that the remainder still needs
    answering, so the caller does not silently swallow it.
    """
    cleaned = _normalise(text)
    if not cleaned or is_stop_command(cleaned):
        return False
    words = cleaned.split()
    return bool(words) and words[0] in STOP_WORDS and len(words) > 1


def strip_stop_prefix(text: str) -> str:
    """
    The request left over after a leading stop word.

    "No, tell me about my projects" → "tell me about my projects". Returns the
    original text when there is no stop prefix to remove.
    """
    cleaned = _normalise(text)
    words = cleaned.split()
    if not words or words[0] not in STOP_WORDS:
        return text
    remainder = " ".join(words[1:]).strip()
    return remainder or text
