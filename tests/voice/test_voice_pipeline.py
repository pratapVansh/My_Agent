"""
Voice pipeline regression tests.

These cover the failure modes that made a spoken conversation break, using the
real chunker/streamer/bridge code and faking only the network boundaries.
"""
import asyncio
import time
from typing import AsyncIterator, List

import pytest

from app.agents.hybrid_router import (
    ROUTE_CONVERSATIONAL,
    ROUTE_TOOL,
    classify_heuristically,
)
from app.config import settings
from app.livekit_worker import TurnProgress, speech_token_stream, watch_for_stall
from app.services.deepgram_bridge import DeepgramBridge
from app.services.text_chunker import TextChunker, text_chunker
from app.services.tts_streamer import TTSStreamer
from app.services.voice_service import TTSUnavailable


async def _tokens(items: List[str], delay: float = 0.0) -> AsyncIterator[str]:
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


# ── TextChunker ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_chunk_is_released_before_the_first_sentence_ends():
    """Time-to-first-audio: the opening chunk must not wait for a full stop."""
    words = "Sure thing let me take a look at that for you right now .".split()
    chunks = [c async for c in text_chunker.chunk_tokens(_tokens([w + " " for w in words]))]

    assert len(chunks) >= 2, "the reply was emitted as one late block"
    assert len(chunks[0].split()) <= 6
    assert " ".join(chunks).split() == words


@pytest.mark.asyncio
async def test_early_consumer_exit_does_not_raise_generator_exit():
    """A barge-in closes the chunker mid-stream.

    Yielding from a `finally` block made this raise
    "async generator ignored GeneratorExit", which surfaced as a crashed turn
    instead of a clean interruption.
    """
    gen = text_chunker.chunk_tokens(_tokens(["Hello ", "world ", "again ", "and "] * 6, delay=0.01))
    first = await gen.__anext__()
    await gen.aclose()  # must not raise
    assert first


@pytest.mark.asyncio
async def test_cancellation_mid_stream_propagates_cleanly():
    async def consume():
        async for _ in text_chunker.chunk_tokens(_tokens(["word "] * 200, delay=0.05)):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.12)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_abbreviations_and_decimals_are_not_sentence_boundaries():
    chunker = TextChunker(first_chunk_words=99, min_chunk_words=99)
    chunks = [
        c
        async for c in chunker.chunk_tokens(
            _tokens(["Dr. ", "Smith ", "charged ", "3.14 ", "dollars ", "total", "."]),
            max_words=99,
        )
    ]
    assert chunks == ["Dr. Smith charged 3.14 dollars total."]


@pytest.mark.asyncio
async def test_stalled_model_still_flushes_buffered_speech():
    async def stalling() -> AsyncIterator[str]:
        yield "Working "
        yield "on "
        yield "it "
        await asyncio.sleep(0.6)
        yield "now."

    chunks = [c async for c in text_chunker.chunk_tokens(stalling())]
    assert chunks[0].strip() != ""
    assert "now." in chunks[-1]


@pytest.mark.asyncio
async def test_no_tokens_produces_no_chunks():
    assert [c async for c in text_chunker.chunk_tokens(_tokens([]))] == []


# ── TTSStreamer ──────────────────────────────────────────────────────────


class FakeAudioSource:
    """Stands in for rtc.AudioSource, recording frame geometry."""

    def __init__(self, queued_duration: float = 0.0):
        self.frames = []
        self.cleared = 0
        self.queued_duration = queued_duration
        self.played_out = 0

    async def capture_frame(self, frame):
        self.frames.append(frame)

    def clear_queue(self):
        self.cleared += 1

    async def wait_for_playout(self):
        self.played_out += 1


class FakeFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


@pytest.fixture
def patched_streamer(monkeypatch):
    """A TTSStreamer whose frame type and TTS provider are fakes."""
    import app.services.tts_streamer as mod

    monkeypatch.setattr(mod.rtc, "AudioFrame", FakeFrame)
    return TTSStreamer()


def _fake_tts(monkeypatch, payload: bytes, fail: bool = False):
    import app.services.tts_streamer as mod

    async def fake_stream(text, voice_id=None):
        if fail:
            raise TTSUnavailable("boom")
        # Deliberately not aligned to a 20 ms frame, to exercise the tail flush.
        for i in range(0, len(payload), 700):
            yield payload[i : i + 700]

    monkeypatch.setattr(mod.voice_service, "synthesize_speech_stream", fake_stream)


@pytest.mark.asyncio
async def test_frames_are_exactly_20ms_and_the_tail_is_not_dropped(
    patched_streamer, monkeypatch
):
    streamer = patched_streamer
    _, bytes_per_frame = streamer._frame_geometry()
    # 2.5 frames' worth, so the final partial frame must be padded and sent.
    _fake_tts(monkeypatch, b"\x01\x02" * (bytes_per_frame + bytes_per_frame // 4))

    source = FakeAudioSource()
    pushed = await streamer.stream_to_track(_tokens(["Hello there."]), source)

    assert pushed == len(source.frames) == 3
    assert all(len(f.data) == bytes_per_frame for f in source.frames)
    assert all(f.sample_rate == streamer.sample_rate for f in source.frames)
    assert all(f.samples_per_channel == streamer._frame_geometry()[0] for f in source.frames)


@pytest.mark.asyncio
async def test_tts_failure_propagates_instead_of_playing_silence(
    patched_streamer, monkeypatch
):
    """Swallowing this made a broken API key look like the assistant ignoring you."""
    _fake_tts(monkeypatch, b"", fail=True)
    source = FakeAudioSource()

    with pytest.raises(TTSUnavailable):
        await patched_streamer.stream_to_track(_tokens(["Hello."]), source)
    assert source.cleared >= 1


@pytest.mark.asyncio
async def test_barge_in_clears_audio_already_queued_in_livekit(
    patched_streamer, monkeypatch
):
    """Cancelling stops new frames; queued frames keep playing unless cleared."""
    _fake_tts(monkeypatch, b"\x00\x01" * 40000)
    source = FakeAudioSource()

    task = asyncio.create_task(
        patched_streamer.stream_to_track(_tokens(["a. ", "b. ", "c. "], delay=0.02), source)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert source.cleared >= 1


# ── DeepgramBridge turn detection ────────────────────────────────────────


class FakeAlternative:
    def __init__(self, transcript):
        self.transcript = transcript


class FakeChannel:
    def __init__(self, transcript):
        self.alternatives = [FakeAlternative(transcript)]


class FakeResult:
    def __init__(self, transcript, is_final=False, speech_final=False):
        self.channel = FakeChannel(transcript)
        self.is_final = is_final
        self.speech_final = speech_final


def _bridge():
    turns, interims = [], []

    async def on_turn(text):
        turns.append(text)

    async def on_interim(text):
        interims.append(text)

    bridge = DeepgramBridge(on_utterance_end=on_turn, on_interim=on_interim)
    # Pretend a socket exists so the stale-connection guard passes.
    bridge._connection = object()
    bridge._connected = True
    return bridge, turns, interims


@pytest.mark.asyncio
async def test_speech_final_starts_the_turn_without_waiting_for_utterance_end():
    """utterance_end_ms is floored at 1000ms; endpointing is the fast path."""
    bridge, turns, _ = _bridge()
    conn = bridge._connection

    await bridge._on_transcript(conn, result=FakeResult("hello", is_final=True, speech_final=True))
    assert turns == ["hello"]


@pytest.mark.asyncio
async def test_utterance_end_does_not_replay_a_turn_speech_final_already_took():
    bridge, turns, _ = _bridge()
    conn = bridge._connection

    await bridge._on_transcript(conn, result=FakeResult("book a slot", is_final=True, speech_final=True))
    await bridge._on_utterance_end_event(conn)

    assert turns == ["book a slot"], "the same utterance was answered twice"


@pytest.mark.asyncio
async def test_utterance_end_is_the_backstop_when_endpointing_never_fires():
    bridge, turns, _ = _bridge()
    conn = bridge._connection

    await bridge._on_transcript(conn, result=FakeResult("what is", is_final=True))
    await bridge._on_transcript(conn, result=FakeResult("the weather", is_final=True))
    assert turns == []

    await bridge._on_utterance_end_event(conn)
    assert turns == ["what is the weather"]


@pytest.mark.asyncio
async def test_interim_results_report_text_without_ending_the_turn():
    """Interims must never trigger a turn — that is the caller's decision."""
    bridge, turns, interims = _bridge()
    conn = bridge._connection

    await bridge._on_transcript(conn, result=FakeResult("hel", is_final=False))
    await bridge._on_transcript(conn, result=FakeResult("hello th", is_final=False))

    assert interims == ["hel", "hello th"]
    assert turns == []


@pytest.mark.asyncio
async def test_events_from_a_replaced_connection_are_ignored():
    """A reconnect must not let the old socket deliver duplicate turns."""
    bridge, turns, interims = _bridge()
    stale = object()

    await bridge._on_transcript(stale, result=FakeResult("ghost", is_final=True, speech_final=True))
    await bridge._on_utterance_end_event(stale)

    assert turns == []
    assert interims == []


@pytest.mark.asyncio
async def test_empty_transcript_never_starts_a_turn():
    bridge, turns, _ = _bridge()
    conn = bridge._connection

    await bridge._on_transcript(conn, result=FakeResult("   ", is_final=True, speech_final=True))
    await bridge._on_utterance_end_event(conn)
    assert turns == []


@pytest.mark.asyncio
async def test_successful_open_resets_the_reconnect_backoff():
    bridge, _, _ = _bridge()
    bridge._consecutive_failures = 4
    await bridge._on_open(bridge._connection)
    assert bridge.consecutive_failures == 0


# ── Routing ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "transcript,expected",
    [
        ("hey", ROUTE_CONVERSATIONAL),
        ("thanks", ROUTE_CONVERSATIONAL),
        ("what can you do", ROUTE_CONVERSATIONAL),
        ("send an email to my professor about the deadline", ROUTE_TOOL),
        ("find me a job in data science", ROUTE_TOOL),
        ("what is my attendance this month", ROUTE_TOOL),
        ("what classes do I have today", ROUTE_TOOL),
    ],
)
def test_common_turns_route_without_an_llm_call(transcript, expected):
    """The router sits in front of every reply, so its cost is felt as latency."""
    assert classify_heuristically(transcript) == expected


def test_ambiguous_input_defers_to_the_llm():
    assert classify_heuristically("explain how transformers handle long contexts") is None


# ── Speaking replies that never stream tokens ────────────────────────────


async def _events(items: List[dict]) -> AsyncIterator[dict]:
    for item in items:
        yield item


async def _drain(events, seen=None):
    seen = seen if seen is not None else []

    async def on_event(event):
        seen.append(event["type"])

    return [t async for t in speech_token_stream(events, on_event)], seen


@pytest.mark.asyncio
async def test_a_clarifying_question_is_spoken_not_just_displayed():
    """The planner's clarification emits only `complete`.

    With nothing yielded, the chunker saw an empty stream and the question was
    shown on screen in silence.
    """
    tokens, seen = await _drain(
        _events([
            {"type": "metadata", "selected_agent": "clarification"},
            {
                "type": "complete",
                "display_text": "Which job did you mean?",
                "speech_text": "Which job did you mean?",
                "success": True,
            },
        ])
    )
    assert tokens == ["Which job did you mean?"]
    assert seen == ["metadata", "complete"]


@pytest.mark.asyncio
async def test_an_unroutable_request_is_spoken():
    tokens, _ = await _drain(
        _events([
            {
                "type": "complete",
                "display_text": "I couldn't process your request.",
                "speech_text": "I couldn't process your request.",
                "success": False,
            },
        ])
    )
    assert tokens == ["I couldn't process your request."]


@pytest.mark.asyncio
async def test_a_streamed_reply_is_not_repeated_by_the_fallback():
    tokens, _ = await _drain(
        _events([
            {"type": "metadata", "selected_agent": "profile"},
            {"type": "token", "token": "Hi ", "accumulated": "Hi "},
            {"type": "token", "token": "there.", "accumulated": "Hi there."},
            {"type": "complete", "display_text": "Hi there.", "speech_text": "Hi there.", "success": True},
        ])
    )
    assert tokens == ["Hi ", "there."], "the completed reply was spoken twice"


@pytest.mark.asyncio
async def test_an_empty_reply_yields_nothing_to_speak():
    tokens, _ = await _drain(
        _events([{"type": "complete", "display_text": "", "speech_text": "", "success": True}])
    )
    assert tokens == []


# ── Caption scheduling (text/audio sync) ─────────────────────────────────


@pytest.mark.asyncio
async def test_each_chunk_is_reported_with_when_it_will_be_heard(
    patched_streamer, monkeypatch
):
    """Captions must be paced by the audio clock, not by token arrival."""
    streamer = patched_streamer
    _, bytes_per_frame = streamer._frame_geometry()
    # 25 frames = 500 ms of audio per chunk.
    _fake_tts(monkeypatch, b"\x00\x01" * (bytes_per_frame * 25 // 2))

    # Pretend LiveKit already holds 1.2 s of undelivered audio.
    source = FakeAudioSource(queued_duration=1.2)
    scheduled: List = []

    before = time.monotonic()
    await streamer.stream_to_track(
        _tokens(["First part.", "Second part."]), source, on_chunk=scheduled.append
    )

    assert [c.text for c in scheduled] == ["First part.", "Second part."]
    for chunk in scheduled:
        assert chunk.duration == pytest.approx(0.5, abs=1e-6)
        # The chunk starts once the already-queued audio has drained.
        assert chunk.starts_at >= before + 1.2 - 0.05
        assert chunk.starts_at <= time.monotonic() + 1.2 + 0.05


@pytest.mark.asyncio
async def test_progress_is_reported_while_audio_is_produced(patched_streamer, monkeypatch):
    """This is what lets a long reply be told apart from a stalled one."""
    _, bytes_per_frame = patched_streamer._frame_geometry()
    _fake_tts(monkeypatch, b"\x00\x01" * (bytes_per_frame * 5 // 2))

    ticks = []
    await patched_streamer.stream_to_track(
        _tokens(["Hello."]), FakeAudioSource(), on_progress=lambda: ticks.append(1)
    )
    assert len(ticks) >= 5


@pytest.mark.asyncio
async def test_a_chunk_that_synthesises_to_nothing_is_skipped(patched_streamer, monkeypatch):
    import app.services.tts_streamer as mod

    async def empty(text, voice_id=None):
        if False:
            yield b""

    monkeypatch.setattr(mod.voice_service, "synthesize_speech_stream", empty)
    scheduled = []
    pushed = await patched_streamer.stream_to_track(
        _tokens(["Hello."]), FakeAudioSource(), on_chunk=scheduled.append
    )
    assert pushed == 0 and scheduled == []


# ── Turn liveness: long replies must not be cancelled ────────────────────


@pytest.mark.asyncio
async def test_a_long_but_progressing_turn_is_never_cancelled(monkeypatch):
    """The 25s wall-clock deadline killed any reply longer than 25s of speech.

    Speaking self-paces to real time, so elapsed time measures the length of the
    answer — only a lack of progress means the turn is wedged.
    """
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 0.3)
    monkeypatch.setattr(settings, "voice_turn_max_seconds", 600.0)

    progress = TurnProgress()
    finished = False

    async def long_turn():
        nonlocal finished
        # Runs far longer than the stall window, but keeps making progress.
        for _ in range(12):
            await asyncio.sleep(0.1)
            progress.touch()
        finished = True

    turn = asyncio.create_task(long_turn())
    guard = asyncio.create_task(watch_for_stall("u", progress, turn))
    await asyncio.gather(turn, return_exceptions=True)
    guard.cancel()

    assert finished, "a healthy long-running turn was cancelled"
    assert progress.abort_reason is None
    assert not turn.cancelled()


@pytest.mark.asyncio
async def test_a_stalled_turn_is_cancelled_and_labelled(monkeypatch):
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 0.2)
    monkeypatch.setattr(settings, "voice_turn_max_seconds", 600.0)

    progress = TurnProgress()

    async def wedged():
        await asyncio.sleep(30)

    turn = asyncio.create_task(wedged())
    guard = asyncio.create_task(watch_for_stall("u", progress, turn))
    with pytest.raises(asyncio.CancelledError):
        await turn
    await asyncio.gather(guard, return_exceptions=True)

    assert progress.abort_reason == "stalled"


@pytest.mark.asyncio
async def test_the_hard_ceiling_still_stops_a_runaway_turn(monkeypatch):
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 30.0)
    monkeypatch.setattr(settings, "voice_turn_max_seconds", 0.3)

    progress = TurnProgress()

    async def runaway():
        while True:
            await asyncio.sleep(0.05)
            progress.touch()  # busy, but never finishing

    turn = asyncio.create_task(runaway())
    guard = asyncio.create_task(watch_for_stall("u", progress, turn))
    with pytest.raises(asyncio.CancelledError):
        await turn
    await asyncio.gather(guard, return_exceptions=True)

    assert progress.abort_reason == "too_long"


async def _turn_like(progress: TurnProgress, body, published: list) -> None:
    """Mirrors execute_turn's cancellation handling so it can be tested directly."""
    guard = asyncio.create_task(watch_for_stall("u", progress, asyncio.current_task()))
    try:
        await body()
    except asyncio.CancelledError:
        if progress.abort_reason is None:
            published.append(("interrupted", None))
            raise
        current = asyncio.current_task()
        if current is not None:
            current.uncancel()
        published.append(("error", progress.abort_reason))
    finally:
        guard.cancel()
        published.append(("turn_end", progress.abort_reason))


@pytest.mark.asyncio
async def test_a_watchdog_cancellation_is_absorbed_so_the_turn_can_report(monkeypatch):
    """The turn must survive its own deadline long enough to publish turn_end.

    Re-raising here would mark the task cancelled and the terminal event would
    never reach the browser, leaving the UI stuck exactly as before.
    """
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 0.15)
    progress = TurnProgress()
    published: list = []

    async def wedged():
        await asyncio.sleep(30)

    task = asyncio.create_task(_turn_like(progress, wedged, published))
    await task  # must not raise

    assert not task.cancelled()
    assert published == [("error", "stalled"), ("turn_end", "stalled")]


@pytest.mark.asyncio
async def test_a_genuine_barge_in_still_propagates_as_cancellation(monkeypatch):
    """Only the watchdog's own cancellation is absorbed; barge-in must not be."""
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 30.0)
    progress = TurnProgress()
    published: list = []

    async def speaking():
        await asyncio.sleep(30)

    task = asyncio.create_task(_turn_like(progress, speaking, published))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert published == [("interrupted", None), ("turn_end", None)]


@pytest.mark.asyncio
async def test_the_watchdog_sets_its_reason_before_cancelling(monkeypatch):
    """Ordering is load-bearing: the turn reads this to tell a stall from a barge-in."""
    monkeypatch.setattr(settings, "voice_turn_stall_seconds", 0.15)
    progress = TurnProgress()
    observed = {}

    async def turn_body():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            observed["reason_at_cancel"] = progress.abort_reason
            raise

    turn = asyncio.create_task(turn_body())
    guard = asyncio.create_task(watch_for_stall("u", progress, turn))
    await asyncio.gather(turn, guard, return_exceptions=True)

    assert observed["reason_at_cancel"] == "stalled"
