"""Behavioural signals stored with each interaction."""

from backend.interactions.interaction_log import InteractionLog
from backend.interactions.interaction_tracker import InteractionTracker
from backend.interactions.settings import InteractionSettings
from backend.interactions.typing_rhythm import TypingRhythm
from ui.editor.cursor_context import CursorContext


def _context(prompt, prefix):
    return CursorContext(index="1.0", prompt=prompt, prefix=prefix, at_word_end=True)


def test_rhythm_is_unknown_before_two_keys():
    rhythm = TypingRhythm(clock=lambda: 0.0)
    rhythm.key()
    assert rhythm.interval_ms is None


def test_rhythm_ignores_thinking_pauses():
    times = iter([0.0, 0.2, 5.0, 5.2])
    rhythm = TypingRhythm(clock=lambda: next(times))
    for _ in range(4):
        rhythm.key()
    assert rhythm.interval_ms == 200


def test_record_keeps_typing_pace_and_previous_outcome(tmp_path):
    log = InteractionLog(InteractionSettings(log_path=str(tmp_path / "log.jsonl")))
    ticks = iter(range(0, 1000))
    tracker = InteractionTracker(log, "test-model", clock=lambda: next(ticks) * 0.25)
    for prefix in ("w", "wo", "wor"):
        tracker.observe(_context("Hello ", prefix))
    tracker.shown(_context("Hello ", "wor"), "ld", 0.9)
    tracker.accepted()
    tracker.shown(_context("Hello world ", "ag"), "ain", 0.8)
    tracker.dismissed()
    first, second = log.read_all()
    assert (first.typing_interval_ms, first.previous_outcome) == (250, None)
    assert second.previous_outcome == "accepted"
