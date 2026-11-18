"""The tracker's record of what happened to each suggestion."""

import pytest

from backend.interactions.interaction_log import InteractionLog, open_interaction_log
from backend.interactions.interaction_tracker import InteractionTracker
from backend.interactions.settings import InteractionSettings
from ui.editor.cursor_context import CursorContext


def _context(prompt, prefix):
    return CursorContext(index="1.0", prompt=prompt, prefix=prefix, at_word_end=True)


@pytest.fixture
def log(tmp_path):
    return InteractionLog(InteractionSettings(log_path=str(tmp_path / "log.jsonl")))


@pytest.fixture
def tracker(log):
    ticks = iter(range(0, 1000))
    return InteractionTracker(log, "test-model", clock=lambda: next(ticks) * 0.25)


def test_accept_records_the_suggested_word(tracker, log):
    tracker.shown(_context("Hello ", "wo"), "rld", 0.9)
    tracker.accepted()
    (record,) = log.read_all()
    assert (record.outcome, record.actual_word, record.model) == ("accepted", "world", "test-model")
    assert (record.prefix, record.completion, record.probability) == ("wo", "rld", 0.9)
    assert record.shown_ms == 250


def test_rejection_waits_for_the_word_the_user_typed(tracker, log):
    tracker.shown(_context("Hello ", "wo"), "rld", 0.9)
    tracker.rejected()
    tracker.observe(_context("Hello ", "wom"))
    tracker.observe(_context("Hello ", "woman"))
    assert log.read_all() == []
    tracker.observe(_context("Hello woman ", ""))
    (record,) = log.read_all()
    assert (record.outcome, record.suggested_word, record.actual_word) == ("rejected", "world", "woman")


def test_rejections_within_one_word_share_the_actual_word(tracker, log):
    tracker.shown(_context("Hello ", "w"), "ay", 0.5)
    tracker.rejected()
    tracker.observe(_context("Hello ", "wo"))
    tracker.shown(_context("Hello ", "wo"), "od", 0.5)
    tracker.rejected()
    tracker.observe(_context("Hello ", "wor"))
    tracker.observe(_context("Hello wor.", ""))
    assert [r.actual_word for r in log.read_all()] == ["wor", "wor"]


def test_abandoned_word_has_no_actual_word(tracker, log):
    tracker.shown(_context("Hello ", "wo"), "rld", 0.9)
    tracker.rejected()
    tracker.observe(_context("Hello ", "w"))
    tracker.observe(_context("Hello ", ""))  # backspaced the whole word away
    assert log.read_all()[0].actual_word is None


def test_dismiss_and_flush_close_open_suggestions(tracker, log):
    tracker.shown(_context("A ", "b"), "c", 0.5)
    tracker.dismissed()
    tracker.shown(_context("A ", "d"), "e", 0.5)
    tracker.rejected()
    tracker.shown(_context("A ", "df"), "g", 0.5)
    tracker.flush()
    assert sorted(r.outcome for r in log.read_all()) == ["dismissed", "dismissed", "rejected"]


def test_stored_context_is_limited(tmp_path):
    settings = InteractionSettings(log_path=str(tmp_path / "log.jsonl"), context_chars=5)
    log = InteractionLog(settings)
    tracker = InteractionTracker(log)
    tracker.shown(_context("a long private sentence ", ""), "x", 0.5)
    tracker.accepted()
    assert log.read_all()[0].context == "ence "


def test_without_a_log_nothing_happens():
    tracker = InteractionTracker(None)
    tracker.shown(_context("Hello ", "wo"), "rld", 0.9)
    tracker.accepted()
    tracker.rejected()
    tracker.observe(_context("Hello ", ""))
    tracker.flush()


def test_logging_can_be_switched_off_in_config():
    assert open_interaction_log({"learning": {"log_interactions": False}}) is None
    assert open_interaction_log({}) is not None
