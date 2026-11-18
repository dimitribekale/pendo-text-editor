"""The simulator's accounting of keystrokes and suggestions."""

import pytest

from backend.prediction.suggestion import Suggestion
from evaluation.typing_simulator import TypingSimulator
from tests.fakes import FakePredictor


def _run(answers, text="hello world"):
    return TypingSimulator(FakePredictor(answers), min_confidence=0.5).run(text)


def test_no_suggestions_saves_nothing():
    result = _run({})
    assert (result.words, result.keystrokes_saved, result.shown) == (2, 0, 0)


def test_correct_suggestion_saves_the_remaining_letters_minus_tab():
    result = _run({("hello ", "w"): [Suggestion("world", 0.9)]})
    # "orld" is 4 letters, accepted with 1 Tab press.
    assert result.keystrokes_saved == 3
    assert result.keystroke_savings == pytest.approx(3 / len("hello world"))
    assert (result.shown, result.shown_correct, result.words_completed) == (1, 1, 1)


def test_wrong_suggestion_counts_against_precision():
    result = _run({("hello ", "w"): [Suggestion("wide", 0.9)]})
    assert (result.shown, result.shown_correct, result.keystrokes_saved) == (1, 0, 0)
    assert result.precision == 0.0


def test_low_confidence_suggestion_is_not_shown():
    result = _run({("hello ", "w"): [Suggestion("world", 0.4)]})
    assert result.shown == 0


def test_typing_stops_querying_once_a_word_is_accepted():
    predictor = FakePredictor({("", "h"): [Suggestion("hello", 0.9)]})
    TypingSimulator(predictor, min_confidence=0.5).run("hello")
    assert predictor.calls == [("", ""), ("", "h")]


def test_latency_is_recorded_for_every_query():
    result = _run({})
    assert len(result.latencies_ms) == len("helloworld")
