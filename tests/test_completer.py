"""Multi-word completions: how far a suggestion is extended."""

from dataclasses import replace

from backend.prediction.completer import Completer
from backend.prediction.settings import PredictionSettings
from backend.prediction.suggestion import Suggestion
from evaluation.typing_simulator import TypingSimulator
from tests.fakes import FakePredictor

_ANSWERS = {
    ("see ", "y"): [Suggestion("you", 0.9)],
    ("see you ", ""): [Suggestion("next", 0.8)],
    ("see you next ", ""): [Suggestion("week", 0.9)],
    ("see you next week ", ""): [Suggestion("then", 0.99)],
}


def _complete(**overrides):
    settings = replace(PredictionSettings(), **overrides)
    return Completer(FakePredictor(_ANSWERS, settings)).complete("see ", "y")


def test_sure_following_words_are_appended():
    completion = _complete(min_phrase_confidence=0.6)
    assert (completion.text, completion.probability) == ("ou next week", 0.9)


def test_extension_stops_when_the_phrase_becomes_unlikely():
    # 0.9 * 0.8 = 0.72 passes, 0.72 * 0.9 = 0.648 does not.
    assert _complete(min_phrase_confidence=0.7).text == "ou next"


def test_phrase_length_is_capped():
    assert _complete(min_phrase_confidence=0.1, max_phrase_words=2).text == "ou next"
    assert _complete(max_phrase_words=1).text == "ou"


def test_nothing_is_offered_below_the_word_threshold():
    assert _complete(min_confidence=0.95) is None


def _simulate(text):
    settings = replace(PredictionSettings(), min_phrase_confidence=0.6)
    return TypingSimulator(FakePredictor(_ANSWERS, settings), 0.5).run(text)


def test_simulator_accepts_a_correct_phrase_with_one_key():
    result = _simulate("see you next week")
    # "ou next week" is 12 characters for a single Tab press.
    assert (result.keystrokes_saved, result.words_completed) == (11, 3)
    assert (result.shown, result.shown_correct, result.shown_overreached) == (1, 1, 0)


def test_simulator_accepts_only_the_correct_start_of_a_phrase():
    result = _simulate("see you next time")
    # "ou next" is 7 characters for two accept-word presses.
    assert (result.keystrokes_saved, result.words_completed) == (5, 2)
    assert result.shown_overreached == 1


def test_simulator_does_not_accept_across_punctuation():
    result = _simulate("see you, next week")
    assert (result.keystrokes_saved, result.shown_overreached) == (1, 1)
