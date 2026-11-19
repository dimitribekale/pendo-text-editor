"""Pure helpers of the prediction package: context, selection, settings."""

from backend.prediction.context import build_context
from backend.prediction.selection import choose_completion
from backend.prediction.settings import PredictionSettings
from backend.prediction.suggestion import Suggestion


def test_word_after_space_gets_a_leading_space_target():
    context = build_context("Hello ", "wo", max_chars=500)
    assert (context.text, context.target) == ("Hello", " wo")


def test_word_at_line_start_has_no_leading_space():
    context = build_context("Hello\n", "wo", max_chars=500)
    assert (context.text, context.target) == ("Hello\n", "wo")


def test_context_keeps_only_the_most_recent_characters():
    assert build_context("a" * 50 + "END ", "", max_chars=3).text == "END"


def test_completion_is_the_rest_of_the_word():
    suggestions = [Suggestion("guitar", 0.9)]
    assert choose_completion(suggestions, "gui", min_confidence=0.3) == "tar"


def test_low_confidence_shows_nothing():
    suggestions = [Suggestion("guitar", 0.2)]
    assert choose_completion(suggestions, "gui", min_confidence=0.3) is None


def test_already_complete_word_shows_nothing():
    suggestions = [Suggestion("the", 0.9), Suggestion("there", 0.1)]
    assert choose_completion(suggestions, "the", min_confidence=0.3) is None


def test_completion_that_saves_no_typing_shows_nothing():
    suggestions = [Suggestion("cats", 0.9)]
    assert choose_completion(suggestions, "cat", min_confidence=0.3) is None


def test_no_suggestions_shows_nothing():
    assert choose_completion([], "x", min_confidence=0.3) is None


def test_settings_read_the_prediction_section():
    config = {"prediction": {"enable": False, "debounce_delay": 150, "model_path": "m"}}
    settings = PredictionSettings.from_config(config)
    assert (settings.enabled, settings.debounce_ms, settings.model_path) == (False, 150, "m")


def test_settings_fall_back_to_defaults():
    assert PredictionSettings.from_config(None) == PredictionSettings()
    assert PredictionSettings.from_config({"prediction": {"unknown": 1}}) == PredictionSettings()
