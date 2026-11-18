"""Word search and predictor behaviour on a scripted model."""

import pytest

from backend.prediction.settings import PredictionSettings
from backend.prediction.word_predictor import WordPredictor
from tests.fakes import FakeLanguageModel

# After a space the model can write " the" as one token, or spell " the"
# and " there" through the sub-word " th". The race is close enough that
# the search has to explore every path before it knows the winner.
SCRIPT = {
    (): {" the": 0.3, " th": 0.45, " cat": 0.25},
    (" th",): {"ere": 0.5, "e": 0.5},
}


@pytest.fixture
def predictor():
    return WordPredictor(FakeLanguageModel(SCRIPT), PredictionSettings())


def test_words_start_with_the_typed_prefix(predictor):
    words = [s.word for s in predictor.predict("Hello ", "th")]
    assert words == ["the", "there"]


def test_probability_is_conditional_on_the_prefix(predictor):
    best = predictor.predict("Hello ", "th")[0]
    # P(the) = 0.3 + 0.45 * 0.5 = 0.525, out of 0.75 for words starting "th".
    assert best.probability == pytest.approx(0.525 / 0.75)


def test_paths_spelling_the_same_word_are_merged(predictor):
    words = [s.word for s in predictor.predict("Hello ", "th")]
    assert words.count("the") == 1


def test_without_prefix_all_next_words_are_ranked(predictor):
    words = [s.word for s in predictor.predict("Hello ", "")]
    assert words[:2] == ["the", "cat"]


def test_impossible_prefix_gives_no_suggestions(predictor):
    assert predictor.predict("Hello ", "zz") == []


def test_prefix_case_is_respected(predictor):
    assert predictor.predict("Hello ", "Th") == []


def test_search_stops_early_once_the_winner_is_certain():
    script = {(): {" the": 0.9, " th": 0.1}, (" th",): {"ere": 1.0}}
    predictor = WordPredictor(FakeLanguageModel(script), PredictionSettings())
    # " there" is never finished: it could not overtake " the" anyway.
    assert [s.word for s in predictor.predict("Hello ", "th")] == ["the"]
