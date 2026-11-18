"""The reject filters keep ordinary prose and drop known artifacts."""

import pytest

from model_training.data_preprocessing.writing_sources import SOURCES

REJECT = {source.genre: source.reject for source in SOURCES}


@pytest.mark.parametrize("genre, text", [
    ("emails", "they hav= e=20 changed the last sentence"),
    ("blogs", "i think im going to be late lol"),
    ("blogs", "well.... that happened"),
    ("academic", r"We show that $p$ is a \cs-algebra"),
    ("academic", "for $a$ and $b$ in the group"),
])
def test_artifacts_are_rejected(genre, text):
    assert REJECT[genre].search(text)


@pytest.mark.parametrize("genre, text", [
    ("emails", "Please confirm the meeting time = 3 pm tomorrow."),
    ("blogs", "I'm sure it's time to give your opinion, I've been waiting..."),
    ("academic", "We study a model with $n$ parameters and report results."),
])
def test_ordinary_prose_is_kept(genre, text):
    assert not REJECT[genre].search(text)
