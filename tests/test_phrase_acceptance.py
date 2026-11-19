"""Accepting a multi-word suggestion one word at a time in the editor."""

import time
import tkinter as tk
from types import SimpleNamespace

from backend.interactions.interaction_log import InteractionLog
from backend.interactions.settings import InteractionSettings
from backend.prediction.suggestion import Suggestion
from tests.fakes import FakePredictor
from ui.editor.prediction_handler import PredictionHandler

_ANSWERS = {
    ("see ", "y"): [Suggestion("you", 0.9)],
    ("see you ", ""): [Suggestion("soon", 0.9)],
}


def test_accept_word_takes_one_word_and_logs_each_step(text_area, tmp_path):
    log = InteractionLog(InteractionSettings(log_path=str(tmp_path / "log.jsonl")))
    handler = PredictionHandler(text_area, text_area, FakePredictor(_ANSWERS), lambda: None, log)
    text_area.insert(tk.INSERT, "see y")
    handler._on_key_release(SimpleNamespace(keysym="y"))
    deadline = time.time() + 0.5
    while time.time() < deadline:
        text_area.update()
        time.sleep(0.01)
    assert text_area.get("1.0", "end-1c") == "see you soon"
    assert handler._on_accept_word(None) == "break"
    assert text_area.index(tk.INSERT) == "1.7"
    handler._on_key_press(SimpleNamespace(keysym="x"))
    handler.cleanup()
    assert text_area.get("1.0", "end-1c") == "see you"
    records = [(r.outcome, r.completion, r.actual_word) for r in log.read_all()]
    assert records == [("accepted", "ou soon", "you"), ("rejected", " soon", None)]


def test_accept_word_without_suggestion_keeps_default_behaviour(text_area):
    handler = PredictionHandler(text_area, text_area, FakePredictor(), lambda: None)
    assert handler._on_accept_word(None) is None
    handler.cleanup()
