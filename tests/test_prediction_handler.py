"""Cursor context reading and the handler's end-to-end flow in a widget."""

import time
import tkinter as tk
from types import SimpleNamespace

from backend.prediction.completion import Completion
from backend.prediction.suggestion import Suggestion
from tests.fakes import FakePredictor
from ui.editor.cursor_context import read_cursor_context
from ui.editor.prediction_handler import PredictionHandler


def _key(keysym):
    return SimpleNamespace(keysym=keysym)


def _pump(widget, seconds=0.5):
    """Let timers, the worker thread and its callback run."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        widget.update()
        time.sleep(0.01)


def _handler(text_area, answers):
    return PredictionHandler(text_area, text_area, FakePredictor(answers), lambda: None)


def test_context_splits_prompt_and_partial_word(text_area):
    text_area.insert("1.0", "Hello wor")
    context = read_cursor_context(text_area, max_chars=500)
    assert (context.prompt, context.prefix, context.at_word_end) == ("Hello ", "wor", True)


def test_context_keeps_apostrophes_inside_the_word(text_area):
    text_area.insert("1.0", "I don'")
    assert read_cursor_context(text_area, max_chars=500).prefix == "don'"


def test_context_detects_cursor_inside_a_word(text_area):
    text_area.insert("1.0", "hello")
    text_area.mark_set(tk.INSERT, "1.2")
    assert read_cursor_context(text_area, max_chars=500).at_word_end is False


def test_typing_shows_suggestion_and_tab_accepts_it(text_area):
    handler = _handler(text_area, {("Hello ", "wo"): [Suggestion("world", 0.9)]})
    text_area.insert(tk.INSERT, "Hello wo")
    handler._on_key_release(_key("o"))
    _pump(text_area)
    assert text_area.get("1.0", "end-1c") == "Hello world"
    assert handler._on_tab(_key("Tab")) == "break"
    assert text_area.index(tk.INSERT) == "1.11"
    handler.cleanup()


def test_next_key_press_dismisses_the_suggestion(text_area):
    handler = _handler(text_area, {("Hello ", "wo"): [Suggestion("world", 0.9)]})
    text_area.insert(tk.INSERT, "Hello wo")
    handler._on_key_release(_key("o"))
    _pump(text_area)
    handler._on_key_press(_key("x"))
    assert text_area.get("1.0", "end-1c") == "Hello wo"
    handler.cleanup()


def test_stale_result_is_discarded_when_text_changed(text_area):
    handler = _handler(text_area, {("Hello ", "wo"): [Suggestion("world", 0.9)]})
    text_area.insert(tk.INSERT, "Hello wo")
    context = read_cursor_context(text_area, max_chars=500)
    text_area.insert(tk.INSERT, "n")
    handler._show(context, Completion("rld", 0.9))
    assert text_area.get("1.0", "end-1c") == "Hello won"
    handler.cleanup()


def test_tab_without_suggestion_keeps_default_behaviour(text_area):
    handler = _handler(text_area, {})
    assert handler._on_tab(_key("Tab")) is None
    handler.cleanup()


def test_accepting_a_suggestion_is_logged(text_area, tmp_path):
    from backend.interactions.interaction_log import InteractionLog
    from backend.interactions.settings import InteractionSettings

    log = InteractionLog(InteractionSettings(log_path=str(tmp_path / "log.jsonl")))
    predictor = FakePredictor({("Hello ", "wo"): [Suggestion("world", 0.9)]})
    handler = PredictionHandler(text_area, text_area, predictor, lambda: None, log)
    text_area.insert(tk.INSERT, "Hello wo")
    handler._on_key_release(_key("o"))
    _pump(text_area)
    handler._on_tab(_key("Tab"))
    handler.cleanup()
    assert [(r.outcome, r.actual_word) for r in log.read_all()] == [("accepted", "world")]
