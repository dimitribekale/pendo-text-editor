"""Connects typing in a Text widget to the predictor and the ghost text."""

from backend.interactions.interaction_tracker import InteractionTracker
from backend.prediction.selection import choose_completion

from .cursor_context import read_cursor_context
from .ghost_text import GhostText
from .key_rules import dismisses_suggestion, requests_prediction
from .prediction_worker import PredictionWorker


class PredictionHandler:
    """Requests a prediction after the user pauses and shows it inline.

    A result is displayed only if the cursor context is unchanged by the
    time it arrives.
    """

    def __init__(self, master, text_area, predictor, change_callback, interaction_log=None):
        self._master = master
        self._text = text_area
        self._settings = predictor.settings
        self._change_callback = change_callback
        self._ghost = GhostText(text_area)
        self._worker = PredictionWorker(master, predictor)
        self._tracker = InteractionTracker(interaction_log, self._settings.model_path)
        self._debounce_timer = None

        text_area.bind("<KeyPress>", self._on_key_press)
        text_area.bind("<KeyRelease>", self._on_key_release)
        text_area.bind("<Button-1>", self.dismiss)
        text_area.bind("<Tab>", self._on_tab)

    def dismiss(self, event=None):
        """Remove any visible suggestion."""
        self._tracker.dismissed()
        self._ghost.clear()

    def cleanup(self):
        """Release the worker thread when the tab is closed."""
        self._cancel_debounce()
        self._worker.shutdown()
        self._tracker.flush()

    def _on_key_press(self, event):
        if dismisses_suggestion(event.keysym):
            self._tracker.rejected()
            self._ghost.clear()

    def _on_tab(self, event):
        if not self._ghost.accept():
            return None
        self._tracker.accepted()
        return "break"

    def _on_key_release(self, event):
        self._change_callback()
        self._tracker.observe(self._read_context())
        if self._settings.enabled and requests_prediction(event.keysym):
            self._cancel_debounce()
            self._worker.cancel()
            self._debounce_timer = self._master.after(
                self._settings.debounce_ms, self._request_prediction
            )

    def _cancel_debounce(self):
        if self._debounce_timer:
            self._master.after_cancel(self._debounce_timer)
            self._debounce_timer = None

    def _read_context(self):
        return read_cursor_context(self._text, self._settings.max_context_chars)

    def _request_prediction(self):
        self._debounce_timer = None
        context = self._read_context()
        if context.at_word_end and (context.prompt.strip() or context.prefix):
            self._worker.request(context, self._show)

    def _show(self, context, suggestions):
        if self._read_context() != context:
            return
        completion = choose_completion(
            suggestions, context.prefix, self._settings.min_confidence
        )
        if completion:
            self._ghost.show(completion)
            self._tracker.shown(context, completion, suggestions[0].probability)
