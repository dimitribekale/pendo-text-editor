"""Follows each suggestion from display to outcome and logs the result."""

import time

from .interaction import ACCEPTED, DISMISSED, REJECTED
from .pending_suggestion import PendingSuggestion


class InteractionTracker:
    """Turns editor events into logged interactions.

    A rejected suggestion is held back until the user finishes the word, so
    the record can include what they actually wrote. With `log=None` every
    method is a harmless no-op.
    """

    def __init__(self, log, model_name="", clock=time.monotonic):
        self._log = log
        self._model = model_name
        self._clock = clock
        self._visible = None
        self._awaiting_word = []
        self._typed_word = ""

    def shown(self, context, completion, probability):
        """A suggestion appeared for the given cursor context."""
        if self._log is None:
            return
        self._visible = PendingSuggestion(
            context=context.prompt[-self._log.settings.context_chars:],
            prefix=context.prefix,
            completion=completion,
            probability=probability,
            model=self._model,
            shown_at=self._clock(),
        )

    def accepted(self):
        """The user pressed Tab."""
        pending = self._hide()
        if pending:
            self._write(pending, ACCEPTED, pending.prefix + pending.completion)

    def dismissed(self):
        """The suggestion disappeared without the user typing over it."""
        pending = self._hide()
        if pending:
            self._write(pending, DISMISSED)

    def rejected(self):
        """The user typed something else; wait to learn which word."""
        pending = self._hide()
        if pending:
            self._typed_word = pending.prefix
            self._awaiting_word.append(pending)

    def observe(self, context):
        """Called after each key: resolves rejections once the word ends."""
        if not self._awaiting_word:
            return
        if context.prefix:
            self._typed_word = context.prefix
            return
        word_just_ended = context.prompt[:-1].endswith(self._typed_word)
        self._resolve(self._typed_word if word_just_ended else None)

    def flush(self):
        """Write out anything still open, e.g. when the tab closes."""
        self.dismissed()
        self._resolve(None)

    def _hide(self):
        pending, self._visible = self._visible, None
        if pending:
            pending.hidden_at = self._clock()
        return pending

    def _resolve(self, typed_word):
        for pending in self._awaiting_word:
            matches = bool(typed_word) and typed_word.startswith(pending.prefix)
            self._write(pending, REJECTED, typed_word if matches else None)
        self._awaiting_word = []

    def _write(self, pending, outcome, actual_word=None):
        self._log.append(pending.finish(outcome, actual_word))
