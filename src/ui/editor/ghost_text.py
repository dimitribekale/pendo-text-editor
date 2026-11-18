"""Shows a completion as gray inline text that is not part of the document."""

import tkinter as tk
from contextlib import contextmanager

_TAG = "ghost_text"


class GhostText:
    """Owns the lifecycle of one inline suggestion in a Text widget.

    The suggestion is drawn inside the widget but kept out of the undo
    history and the modified flag, so it never counts as an edit until the
    user accepts it.
    """

    def __init__(self, text_area, color="gray"):
        self._text = text_area
        self._text.tag_config(_TAG, foreground=color)

    @property
    def visible(self):
        return bool(self._text.tag_ranges(_TAG))

    def show(self, completion):
        """Display `completion` after the cursor, replacing any previous one."""
        self.clear()
        with self._untracked():
            cursor = self._text.index(tk.INSERT)
            self._text.insert(cursor, completion, _TAG)
            self._text.mark_set(tk.INSERT, cursor)

    def clear(self):
        """Remove the suggestion, leaving the document as the user typed it."""
        ranges = self._text.tag_ranges(_TAG)
        if ranges:
            with self._untracked():
                self._text.delete(ranges[0], ranges[-1])

    def accept(self):
        """Turn the suggestion into real, undoable text.

        Returns:
            bool: True if there was a suggestion to accept.
        """
        ranges = self._text.tag_ranges(_TAG)
        if not ranges:
            return False
        completion = self._text.get(ranges[0], ranges[-1])
        self.clear()
        self._text.insert(tk.INSERT, completion)
        return True

    @contextmanager
    def _untracked(self):
        """Make edits that leave undo history and the modified flag alone."""
        was_modified = self._text.edit_modified()
        self._text.config(undo=False)
        try:
            yield
        finally:
            self._text.config(undo=True)
            self._text.edit_modified(was_modified)
