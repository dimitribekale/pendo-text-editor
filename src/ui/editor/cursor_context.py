"""Reads what the predictor needs to know about the cursor position."""

import re
import tkinter as tk
from typing import NamedTuple

# Letters of the word being typed, allowing inner apostrophes ("don't").
_PARTIAL_WORD = re.compile(r"(?:[^\W_]+['’]?)*$")


class CursorContext(NamedTuple):
    """A snapshot of the cursor; two equal snapshots mean nothing changed."""

    index: str
    prompt: str
    prefix: str
    at_word_end: bool


def read_cursor_context(text_area, max_chars):
    """Capture the text around the cursor.

    Args:
        text_area: The tk.Text widget being edited.
        max_chars: How much text before the cursor to read.

    Returns:
        CursorContext: `prompt` is the raw text before the current word,
        `prefix` the letters of that word typed so far, and `at_word_end`
        is False when the cursor sits in the middle of a word.
    """
    index = text_area.index(tk.INSERT)
    before = text_area.get(f"{index}-{max_chars}c", index)
    prefix = _PARTIAL_WORD.search(before).group()
    next_char = text_area.get(index)
    return CursorContext(
        index=index,
        prompt=before[: len(before) - len(prefix)],
        prefix=prefix,
        at_word_end=not next_char.isalnum(),
    )
