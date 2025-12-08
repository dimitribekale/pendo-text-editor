"""
Text manipulation utilities for Pendo Text Editor.

Provides reusable functions for text widget operations, particularly
for word extraction and cursor positioning.
"""

import re
import tkinter as tk


def extract_partial_word(text_area, cursor_index):
    """
    Extract the current partial word being typed at cursor position.

    This function looks backwards from the cursor to find the word boundary
    and extracts any alphanumeric characters (partial word) being typed.

    Args:
        text_area: tk.Text widget instance
        cursor_index: Current cursor position (e.g., "1.5")

    Returns:
        tuple: (partial_word, start_of_word_index)
            - partial_word: The partial word string at cursor (empty if none)
            - start_of_word_index: Index where the partial word starts

    Example:
        If text is "hello w|orld" (cursor at |), returns ("w", "1.6")
        If text is "hello |world" (cursor at |), returns ("", "1.6")
    """
    line_text_before_cursor = text_area.get(f"{cursor_index} linestart", cursor_index)
    current_word_match = re.search(r"\b(\w*)$", line_text_before_cursor)
    partial_word = current_word_match.group(1) if current_word_match else ""

    if partial_word:
        start_of_word_index = f"{cursor_index}-{len(partial_word)}c"
    else:
        start_of_word_index = cursor_index

    return partial_word, start_of_word_index


def get_prompt_before_cursor(text_area, cursor_index, exclude_partial_word=True):
    """
    Get text content from start to cursor, optionally excluding partial word.

    This is useful for generating prediction prompts where you may want to
    exclude the partial word currently being typed.

    Args:
        text_area: tk.Text widget instance
        cursor_index: Current cursor position (e.g., "1.5")
        exclude_partial_word: If True, excludes the partial word at cursor.
                            If False, includes everything up to cursor.

    Returns:
        str: Prompt text from start of document to cursor (or before partial word)

    Example:
        Text: "hello wor|ld" (cursor at |)
        exclude_partial_word=True: returns "hello "
        exclude_partial_word=False: returns "hello wor"
    """
    if exclude_partial_word:
        partial_word, start_of_word_index = extract_partial_word(text_area, cursor_index)
        prompt_end_index = start_of_word_index
    else:
        prompt_end_index = cursor_index

    return text_area.get("1.0", prompt_end_index).strip()


def replace_word_at_cursor(text_area, cursor_index, replacement_text):
    """
    Replace the partial word at cursor with new text.

    This is useful for accepting suggestions and replacing the partial
    word with the complete suggestion.

    Args:
        text_area: tk.Text widget instance
        cursor_index: Current cursor position
        replacement_text: Text to replace partial word with

    Returns:
        str: New cursor position after replacement

    Example:
        Text: "hello w|orld" (cursor at |), replacement="world"
        Result: "hello world|" (cursor moved to end)
    """
    partial_word, start_of_word_index = extract_partial_word(text_area, cursor_index)

    # Delete the partial word
    if partial_word:
        text_area.delete(start_of_word_index, cursor_index)

    # Insert replacement text
    text_area.insert(tk.INSERT, replacement_text)

    # Return new cursor position
    return text_area.index(tk.INSERT)
