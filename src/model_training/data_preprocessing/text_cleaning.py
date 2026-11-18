"""Text cleaners that undo dataset-specific formatting artifacts."""

import re

_MIN_CHARS = 200
_MAX_CHARS = 8000
_MIN_ASCII_SHARE = 0.97

_SPACE_BEFORE_PUNCTUATION = re.compile(r" +([,.!?;:%)\]])")
_SPACE_BEFORE_CONTRACTION = re.compile(r" +(n't|'s|'re|'ve|'m|'ll|'d)\b", re.IGNORECASE)
_SPACE_AFTER_OPENING = re.compile(r"([(\[]) +")
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_NEWS_DATELINE = re.compile(r"^\s*(?:[A-Z][A-Za-z .,]*)?\(CNN\)\s*(?:--\s*)?")


def normalize_whitespace(text):
    """Collapse runs of spaces and of blank lines; trim the ends."""
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def unwrap_lines(text):
    """Join hard-wrapped lines into paragraphs (blank lines are kept)."""
    return _SINGLE_NEWLINE.sub(" ", text)


def detokenize(text):
    """Reattach punctuation that a tokenizer separated from its word."""
    text = text.replace("<newline>", "\n")
    text = re.sub(r"`` ?", '"', text)
    text = re.sub(r" ?''", '"', text)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    text = _SPACE_BEFORE_CONTRACTION.sub(r"\1", text)
    return _SPACE_AFTER_OPENING.sub(r"\1", text)


def strip_news_dateline(text):
    """Remove the "(CNN) --" marker that opens agency articles."""
    return _NEWS_DATELINE.sub("", text)


def remove_blog_markers(text):
    """Drop the placeholder the blog corpus uses for removed links."""
    return text.replace("urlLink", "")


def truncate_at_paragraph(text):
    """Limit very long documents, cutting at a paragraph or sentence end."""
    if len(text) <= _MAX_CHARS:
        return text
    cut = text[:_MAX_CHARS]
    end = max(cut.rfind("\n"), cut.rfind(". ") + 1)
    return cut[:end].strip() if end > _MIN_CHARS else cut


def is_usable(text):
    """Keep documents long enough to learn from and written in plain English."""
    if len(text) < _MIN_CHARS:
        return False
    ascii_chars = sum(1 for char in text if char.isascii())
    return ascii_chars / len(text) >= _MIN_ASCII_SHARE
