"""Prepares the text before the cursor for the language model."""

from dataclasses import dataclass

_INLINE_SPACE = " \t"


@dataclass(frozen=True)
class PredictionContext:
    """What the model reads, and the string the next word must start with.

    `target` is the typed prefix, preceded by a space when the word follows
    one, because the model's tokens carry their leading space.
    """

    text: str
    target: str


def build_context(prompt, prefix, max_chars):
    """Split raw editor text into model context and the constrained target.

    Args:
        prompt: Raw text before the word being typed (not stripped).
        prefix: Letters of the current word typed so far, possibly empty.
        max_chars: Upper bound on the context handed to the model.
    """
    text = prompt.rstrip(_INLINE_SPACE)
    follows_space = len(text) < len(prompt)
    lead = " " if follows_space and text else ""
    return PredictionContext(text=text[-max_chars:], target=lead + prefix)
