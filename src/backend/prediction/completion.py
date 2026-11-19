"""The text offered to the user at the cursor."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Completion:
    """Ghost text to display and how sure the model is of its first word.

    `text` finishes the current word and may continue with further whole
    words, each preceded by a space.
    """

    text: str
    probability: float
