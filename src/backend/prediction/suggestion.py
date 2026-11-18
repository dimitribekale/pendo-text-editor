"""The value object a predictor returns."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Suggestion:
    """A complete candidate word and the model's confidence in it.

    `probability` is conditional on the letters the user has already typed,
    so it can be compared against a display threshold directly.
    """

    word: str
    probability: float
