"""One suggestion shown to the user and what they did with it."""

from dataclasses import asdict, dataclass
from typing import Optional

ACCEPTED = "accepted"    # the user pressed Tab
REJECTED = "rejected"    # the user kept typing instead
DISMISSED = "dismissed"  # the suggestion went away for another reason


@dataclass(frozen=True)
class Interaction:
    """A finished suggestion, ready to be logged.

    `actual_word` is what the user ended up writing, when that is known. It
    is the ground truth a model can learn from, and it shows whether a
    rejected suggestion would in fact have been correct.
    """

    time: str
    outcome: str
    context: str
    prefix: str
    completion: str
    probability: float
    shown_ms: int
    model: str
    actual_word: Optional[str] = None
    typing_interval_ms: Optional[int] = None  # user's pace when it appeared
    previous_outcome: Optional[str] = None  # what happened to the one before

    @property
    def suggested_word(self):
        return self.prefix + self.completion

    def to_record(self):
        """Plain dict for serialization."""
        return asdict(self)
