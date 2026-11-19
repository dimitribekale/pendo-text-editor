"""A suggestion whose outcome is not fully known yet."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .interaction import Interaction


@dataclass
class PendingSuggestion:
    """What was shown, plus the timing needed to finish the record."""

    context: str
    prefix: str
    completion: str
    probability: float
    model: str
    shown_at: float
    typing_interval_ms: Optional[int] = None
    previous_outcome: Optional[str] = None
    hidden_at: float = 0.0

    def finish(self, outcome, actual_word=None):
        """Build the final interaction once the outcome is known."""
        return Interaction(
            time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            outcome=outcome,
            context=self.context,
            prefix=self.prefix,
            completion=self.completion,
            probability=round(self.probability, 4),
            shown_ms=round((self.hidden_at - self.shown_at) * 1000),
            model=self.model,
            actual_word=actual_word,
            typing_interval_ms=self.typing_interval_ms,
            previous_outcome=self.previous_outcome,
        )
