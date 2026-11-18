"""A partially generated word inside the beam search."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hypothesis:
    """Tokens generated so far, the text they spell, and their log-probability.

    `parent` is the position of the hypothesis this one grew from within the
    previous search step, which is how the model's cached state is matched up.
    """

    token_ids: tuple = ()
    text: str = ""
    log_prob: float = 0.0
    parent: int = 0

    def covers(self, target):
        """True once the generated text contains the whole target prefix."""
        return len(self.text) >= len(target)

    def extended(self, token_id, piece, token_log_prob, parent):
        """Return a new hypothesis with one more token appended."""
        return Hypothesis(
            token_ids=self.token_ids + (token_id,),
            text=self.text + piece,
            log_prob=self.log_prob + token_log_prob,
            parent=parent,
        )
