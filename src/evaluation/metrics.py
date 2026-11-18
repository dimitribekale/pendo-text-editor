"""Accumulates what happened during a simulated typing session."""

from dataclasses import dataclass, field
from statistics import mean, quantiles


@dataclass
class SimulationResult:
    """Counters for one run, plus the ratios derived from them."""

    characters: int = 0
    keystrokes_saved: int = 0
    words: int = 0
    words_completed: int = 0
    shown: int = 0
    shown_correct: int = 0
    latencies_ms: list = field(default_factory=list)

    @property
    def keystroke_savings(self):
        """Share of keystrokes the user did not have to type."""
        return self.keystrokes_saved / self.characters if self.characters else 0.0

    @property
    def precision(self):
        """Share of displayed suggestions that were the intended word."""
        return self.shown_correct / self.shown if self.shown else 0.0

    @property
    def word_coverage(self):
        """Share of words finished by accepting a suggestion."""
        return self.words_completed / self.words if self.words else 0.0

    @property
    def latency_mean_ms(self):
        return mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def latency_p95_ms(self):
        if len(self.latencies_ms) < 2:
            return self.latency_mean_ms
        return quantiles(self.latencies_ms, n=20)[-1]
