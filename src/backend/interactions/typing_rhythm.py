"""Estimates how fast the user is typing right now."""

from collections import deque
from statistics import median

_PAUSE_SECONDS = 2.0  # longer gaps are thinking, not typing


class TypingRhythm:
    """Remembers the gaps between recent key presses.

    Fast typists gain less from a suggestion than slow ones, so the pace at
    the moment a suggestion appears helps explain whether it was accepted.
    """

    def __init__(self, clock, window=10):
        self._clock = clock
        self._gaps = deque(maxlen=window)
        self._last_key = None

    def key(self):
        """Record that a key was just pressed."""
        now = self._clock()
        if self._last_key is not None and now - self._last_key <= _PAUSE_SECONDS:
            self._gaps.append(now - self._last_key)
        self._last_key = now

    @property
    def interval_ms(self):
        """Typical gap between recent keys, or None before enough typing."""
        return round(median(self._gaps) * 1000) if self._gaps else None
