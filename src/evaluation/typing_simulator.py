"""Replays a text one keystroke at a time against a predictor."""

import re
import time

from backend.prediction.selection import choose_completion

from .metrics import SimulationResult

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
_ACCEPT_KEYSTROKES = 1  # pressing Tab


class TypingSimulator:
    """Models a user who accepts a suggestion only when it is exactly right.

    Before each letter of each word the predictor is asked for a completion,
    using the same display rule as the editor. A correct completion is
    accepted with Tab; a wrong one is ignored and typing continues.
    """

    def __init__(self, predictor, min_confidence):
        self._predictor = predictor
        self._min_confidence = min_confidence

    def run(self, text):
        """Simulate typing `text` and return the collected metrics."""
        result = SimulationResult(characters=len(text))
        for match in _WORD.finditer(text):
            result.words += 1
            self._type_word(text[: match.start()], match.group(), result)
        return result

    def _type_word(self, prompt, word, result):
        for typed in range(len(word)):
            prefix = word[:typed]
            completion = self._timed_completion(prompt, prefix, result)
            if completion is None:
                continue
            result.shown += 1
            if prefix + completion == word:
                result.shown_correct += 1
                result.words_completed += 1
                result.keystrokes_saved += len(completion) - _ACCEPT_KEYSTROKES
                return

    def _timed_completion(self, prompt, prefix, result):
        started = time.perf_counter()
        suggestions = self._predictor.predict(prompt, prefix)
        result.latencies_ms.append((time.perf_counter() - started) * 1000)
        return choose_completion(suggestions, prefix, self._min_confidence)
