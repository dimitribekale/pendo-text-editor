"""Replays a text one keystroke at a time against a predictor."""

import re
import time
from dataclasses import replace

from backend.prediction.completer import Completer

from .metrics import SimulationResult

_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


class TypingSimulator:
    """Models a user who accepts exactly the correct part of a suggestion.

    Before each letter of each word the editor's own completer is asked for
    a completion. If all of it is right it is accepted with one key (Tab).
    If only its first words are right they are accepted one key per word.
    If its first word is wrong it is ignored and typing continues.
    """

    def __init__(self, predictor, min_confidence):
        settings = replace(predictor.settings, min_confidence=min_confidence)
        self._completer = Completer(predictor, settings)

    def run(self, text):
        """Simulate typing `text` and return the collected metrics."""
        result = SimulationResult(characters=len(text))
        words = list(_WORD.finditer(text))
        result.words = len(words)
        position = 0
        while position < len(words):
            position += self._type_word(text, words, position, result)
        return result

    def _type_word(self, text, words, position, result):
        """Type one word; return how many words that finished (at least 1)."""
        start, word = words[position].start(), words[position].group()
        for typed in range(len(word)):
            completion = self._timed_completion(text[:start], word[:typed], result)
            if completion is None:
                continue
            result.shown += 1
            offered = (word[:typed] + completion.text).split(" ")
            taken = self._correct_words(offered, text, words, position)
            if not taken:
                continue
            key_presses = 1 if taken == len(offered) else taken
            written = words[position + taken - 1].end() - (start + typed)
            result.keystrokes_saved += written - key_presses
            result.shown_correct += 1
            result.shown_overreached += taken < len(offered)
            result.words_completed += taken
            return taken
        return 1

    @staticmethod
    def _correct_words(offered, text, words, position):
        """How many leading words of the suggestion match the text ahead."""
        count = 0
        for offset, offered_word in enumerate(offered):
            index = position + offset
            if index >= len(words) or words[index].group() != offered_word:
                break
            if offset and text[words[index - 1].end(): words[index].start()] != " ":
                break
            count += 1
        return count

    def _timed_completion(self, prompt, prefix, result):
        started = time.perf_counter()
        completion = self._completer.complete(prompt, prefix)
        result.latencies_ms.append((time.perf_counter() - started) * 1000)
        return completion
