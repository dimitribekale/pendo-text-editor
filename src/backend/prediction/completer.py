"""Builds the completion to offer: the current word, then any sure follow-ups."""

from .completion import Completion
from .selection import choose_completion


class Completer:
    """Decides what, if anything, to show for a cursor position.

    The current word is offered when the predictor is confident enough.
    Further words are appended for as long as the whole phrase stays above
    the phrase threshold, so several words can be accepted with one key.
    """

    def __init__(self, predictor, settings=None):
        self.settings = settings or predictor.settings
        self._predictor = predictor

    def complete(self, prompt, prefix=""):
        """Return the `Completion` to display, or None to stay quiet.

        Args:
            prompt: Raw text before the current word (keep trailing spaces).
            prefix: Letters of the current word typed so far.
        """
        suggestions = self._predictor.predict(prompt, prefix)
        text = choose_completion(suggestions, prefix, self.settings.min_confidence)
        if text is None:
            return None
        best = suggestions[0]
        following = self._following_words(prompt + best.word, best.probability)
        return Completion(text + "".join(" " + word for word in following), best.probability)

    def _following_words(self, written, confidence):
        """Words expected after `written`, while the phrase remains likely."""
        words = []
        while len(words) < self.settings.max_phrase_words - 1:
            candidates = self._predictor.predict(written + " ", "")
            if not candidates:
                break
            confidence *= candidates[0].probability
            if confidence < self.settings.min_phrase_confidence:
                break
            words.append(candidates[0].word)
            written += " " + candidates[0].word
        return words
