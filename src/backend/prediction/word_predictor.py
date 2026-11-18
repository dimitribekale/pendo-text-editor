"""The predictor the editor and the simulator talk to."""

from .context import build_context
from .language_model import LanguageModel
from .settings import PredictionSettings
from .suggestion import Suggestion
from .vocabulary import TokenVocabulary
from .word_search import WordSearch


class WordPredictor:
    """Ranks complete words for the position being typed."""

    def __init__(self, model, settings):
        self.settings = settings
        self._model = model
        self._search = WordSearch(
            model=model,
            vocabulary=TokenVocabulary(model.tokenizer, model.device),
            beam_width=settings.beam_width,
            max_word_tokens=settings.max_word_tokens,
        )

    @classmethod
    def from_settings(cls, settings=None):
        """Load the configured model and build a ready-to-use predictor."""
        settings = settings or PredictionSettings()
        model = LanguageModel(settings.resolved_model_path(), settings.device)
        return cls(model, settings)

    def predict(self, prompt, prefix=""):
        """Suggest words for the cursor position, most probable first.

        Args:
            prompt: Raw text before the current word (keep trailing spaces).
            prefix: Letters of the current word typed so far.

        Returns:
            list[Suggestion]: Whole words, each starting with `prefix`.
        """
        context = build_context(prompt, prefix, self.settings.max_context_chars)
        lead_length = len(context.target) - len(prefix)
        found = self._search.search(self._model.encode(context.text), context.target)
        suggestions = [
            Suggestion(word=text[lead_length:], probability=probability)
            for text, probability in found.items()
        ]
        return sorted(suggestions, key=lambda s: s.probability, reverse=True)
