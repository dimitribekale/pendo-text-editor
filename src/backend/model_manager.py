"""Holds the single predictor shared by every editor tab."""

import logging

from backend.app_config import load_config
from backend.prediction.settings import PredictionSettings
from backend.prediction.word_predictor import WordPredictor

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton that loads the language model once, on first use."""

    _instance = None
    _predictor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_predictor(self):
        """Return the shared predictor, loading the model if needed."""
        if self._predictor is None:
            settings = PredictionSettings.from_config(load_config())
            logger.info("Loading prediction model %s", settings.model_path)
            self._predictor = WordPredictor.from_settings(settings)
        return self._predictor

    def is_loaded(self):
        """Check whether the model has been loaded yet."""
        return self._predictor is not None
