"""Prediction settings and how they are read from the application config."""

from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# config.yaml key (under `prediction:`) -> settings field
_CONFIG_KEYS = {
    "enable": "enabled",
    "model_path": "model_path",
    "device": "device",
    "max_context_length": "max_context_chars",
    "debounce_delay": "debounce_ms",
    "num_suggestions": "beam_width",
    "max_word_tokens": "max_word_tokens",
    "min_confidence": "min_confidence",
}


@dataclass(frozen=True)
class PredictionSettings:
    """Everything that tunes prediction, with safe defaults."""

    enabled: bool = True
    model_path: str = "bekalebendong/pendo-distilgpt2-wikitext"
    device: str = "auto"
    max_context_chars: int = 500
    debounce_ms: int = 300
    beam_width: int = 5
    max_word_tokens: int = 4
    min_confidence: float = 0.3

    @classmethod
    def from_config(cls, config):
        """Build settings from the loaded config dict, ignoring unknown keys."""
        section = (config or {}).get("prediction") or {}
        known = {field.name for field in fields(cls)}
        values = {
            _CONFIG_KEYS[key]: value
            for key, value in section.items()
            if key in _CONFIG_KEYS and _CONFIG_KEYS[key] in known
        }
        return cls(**values)

    def resolved_model_path(self):
        """Return a local directory if `model_path` names one, else the hub id."""
        candidate = Path(self.model_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return str(candidate) if candidate.is_dir() else self.model_path
