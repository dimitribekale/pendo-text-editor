"""Settings for interaction logging, read from the `learning` config section."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_LOG_PATH = Path.home() / ".pendo" / "interactions.jsonl"


@dataclass(frozen=True)
class InteractionSettings:
    """Controls whether, where and how much is recorded."""

    enabled: bool = True
    log_path: Optional[str] = None
    context_chars: int = 200

    @classmethod
    def from_config(cls, config):
        """Build settings from the loaded config dict."""
        section = (config or {}).get("learning") or {}
        defaults = cls()
        return cls(
            enabled=section.get("log_interactions", defaults.enabled),
            log_path=section.get("log_path", defaults.log_path),
            context_chars=section.get("context_chars", defaults.context_chars),
        )

    def resolved_log_path(self):
        """The file to write: the configured one, or the per-user default."""
        return Path(self.log_path).expanduser() if self.log_path else DEFAULT_LOG_PATH
