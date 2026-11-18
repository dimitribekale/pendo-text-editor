"""Append-only JSON Lines file of interactions, kept on this machine only."""

import json
import logging
import threading

from .interaction import Interaction
from .settings import InteractionSettings

logger = logging.getLogger(__name__)


class InteractionLog:
    """Stores interactions durably; a logging failure never disturbs typing."""

    def __init__(self, settings):
        self.settings = settings
        self._path = settings.resolved_log_path()
        self._lock = threading.Lock()

    def append(self, interaction):
        """Write one interaction as a single JSON line."""
        line = json.dumps(interaction.to_record(), ensure_ascii=False)
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")
        except OSError:
            logger.exception("Could not write interaction log %s", self._path)

    def read_all(self):
        """Return every stored interaction, oldest first."""
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as file:
            return [Interaction(**json.loads(line)) for line in file if line.strip()]


def open_interaction_log(config):
    """Return the log described by the config, or None when logging is off."""
    settings = InteractionSettings.from_config(config)
    return InteractionLog(settings) if settings.enabled else None
