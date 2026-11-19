"""Sets the window icon."""

import logging
from pathlib import Path

from PIL import Image, ImageTk

logger = logging.getLogger(__name__)
_ICON_PATH = Path(__file__).resolve().parents[1] / "images" / "pendo_icon.png"


def set_app_icon(window):
    """Use the Pendo icon; keep the default one if it cannot be loaded."""
    try:
        window.icon = ImageTk.PhotoImage(Image.open(_ICON_PATH))
        window.iconphoto(True, window.icon)
    except (OSError, RuntimeError):
        logger.warning("Could not load the icon at %s", _ICON_PATH, exc_info=True)
