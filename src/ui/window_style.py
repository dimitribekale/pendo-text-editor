"""Applies the colour theme to the main window."""

import ctypes
import logging
from tkinter import ttk

from backend.app_config import get_value

from .theme import DARK_THEME, LIGHT_THEME, get_system_theme

logger = logging.getLogger(__name__)
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20


def resolve_theme(config):
    """Return (name, palette) for the theme chosen in the config."""
    name = get_value(config, "theme.mode", "system")
    if name not in ("light", "dark"):
        name = get_system_theme()
    return name, DARK_THEME if name == "dark" else LIGHT_THEME


def apply_theme(window, name, theme):
    """Style the ttk widgets and, on Windows, darken the title bar."""
    style = ttk.Style(window)
    style.theme_use("clam")
    style.configure(".", background=theme["bg"], foreground=theme["fg"])
    style.configure("TFrame", background=theme["bg"])
    style.configure("TButton", background=theme["widget_bg"], foreground=theme["fg"])
    style.map("TButton", background=[("active", theme["select_bg"])])
    style.configure("TNotebook", background=theme["bg"], borderwidth=0)
    style.configure(
        "TNotebook.Tab", background=theme["bg"], foreground=theme["fg"], borderwidth=0
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", theme["select_bg"])],
        foreground=[("selected", theme["fg"])],
    )
    if name == "dark":
        _darken_title_bar(window)


def _darken_title_bar(window):
    try:
        handle = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(2)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            handle, _DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value)
        )
    except (AttributeError, OSError):  # not Windows, or an older version
        logger.debug("Dark title bar is not available", exc_info=True)
