"""Pendo's main window: wires the menu, tabs, status bar and shortcuts."""

import ctypes
import logging
import tkinter as tk
from tkinter import ttk

from backend.app_config import get_value, load_config
from backend.interactions.interaction_log import open_interaction_log
from backend.model_manager import ModelManager
from ui.app_icon import set_app_icon
from ui.app_managers.file_manager import FileManager
from ui.app_managers.tab_manager import TabManager
from ui.contextmenu import ContextMenu
from ui.menubar import MenuBar
from ui.settings_window import SettingsWindow
from ui.statusbar import StatusBar
from ui.window_style import apply_theme, resolve_theme


class Pendo(tk.Tk):
    """The application window."""

    def __init__(self):
        super().__init__()
        self.title("Pendo")
        self.geometry("1200x800")
        self.minsize(600, 400)
        set_app_icon(self)
        self.app_config = load_config()
        self.theme_name, self.theme = resolve_theme(self.app_config)
        apply_theme(self, self.theme_name, self.theme)
        self.model_manager = ModelManager()
        self.interaction_log = open_interaction_log(self.app_config)
        self.file_manager = FileManager(self)
        self.tab_manager = TabManager(self)
        self._build_layout()
        self._bind_shortcuts()

        self.file_manager.new_file()
        self.protocol("WM_DELETE_WINDOW", self.tab_manager._on_quit)
        self.update_idletasks()

    def _build_layout(self):
        commands = {
            "new_file": self.file_manager.new_file,
            "open_file": self.file_manager.open_file,
            "save_file": self.file_manager.save_file,
            "save_as_file": self.file_manager.save_as_file,
            "exit": self.tab_manager._on_quit,
            "open_settings": self.open_settings_window,
            "close_tab": self.tab_manager._close_current_tab,
        }
        self.config(menu=MenuBar(self, commands, self.theme))
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=1, column=0, sticky="ew")

    def _bind_shortcuts(self):
        self.notebook.bind("<<NotebookTabChanged>>", self.tab_manager._on_tab_changed)
        self.bind("<Control-n>", self.file_manager.new_file)
        self.bind("<Control-o>", self.file_manager.open_file)
        self.bind("<Control-s>", self.file_manager.save_file)
        self.bind("<Control-Shift-S>", self.file_manager.save_as_file)
        self.bind("<Control-w>", self.tab_manager._close_current_tab)

    def _apply_settings(self, config):
        self.app_config = config
        font_family = get_value(config, "editor.font_family", "Consolas")
        font_size = get_value(config, "editor.font_size", 12)
        for tab_id in self.notebook.tabs():
            self.notebook.nametowidget(tab_id).set_font(font_family, font_size)

    def open_settings_window(self, event=None):
        SettingsWindow(self, self.app_config, self._apply_settings)

    def _on_change(self, event=None):
        """Refresh the status bar and the tab's unsaved-changes asterisk."""
        editor_frame = self.tab_manager.get_current_editor_frame()
        if not editor_frame:
            return
        self.status_bar.update_status(editor_frame.text_area)
        self.tab_manager.refresh_title(editor_frame)

    def _bind_context_menu(self, editor_frame):
        context_menu = ContextMenu(self, editor_frame.text_area)
        editor_frame.text_area.bind("<Button-3>", context_menu.show)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):  # not Windows
        pass
    Pendo().mainloop()
