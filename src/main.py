import os
import tkinter as tk
from tkinter import ttk
import ctypes
from ui.menubar import MenuBar
from ui.statusbar import StatusBar
from ui.contextmenu import ContextMenu
from ui.toolbar import Toolbar
from ui.editor_frame import EditorFrame
from ui.settings_window import SettingsWindow
from backend.app_config import load_config, get_value
from backend.model_manager import ModelManager
from ui.theme import get_system_theme, DARK_THEME, LIGHT_THEME
from PIL import Image, ImageTk
from ui.app_managers.file_manager import FileManager
from ui.app_managers.tab_manager import TabManager

class Pendo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pendo")
        self.geometry("1200x800")
        self.minsize(600, 400)

        # Icon
        try:
            icon_path = os.path.join(os.path.dirname(__file__), 'images', 'pendo_icon.png')
            img = Image.open(icon_path)
            self.icon = ImageTk.PhotoImage(img)
            self.iconphoto(True, self.icon)
        except Exception as e:
            print(f"Error setting custom icon: {e}. Falling back to default.")
            # Fallback to the blue square if custom icon fails
            try:
                img = Image.new('RGB', (32, 32), color='#0078D7')
                self.icon = ImageTk.PhotoImage(img)
                self.iconphoto(True, self.icon)
            except Exception as e_fallback:
                print(f"Error setting fallback icon: {e_fallback}")

        # Theme
        self.app_config = load_config()
        theme_mode = get_value(self.app_config, "theme.mode", "system")
        if theme_mode == "system":
            self.theme_name = get_system_theme()
        elif theme_mode in ["light", "dark"]:
            self.theme_name = theme_mode
        else:
            self.theme_name = get_system_theme()
        self.theme = DARK_THEME if self.theme_name == "dark" else LIGHT_THEME

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure('.', background=self.theme["bg"], foreground=self.theme["fg"])
        style.configure('TFrame', background=self.theme["bg"])
        style.configure('TButton', background=self.theme["widget_bg"], foreground=self.theme["fg"])
        style.map('TButton', background=[('active', self.theme["select_bg"])])
        style.configure("TNotebook", background=self.theme["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=self.theme["bg"], foreground=self.theme["fg"], borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", self.theme["select_bg"])], foreground=[("selected", self.theme["fg"])])

        # Dark Title Bar (Windows only)
        if self.theme_name == 'dark':
            try:
                HWND = ctypes.windll.user32.GetParent(self.winfo_id())
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                value = 2 # Enable dark mode
                ctypes.windll.dwmapi.DwmSetWindowAttribute(HWND, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(ctypes.c_int(value)), ctypes.sizeof(ctypes.c_int(value)))
            except Exception as e:
                print(f"Error setting dark title bar: {e}")

        # Instantiate Managers
        self.model_manager = ModelManager()
        self.file_manager = FileManager(self)
        self.tab_manager = TabManager(self)

        menu_commands = {
            "new_file": self.file_manager.new_file,
            "open_file": self.file_manager.open_file,
            "save_file": self.file_manager.save_file,
            "save_as_file": self.file_manager.save_as_file,
            "exit": self.tab_manager._on_quit,
            "open_settings": self.open_settings_window,
            "close_tab": self.tab_manager._close_current_tab
        }
        self.config(menu=MenuBar(self, menu_commands, self.theme))

        # Configure grid weights for main window to allow resizing
        self.grid_rowconfigure(0, weight=0) # Toolbar row
        self.grid_rowconfigure(1, weight=1) # Notebook row
        self.grid_rowconfigure(2, weight=0) # Status bar row
        self.grid_columnconfigure(0, weight=1)

        toolbar = Toolbar(self, menu_commands)
        toolbar.grid(row=0, column=0, sticky="ew")

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=2, column=0, sticky="ew")

        self.notebook.bind("<<NotebookTabChanged>>", self.tab_manager._on_tab_changed)
        self.bind("<Control-n>", self.file_manager.new_file)
        self.bind("<Control-o>", self.file_manager.open_file)
        self.bind("<Control-s>", self.file_manager.save_file)
        self.bind("<Control-Shift-S>", self.file_manager.save_as_file)
        self.bind("<Control-w>", self.tab_manager._close_current_tab)

        self.file_manager.new_file() # Open initial untitled file
        self._apply_settings(self.app_config)

        self.protocol("WM_DELETE_WINDOW", self.tab_manager._on_quit)
        self.update_idletasks() # Force layout update


    def _apply_settings(self, config):
        self.app_config = config
        font_family = get_value(config, "editor.font_family", "Consolas")
        font_size = get_value(config, "editor.font_size", 12)

        for tab_id in self.notebook.tabs():
            editor_frame = self.notebook.nametowidget(tab_id)
            editor_frame.set_font(font_family, font_size)

    def open_settings_window(self, event=None):
        SettingsWindow(self, self.app_config, self._apply_settings)

    def _on_change(self, event=None):
        editor_frame = self.tab_manager.get_current_editor_frame()
        if editor_frame:
            self.status_bar.update_status(editor_frame.text_area)
            # Update tab title with asterisk if modified
            try:
                is_modified = editor_frame.text_area.edit_modified()
                current_title = self.notebook.tab(editor_frame, "text")
                if is_modified and not current_title.endswith("*"):
                    self.notebook.tab(editor_frame, text=current_title + "*")
                elif not is_modified and current_title.endswith("*"):
                    self.notebook.tab(editor_frame, text=current_title[:-1])
            except tk.TclError: # Happens when tab is being deleted
                pass

    def _bind_context_menu(self, editor_frame):
        context_menu = ContextMenu(self, editor_frame.text_area)
        editor_frame.text_area.bind("<Button-3>", lambda e: context_menu.show(e))

if __name__ == "__main__":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e:
        print(f"Error setting DPI awareness: {e}")
    app = Pendo()
    app.mainloop()