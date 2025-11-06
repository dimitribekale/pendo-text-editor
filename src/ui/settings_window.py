import tkinter as tk
from tkinter import ttk
from tkinter.font import families
from backend.app_config import save_config, get_value

class SettingsWindow(tk.Toplevel):
    def __init__(self, master, config, apply_callback):
        super().__init__(master)
        self.title("Settings")
        self.transient(master)
        self.geometry("350x250")

        self.config = config
        self.apply_callback = apply_callback

        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Font Family
        ttk.Label(self.main_frame, text="Font Family:").grid(row=0, column=0, sticky="w")
        self.font_family_var = tk.StringVar(value=get_value(self.config, "editor.font_family", "Consolas"))
        self.font_family_combo = ttk.Combobox(self.main_frame, textvariable=self.font_family_var)
        self.font_family_combo['values'] = sorted([f for f in families() if f.startswith('@') is False])
        self.font_family_combo.grid(row=0, column=1, sticky="ew")

        # Font Size
        ttk.Label(self.main_frame, text="Font Size:").grid(row=1, column=0, sticky="w")
        self.font_size_var = tk.IntVar(value=get_value(self.config, "editor.font_size", 12))
        self.font_size_spinbox = ttk.Spinbox(self.main_frame, from_=8, to=30, textvariable=self.font_size_var)
        self.font_size_spinbox.grid(row=1, column=1, sticky="ew")

        ttk.Label(self.main_frame, text="Theme:").grid(row=2, column=0, sticky="w")
        self.theme_var = tk.StringVar(value=get_value(self.config, "theme.mode", "system"))
        self.theme_combo = ttk.Combobox(self.main_frame, textvariable=self.theme_var, state="readonly")
        self.theme_combo["values"] = ["system", "light", "dark"]
        self.theme_combo.grid(row=2, column=1, sticky="ew")


        self.main_frame.columnconfigure(1, weight=1)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        apply_button = ttk.Button(button_frame, text="Apply", command=self._on_apply)
        apply_button.pack(side=tk.RIGHT, padx=5)

        close_button = ttk.Button(button_frame, text="Close", command=self.destroy)
        close_button.pack(side=tk.RIGHT)

    def _on_apply(self):
        current_config = self.config if self.config else {}

        if "editor" not in current_config:
            current_config["editor"] = {}
        if "theme" not in current_config:
            current_config["theme"] = {}
        
        current_config["editor"]["font_family"] = self.font_family_var.get()
        current_config["editor"]["font_size"] = self.font_size_var.get()
        current_config["theme"]["mode"] = self.theme_var.get()

        save_config(current_config)
        if self.apply_callback:
            self.apply_callback(current_config)
