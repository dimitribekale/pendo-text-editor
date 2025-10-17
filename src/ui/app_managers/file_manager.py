import tkinter as tk
from tkinter import filedialog, messagebox
import os
from ..editor_frame import EditorFrame

class FileManager:
    def __init__(self, app):
        self.app = app

    def new_file(self, event=None):
        editor_frame = EditorFrame(self.app.notebook, theme=self.app.theme, change_callback=self.app._on_change)
        editor_frame.file_path = None
        self.app.notebook.add(editor_frame, text="Untitled")
        self.app.notebook.select(editor_frame)
        self.app._bind_context_menu(editor_frame)
        self.app._apply_settings(self.app.settings) # Apply settings to new tab
        editor_frame.text_area.edit_modified(False)

    def open_file(self, event=None):
        file_path = filedialog.askopenfilename(defaultextension=".txt",
                                               filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            editor_frame = EditorFrame(self.app.notebook, theme=self.app.theme, change_callback=self.app._on_change)
            editor_frame.file_path = file_path
            with open(file_path, "r") as file:
                editor_frame.set_text(file.read())
            self.app.notebook.add(editor_frame, text=os.path.basename(file_path))
            self.app.notebook.select(editor_frame)
            self.app._bind_context_menu(editor_frame)
            self.app._apply_settings(self.app.settings) # Apply settings to new tab
            editor_frame.text_area.edit_modified(False)

    def save_file(self, event=None):
        editor_frame = self.app.get_current_editor_frame()
        if not editor_frame:
            return False # Indicate failure

        if editor_frame.file_path:
            with open(editor_frame.file_path, "w") as file:
                file.write(editor_frame.get_text())
            editor_frame.text_area.edit_modified(False)
            self.app._on_change() # Update title
            return True # Indicate success
        else:
            return self.save_as_file()

    def save_as_file(self, event=None):
        editor_frame = self.app.get_current_editor_frame()
        if not editor_frame:
            return False # Indicate failure

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w") as file:
                file.write(editor_frame.get_text())
            editor_frame.file_path = file_path
            self.app.notebook.tab(editor_frame, text=os.path.basename(file_path))
            editor_frame.text_area.edit_modified(False)
            self.app._on_change() # Update title
            return True # Indicate success
        return False # Indicate failure (user cancelled)
