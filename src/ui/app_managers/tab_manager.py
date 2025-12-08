import tkinter as tk
from tkinter import messagebox

class TabManager:
    def __init__(self, app):
        self.app = app

    def _on_tab_changed(self, event=None):
        self.app._on_change()

    def get_current_editor_frame(self):
        if not self.app.notebook.tabs():
            return None
        selected_tab = self.app.notebook.select()
        return self.app.notebook.nametowidget(selected_tab)

    def _close_current_tab(self, event=None):
        editor_frame = self.get_current_editor_frame()
        if not editor_frame:
            return

        if editor_frame.text_area.edit_modified():
            file_name = self.app.notebook.tab(editor_frame, "text").replace("*", "")
            result = messagebox.askyesnocancel("Save Changes?", f"Do you want to save changes to {file_name}?")
            if result is True: # Yes
                if not self.app.file_manager.save_file():
                    return # Don't close if save was cancelled
            elif result is None: # Cancel
                return

        # Clean up resources before closing tab
        if hasattr(editor_frame, 'cleanup'):
            editor_frame.cleanup()

        self.app.notebook.forget(editor_frame)
        if not self.app.notebook.tabs():
            self.app.file_manager.new_file() # Create a new tab if the last one is closed

    def _on_quit(self):
        # Iterate over a copy of the tabs, as we might be closing them
        for tab_id in list(self.app.notebook.tabs()):
            self.app.notebook.select(tab_id)
            editor_frame = self.app.notebook.nametowidget(tab_id)

            if editor_frame.text_area.edit_modified():
                file_name = self.app.notebook.tab(editor_frame, "text").replace("*", "")
                result = messagebox.askyesnocancel("Save Changes?", f"Do you want to save changes to {file_name}?")
                
                if result is True: # Yes, save
                    if not self.app.file_manager.save_file():
                        # User cancelled the save dialog, so abort quitting
                        return 
                elif result is None: # Cancel
                    # User cancelled the quit operation
                    return

        # Clean up all tabs before destroying
        for tab_id in list(self.app.notebook.tabs()):
            editor_frame = self.app.notebook.nametowidget(tab_id)
            if hasattr(editor_frame, 'cleanup'):
                editor_frame.cleanup()

        # If we've gotten through the loop, it's safe to destroy
        self.app.destroy()
