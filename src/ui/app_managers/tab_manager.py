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

    def _handle_unsaved_changes(self, editor_frame):
        """
        Prompt user to save changes if needed.

        Returns:
            bool: True to continue, False to cancel operation
        """
        if not editor_frame.text_area.edit_modified():
            return True  # No changes, continue

        file_name = self.app.notebook.tab(editor_frame, "text").replace("*", "")
        result = messagebox.askyesnocancel(
            "Save Changes?",
            f"Do you want to save changes to {file_name}?"
        )

        if result is True:  # Yes, save
            return self.app.file_manager.save_file()
        elif result is False:  # No, don't save
            return True
        else:  # Cancel
            return False

    def _cleanup_all_tabs(self):
        """Clean up resources for all tabs."""
        for tab_id in list(self.app.notebook.tabs()):
            editor_frame = self.app.notebook.nametowidget(tab_id)
            if hasattr(editor_frame, 'cleanup'):
                editor_frame.cleanup()

    def _close_current_tab(self, event=None):
        editor_frame = self.get_current_editor_frame()
        if not editor_frame:
            return

        if not self._handle_unsaved_changes(editor_frame):
            return  # User cancelled

        # Clean up resources before closing tab
        if hasattr(editor_frame, 'cleanup'):
            editor_frame.cleanup()

        self.app.notebook.forget(editor_frame)
        if not self.app.notebook.tabs():
            self.app.file_manager.new_file() # Create a new tab if the last one is closed

    def _on_quit(self):
        """Handle application quit with save prompts."""
        # Check all tabs for unsaved changes
        for tab_id in list(self.app.notebook.tabs()):
            self.app.notebook.select(tab_id)
            editor_frame = self.app.notebook.nametowidget(tab_id)

            if not self._handle_unsaved_changes(editor_frame):
                # User cancelled, abort quit
                return

        # Clean up all tabs before destroying
        self._cleanup_all_tabs()
        self.app.destroy()
