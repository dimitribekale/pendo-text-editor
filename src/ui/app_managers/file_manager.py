"""Creates, opens and saves documents, one per tab."""

import logging
import os
from tkinter import filedialog, messagebox

from backend.app_config import get_value

from ..editor_frame import EditorFrame
from ..utils.file_io_utils import (
    check_file_size_limit,
    read_file_with_encoding_fallback,
    write_file_atomic,
)

logger = logging.getLogger(__name__)
_FILE_TYPES = [("Text Files", "*.txt"), ("All Files", "*.*")]


class FileManager:
    """Handles the File menu actions for the application window."""

    def __init__(self, app):
        self.app = app

    def new_file(self, event=None):
        """Open an empty, untitled tab."""
        self._add_tab("Untitled", file_path=None, content="")

    def open_file(self, event=None):
        """Ask for a file and show it in a new tab."""
        file_path = filedialog.askopenfilename(defaultextension=".txt", filetypes=_FILE_TYPES)
        if not file_path:
            return
        content = self._read(file_path)
        if content is not None:
            self._add_tab(os.path.basename(file_path), file_path, content)
            logger.info("Opened %s", file_path)

    def save_file(self, event=None):
        """Save the current tab, asking for a name if it has none; True if saved."""
        editor_frame = self.app.tab_manager.get_current_editor_frame()
        if not editor_frame:
            return False
        if not editor_frame.file_path:
            return self.save_as_file()
        return self._write(editor_frame, editor_frame.file_path)

    def save_as_file(self, event=None):
        """Save the current tab under a name the user chooses; True if saved."""
        editor_frame = self.app.tab_manager.get_current_editor_frame()
        if not editor_frame:
            return False
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=_FILE_TYPES)
        if not file_path or not self._write(editor_frame, file_path):
            return False
        editor_frame.file_path = file_path
        self.app.notebook.tab(editor_frame, text=os.path.basename(file_path))
        return True

    def _add_tab(self, title, file_path, content):
        editor_frame = EditorFrame(
            self.app.notebook,
            theme=self.app.theme,
            change_callback=self.app._on_change,
            predictor=self.app.model_manager.get_predictor(),
            interaction_log=self.app.interaction_log,
        )
        editor_frame.file_path = file_path
        if content:
            editor_frame.set_text(content)
        self.app._bind_context_menu(editor_frame)
        self.app.notebook.add(editor_frame, text=title)
        self.app.notebook.select(editor_frame)
        self.app._apply_settings(self.app.app_config)
        editor_frame.text_area.edit_modified(False)

    def _read(self, file_path):
        """Return the file's text, or None after telling the user why not."""
        max_size_mb = get_value(self.app.app_config, "file.max_file_size_mb", 100)
        is_valid, error = check_file_size_limit(file_path, max_size_mb)
        if not is_valid:
            messagebox.showerror("File Too Large", error)
            return None
        content, _, warning = read_file_with_encoding_fallback(file_path)
        if content is None:
            messagebox.showerror("Encoding Error", warning)
        elif warning:
            messagebox.showwarning("Encoding Warning", warning)
        return content

    def _write(self, editor_frame, file_path):
        success, error = write_file_atomic(file_path, editor_frame.get_text())
        if not success:
            messagebox.showerror("Save Error", error)
            return False
        editor_frame.text_area.edit_modified(False)
        self.app._on_change()
        logger.info("Saved %s", file_path)
        return True
