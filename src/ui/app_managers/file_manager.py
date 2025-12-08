import tkinter as tk
from tkinter import filedialog, messagebox
import os
import tempfile
import shutil
import logging
from ..editor_frame import EditorFrame
from ..utils.file_io_utils import check_file_size_limit, read_file_with_encoding_fallback, write_file_atomic
from backend.app_config import get_value

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileManager:
    def __init__(self, app):
        self.app = app

    def _check_file_size(self, file_path, max_size_mb=10):
        """Check if file is within acceptable limits (UI wrapper for utility)."""
        is_valid, error_msg = check_file_size_limit(file_path, max_size_mb)
        if not is_valid:
            messagebox.showerror("File Too Large", error_msg)
        return is_valid

    def _read_file_safe(self, file_path):
        """Safely read file with encoding detection (UI wrapper for utility)."""
        content, encoding, warning = read_file_with_encoding_fallback(file_path)

        if content is None:
            messagebox.showerror("Encoding Error", warning)
            return None

        if warning:
            messagebox.showwarning("Encoding Warning", warning)

        return content

    def _write_file_atomic(self, file_path, content):
        """Atomically write content to file (UI wrapper for utility)."""
        success, error_msg = write_file_atomic(file_path, content)
        if not success:
            messagebox.showerror("Save Error", error_msg)
        return success
        
    def _create_configured_editor_frame(self):
        """
        Create a new editor frame with standard configuration.

        Returns:
            EditorFrame: Configured editor frame ready to be added to notebook
        """
        editor_frame = EditorFrame(
            self.app.notebook,
            theme=self.app.theme,
            change_callback=self.app._on_change,
            predictor=self.app.model_manager.get_predictor()
        )

        # Apply standard post-creation configuration
        self.app._bind_context_menu(editor_frame)
        self.app._apply_settings(self.app.app_config)

        return editor_frame

    def new_file(self, event=None):
        """
        Create a new untitled file in a new tab.

        Args:
            event: Optional event object from UI binding
        """
        editor_frame = self._create_configured_editor_frame()
        editor_frame.file_path = None

        self.app.notebook.add(editor_frame, text="Untitled")
        self.app.notebook.select(editor_frame)
        editor_frame.text_area.edit_modified(False)

    def _create_editor_frame_with_file(self, file_path, content):
        """Create and configure an editor frame with file content."""
        editor_frame = self._create_configured_editor_frame()
        editor_frame.file_path = file_path
        editor_frame.set_text(content)

        self.app.notebook.add(editor_frame, text=os.path.basename(file_path))
        self.app.notebook.select(editor_frame)
        editor_frame.text_area.edit_modified(False)

    def open_file(self, event=None):
        """
        Open an existing file from disk and display it in a new tab.

        Args:
            event: Optional event object from UI binding
        """
        file_path = filedialog.askopenfilename(defaultextension=".txt",
                                               filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])

        if not file_path:
            return

        logging.info(f"Opening file: {file_path}")

        # Validate and read file
        max_size = get_value(self.app.app_config, "file.max_size_mb", 10)

        if not self._check_file_size(file_path, max_size):
            logging.warning(f"File too large: {file_path}")
            return

        content = self._read_file_safe(file_path)
        if content is None:
            logging.error(f"Failed to read file: {file_path}")
            return

        # Create editor frame with loaded content
        self._create_editor_frame_with_file(file_path, content)
        logging.info(f"Successfully opened file: {file_path}")

    def save_file(self, event=None):
        """
        Save the current file. If no file path exists, calls save_as_file.

        Args:
            event: Optional event object from UI binding

        Returns:
            bool: True if save succeeded, False otherwise
        """
        editor_frame = self.app.tab_manager.get_current_editor_frame()
        if not editor_frame:
            return False # Indicate failure

        if editor_frame.file_path:
            logging.info(f"Saving file: {editor_frame.file_path}")
            success = self._write_file_atomic(editor_frame.file_path,
                                              editor_frame.get_text())
            if success:
                editor_frame.text_area.edit_modified(False)
                self.app._on_change() # Update title
                logging.info(f"Successfully saved file: {editor_frame.file_path}")
            else:
                logging.error(f"Failed to save file: {editor_frame.file_path}")
            return success
        else:
            return self.save_as_file()

    def save_as_file(self, event=None):
        """
        Save the current file with a new name/location.

        Args:
            event: Optional event object from UI binding

        Returns:
            bool: True if save succeeded, False otherwise
        """
        editor_frame = self.app.tab_manager.get_current_editor_frame()
        if not editor_frame:
            return False # Indicate failure

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            success = self._write_file_atomic(file_path, editor_frame.get_text())
            if success:
                editor_frame.file_path = file_path
                self.app.notebook.tab(editor_frame, text=os.path.basename(file_path))
                editor_frame.text_area.edit_modified(False)
                self.app._on_change()
            return success
        return False # Indicate failure (user cancelled)