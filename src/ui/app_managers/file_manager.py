import tkinter as tk
from tkinter import filedialog, messagebox
import os
import tempfile
import shutil
from ..editor_frame import EditorFrame
from backend.app_config import get_value

class FileManager:
    def __init__(self, app):
        self.app = app

    def _check_file_size(self, file_path, max_size_mb=10):
        """
        Check if file is withing acceptable limits.
        """
        try:
            file_size = os.path.getsize(file_path)
            max_size_bytes = max_size_mb * 1024 * 1024

            if file_size > max_size_bytes:
                messagebox.showerror(
                    "File Too Large",
                    f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds maximum allowed size ({max_size_mb} MB).\n\n"
                    f"Please use a specialized editor for large files."
                )
                return False
            return True
        except OSError as e:
            messagebox.showerror("Error", f"Cannot access file: {e}")
            return False
        
    def _read_file_safe(self, file_path):
        """
        Safely read file with encoding detection and error handling.
        Returns file contents as a string, or None if failed.
        """
        # Try UTF-8
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            # UTF-8 failed, try with Latin-1
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    content = file.read()
                    messagebox.showwarning(
                        "Encoding Warning",
                        "File was opened with Latin-1 encoding.\n"
                        "Some characters may not display correctly."
                    )
                    return content
            except Exception as e:
                messagebox.showerror(
                    "Encoding Error",
                    f"Cannot read file with supported encodings.\n\n{e}"
                )
                return None
        except Exception as e:
            messagebox.showerror("Error", f"Cannot read the file: {e}")

    def _write_file_atomic(self, file_path, content):
        """
        Atomically write content to file (prevents data loss on failure).
        Uses temporary file + rename strategy for atomic operation.
        
        Args:
            file_path: Destination file path
            content: Content to write
        
        Returns: True is successful, False otherwise.
        """
        try:
            # Create a temp file in the directory (same filesystem for atomic rename)
            dir_path = os.path.dirname(file_path) or "."
            fd, temp_path = tempfile.mkstemp(dir=dir_path, prefix=".tmp_", suffix=".txt")

            try:
                # Write to temp file
                with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                    temp_file.write(content)
                
                # Atomic rename (replaces original file)
                # On Windows, the target need to be removed first
                if os.name == "nt" and os.path.exists(file_path):
                    os.remove(file_path)
                shutil.move(temp_path, file_path)
                return True
            
            except Exception as e:
                # Clean up temp file on error
                try:
                    os.remove(temp_path)
                except:
                    pass
                raise e
        except Exception as e:
            messagebox.showerror(
                "Save Error",
                f"Cannot save file: {e}\n\n"
                f"Your changes have NOT been saved."
            )
            return False
        
    def new_file(self, event=None):
        editor_frame = EditorFrame(self.app.notebook,
                                   theme=self.app.theme,
                                   change_callback=self.app._on_change,
                                   predictor=self.app.model_manager.get_predictor())
        editor_frame.file_path = None
        self.app.notebook.add(editor_frame, text="Untitled")
        self.app.notebook.select(editor_frame)
        self.app._bind_context_menu(editor_frame)
        self.app._apply_settings(self.app.app_config) # Apply settings to new tab
        editor_frame.text_area.edit_modified(False)

    def open_file(self, event=None):
        file_path = filedialog.askopenfilename(defaultextension=".txt",
                                               filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            max_size = get_value(self.app.app_config, "file.max_size_mb", 10)

            if not self._check_file_size(file_path, max_size):
                return # File too large
            
            content = self._read_file_safe(file_path)
            if content is None:
                return # Read failed
            
            editor_frame = EditorFrame(self.app.notebook,
                                       theme=self.app.theme,
                                       change_callback=self.app._on_change,
                                       predictor=self.app.model_manager.get_predictor())
            editor_frame.file_path = file_path
            editor_frame.set_text(content)
            self.app.notebook.add(editor_frame,
                                  text=os.path.basename(file_path))
            self.app.notebook.select(editor_frame)
            self.app._bind_context_menu(editor_frame)
            self.app._apply_settings(self.app.app_config) # Apply settings to new tab
            editor_frame.text_area.edit_modified(False)

    def save_file(self, event=None):
        editor_frame = self.app.tab_manager.get_current_editor_frame()
        if not editor_frame:
            return False # Indicate failure

        if editor_frame.file_path:
            success = self._write_file_atomic(editor_frame.file_path,
                                              editor_frame.get_text())
            if success:
                editor_frame.text_area.edit_modified(False)
                self.app._on_change() # Update title
            return success
        else:
            return self.save_as_file()

    def save_as_file(self, event=None):
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