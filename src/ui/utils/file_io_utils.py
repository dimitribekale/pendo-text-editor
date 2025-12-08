"""
File I/O utilities for Pendo Text Editor.

Provides reusable file operation functions with proper error handling.
These are pure functions that don't directly interact with UI - they return
success/failure tuples that calling code can use to display appropriate messages.
"""

import os
import tempfile
import shutil
import logging


def check_file_size_limit(file_path, max_size_mb=10):
    """
    Check if file size is within acceptable limits.

    Args:
        file_path: Path to file to check
        max_size_mb: Maximum allowed size in megabytes

    Returns:
        tuple: (is_valid, error_message)
            - is_valid: True if file is within limit, False otherwise
            - error_message: Descriptive error message if invalid, None if valid

    Example:
        is_valid, error_msg = check_file_size_limit("/path/to/file.txt", 10)
        if not is_valid:
            print(error_msg)
    """
    try:
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024

        if file_size > max_size_bytes:
            error_msg = (
                f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds "
                f"maximum allowed size ({max_size_mb} MB).\n\n"
                f"Please use a specialized editor for large files."
            )
            return False, error_msg

        return True, None

    except OSError as e:
        return False, f"Cannot access file: {e}"


def read_file_with_encoding_fallback(file_path):
    """
    Safely read file with encoding detection and error handling.

    Attempts to read file with UTF-8 encoding first, falls back to Latin-1
    if UTF-8 fails. This handles most common encoding scenarios.

    Args:
        file_path: Path to file to read

    Returns:
        tuple: (content, encoding_used, warning_message)
            - content: File content as string, or None if read failed
            - encoding_used: The encoding that succeeded ("utf-8", "latin-1", or None)
            - warning_message: Warning or error message (None if successful UTF-8 read)

    Example:
        content, encoding, warning = read_file_with_encoding_fallback("/path/file.txt")
        if content is None:
            print(f"Error: {warning}")
        elif warning:
            print(f"Warning: {warning}")
    """
    # Try UTF-8 first (most common)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read(), "utf-8", None

    except UnicodeDecodeError:
        # UTF-8 failed, try with Latin-1
        try:
            with open(file_path, "r", encoding="latin-1") as file:
                content = file.read()
                warning = (
                    "File was opened with Latin-1 encoding.\n"
                    "Some characters may not display correctly."
                )
                return content, "latin-1", warning

        except Exception as e:
            error_msg = f"Cannot read file with supported encodings.\n\n{e}"
            return None, None, error_msg

    except Exception as e:
        return None, None, f"Cannot read the file: {e}"


def write_file_atomic(file_path, content):
    """
    Atomically write content to file (prevents data loss on failure).

    Uses temporary file + rename strategy for atomic operation. This ensures
    that if the write fails, the original file remains intact.

    The atomic rename strategy:
    1. Write to temporary file in same directory
    2. If write succeeds, atomically rename temp file to target
    3. If write fails, temp file is cleaned up, original untouched

    Args:
        file_path: Destination file path
        content: Content to write (string)

    Returns:
        tuple: (success, error_message)
            - success: True if write succeeded, False otherwise
            - error_message: Error description if failed, None if successful

    Example:
        success, error = write_file_atomic("/path/file.txt", "content")
        if not success:
            print(f"Write failed: {error}")

    Note:
        On Windows, the target file must be removed before rename.
        This is handled automatically.
    """
    try:
        # Create temp file in same directory (same filesystem for atomic rename)
        dir_path = os.path.dirname(file_path) or "."
        fd, temp_path = tempfile.mkstemp(dir=dir_path, prefix=".tmp_", suffix=".txt")

        try:
            # Write to temp file
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(content)

            # Atomic rename (replaces original file)
            # On Windows, target must be removed first
            if os.name == "nt" and os.path.exists(file_path):
                os.remove(file_path)
            shutil.move(temp_path, file_path)

            return True, None

        except Exception as e:
            # Clean up temp file on error
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logging.error(f"Failed to clean up temporary file {temp_path}: {cleanup_error}")
            raise e

    except Exception as e:
        error_msg = (
            f"Cannot save file: {e}\n\n"
            f"Your changes have NOT been saved."
        )
        return False, error_msg
