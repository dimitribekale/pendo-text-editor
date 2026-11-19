"""Reads and writes documents; reports problems instead of showing dialogs."""

import logging
import os
import tempfile

logger = logging.getLogger(__name__)


def check_file_size_limit(file_path, max_size_mb=10):
    """Check that a file is small enough to open.

    Returns:
        tuple[bool, str | None]: (is_valid, error message when it is not).
    """
    try:
        size_mb = os.path.getsize(file_path) / 1024 / 1024
    except OSError as error:
        return False, f"Cannot access file: {error}"
    if size_mb > max_size_mb:
        return False, (
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed size "
            f"({max_size_mb} MB).\n\nPlease use a specialized editor for large files."
        )
    return True, None


def read_file_with_encoding_fallback(file_path):
    """Read a text file as UTF-8, falling back to Latin-1.

    Returns:
        tuple[str | None, str | None, str | None]: (content, encoding used,
        warning or error message). Content is None when reading failed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read(), "utf-8", None
    except UnicodeDecodeError:
        pass
    except OSError as error:
        return None, None, f"Cannot read the file: {error}"
    try:
        with open(file_path, "r", encoding="latin-1") as file:
            warning = (
                "File was opened with Latin-1 encoding.\n"
                "Some characters may not display correctly."
            )
            return file.read(), "latin-1", warning
    except OSError as error:
        return None, None, f"Cannot read file with supported encodings.\n\n{error}"


def write_file_atomic(file_path, content):
    """Write through a temporary file so a failed save keeps the original.

    Returns:
        tuple[bool, str | None]: (success, error message when it failed).
    """
    directory = os.path.dirname(file_path) or "."
    try:
        handle, temp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".txt")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
                temp_file.write(content)
            os.replace(temp_path, file_path)
        except OSError:
            _remove_quietly(temp_path)
            raise
    except OSError as error:
        return False, f"Cannot save file: {error}\n\nYour changes have NOT been saved."
    return True, None


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        logger.exception("Failed to clean up temporary file %s", path)
