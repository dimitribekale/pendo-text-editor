"""
Utility modules for Pendo Text Editor.

This package contains reusable utility functions extracted from UI components
to reduce code duplication and improve testability.
"""

from .text_utils import extract_partial_word, get_prompt_before_cursor, replace_word_at_cursor
from .file_io_utils import check_file_size_limit, read_file_with_encoding_fallback, write_file_atomic
from .notebook_utils import NotebookOperations

__all__ = [
    'extract_partial_word',
    'get_prompt_before_cursor',
    'replace_word_at_cursor',
    'check_file_size_limit',
    'read_file_with_encoding_fallback',
    'write_file_atomic',
    'NotebookOperations',
]
