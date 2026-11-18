"""Shared fixtures."""

import tkinter as tk

import pytest


@pytest.fixture(scope="session")
def tk_root():
    """One hidden window for the whole run; skipped without a display."""
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def text_area(tk_root):
    """A fresh, real Text widget for each test."""
    widget = tk.Text(tk_root, undo=True)
    widget.pack()
    yield widget
    widget.destroy()
