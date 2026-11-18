"""Ghost text must never behave like a real edit until accepted."""

import tkinter as tk

from ui.editor.ghost_text import GhostText


def _content(text_area):
    return text_area.get("1.0", "end-1c")


def _type(text_area, text):
    text_area.insert(tk.INSERT, text)
    text_area.edit_separator()


def test_show_keeps_cursor_before_the_suggestion(text_area):
    _type(text_area, "gui")
    GhostText(text_area).show("tar")
    assert _content(text_area) == "guitar"
    assert text_area.index(tk.INSERT) == "1.3"


def test_clear_restores_the_document(text_area):
    _type(text_area, "gui")
    ghost = GhostText(text_area)
    ghost.show("tar")
    ghost.clear()
    assert _content(text_area) == "gui"
    assert not ghost.visible


def test_typing_while_visible_does_not_lose_the_typed_character(text_area):
    _type(text_area, "gui")
    ghost = GhostText(text_area)
    ghost.show("tar")
    text_area.insert(tk.INSERT, "d")
    ghost.clear()
    assert _content(text_area) == "guid"


def test_show_does_not_mark_the_document_modified(text_area):
    text_area.edit_modified(False)
    GhostText(text_area).show("tar")
    assert not text_area.edit_modified()


def test_accept_inserts_text_and_moves_cursor_to_its_end(text_area):
    _type(text_area, "gui")
    ghost = GhostText(text_area)
    ghost.show("tar")
    assert ghost.accept() is True
    assert _content(text_area) == "guitar"
    assert text_area.index(tk.INSERT) == "1.6"
    assert not ghost.visible


def test_accept_without_suggestion_reports_false(text_area):
    assert GhostText(text_area).accept() is False


def test_undo_after_accept_removes_only_the_completion(text_area):
    _type(text_area, "gui")
    ghost = GhostText(text_area)
    ghost.show("tar")
    ghost.accept()
    text_area.edit_undo()
    assert _content(text_area) == "gui"


def test_dismissed_suggestion_leaves_no_trace_in_undo_history(text_area):
    _type(text_area, "gui")
    ghost = GhostText(text_area)
    ghost.show("tar")
    ghost.clear()
    text_area.edit_undo()
    assert _content(text_area) == ""
