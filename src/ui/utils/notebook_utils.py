"""
Notebook operation utilities for Pendo Text Editor.

Provides a helper class that encapsulates common notebook/tab operations,
reducing direct dependency on the notebook widget in managers.
"""


class NotebookOperations:
    """Helper for common notebook operations."""

    def __init__(self, notebook):
        """
        Initialize with notebook widget.

        Args:
            notebook: ttk.Notebook widget instance
        """
        self.notebook = notebook

    def get_all_tab_ids(self):
        """
        Get list of all tab IDs.

        Returns:
            list: List of tab ID strings
        """
        return list(self.notebook.tabs())

    def get_editor_frame(self, tab_id):
        """
        Get editor frame widget for tab ID.

        Args:
            tab_id: Tab identifier string

        Returns:
            EditorFrame: Editor frame widget
        """
        return self.notebook.nametowidget(tab_id)

    def get_current_editor_frame(self):
        """
        Get currently selected editor frame.

        Returns:
            EditorFrame or None: Current editor frame, or None if no tabs exist
        """
        if not self.notebook.tabs():
            return None
        selected_tab = self.notebook.select()
        return self.get_editor_frame(selected_tab)

    def select_tab(self, tab_id):
        """
        Select a specific tab.

        Args:
            tab_id: Tab identifier string
        """
        self.notebook.select(tab_id)

    def close_tab(self, editor_frame):
        """
        Remove a tab from the notebook.

        Args:
            editor_frame: EditorFrame widget to remove
        """
        self.notebook.forget(editor_frame)

    def has_tabs(self):
        """
        Check if notebook has any tabs.

        Returns:
            bool: True if there are tabs, False otherwise
        """
        return bool(self.notebook.tabs())

    def get_tab_title(self, editor_frame):
        """
        Get the display title for a tab.

        Args:
            editor_frame: EditorFrame widget

        Returns:
            str: Tab title text
        """
        return self.notebook.tab(editor_frame, "text")

    def set_tab_title(self, editor_frame, title):
        """
        Set the display title for a tab.

        Args:
            editor_frame: EditorFrame widget
            title: New title text
        """
        self.notebook.tab(editor_frame, text=title)

    def add_tab(self, editor_frame, title):
        """
        Add a new tab to the notebook.

        Args:
            editor_frame: EditorFrame widget to add
            title: Tab title text
        """
        self.notebook.add(editor_frame, text=title)

    def select_editor_frame(self, editor_frame):
        """
        Select a tab by its editor frame.

        Args:
            editor_frame: EditorFrame widget to select
        """
        self.notebook.select(editor_frame)
