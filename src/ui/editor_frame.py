"""One editor tab: the text area, its line numbers and its predictions."""

import tkinter as tk
from tkinter import ttk

from .editor.line_numbers_widget import LineNumbers
from .editor.prediction_handler import PredictionHandler


class EditorFrame(ttk.Frame):
    """A text area with line numbers, a scrollbar and inline suggestions."""

    def __init__(self, master, theme, change_callback, predictor, interaction_log=None, **kwargs):
        super().__init__(master, **kwargs)
        self.file_path = None
        self.text_area = tk.Text(
            self,
            wrap=tk.WORD,
            undo=True,
            bg=theme["text_bg"],
            fg=theme["text_fg"],
            insertbackground=theme["cursor"],
            selectbackground=theme["select_bg"],
        )
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.linenumbers = LineNumbers(self, self.text_area, theme=theme, width=30)
        self.linenumbers.pack(side=tk.LEFT, fill=tk.Y)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text_area.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=self._on_scroll)

        self.prediction_handler = PredictionHandler(
            master=self,
            text_area=self.text_area,
            predictor=predictor,
            change_callback=change_callback,
            interaction_log=interaction_log,
        )
        self.text_area.bind("<<Modified>>", change_callback)

    def get_text(self):
        """The document text; an unaccepted suggestion is never part of it."""
        self.prediction_handler.dismiss()
        return self.text_area.get(1.0, tk.END)

    def set_text(self, text):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(1.0, text)

    def set_font(self, font_family, font_size):
        self.text_area.config(font=(font_family, font_size))

    def cleanup(self):
        """Release the prediction worker when the tab is closed."""
        self.text_area.unbind("<<Modified>>")
        self.prediction_handler.cleanup()

    def _on_scroll(self, *args):
        self.scrollbar.set(*args)
        self.linenumbers.redraw()
