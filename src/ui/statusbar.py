import tkinter as tk
from tkinter import ttk

class StatusBar(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.line_col_label = ttk.Label(self, text="Line: 1, Column: 1")
        self.line_col_label.pack(side=tk.RIGHT, padx=5)

        self.char_count_label = ttk.Label(self, text="Chars: 0")
        self.char_count_label.pack(side=tk.RIGHT, padx=5)

        self.word_count_label = ttk.Label(self, text="Words: 0")
        self.word_count_label.pack(side=tk.RIGHT, padx=5)

    def update_status(self, text_area):
        """Update status bar (optimized with caching)"""
        # Get cursor position
        cursor_pos = text_area.index(tk.INSERT)
        line, col = cursor_pos.split('.')
        self.line_col_label.config(text=f"Line: {line}, Column: {int(col) + 1}")

        # Use text widget's built-in count method (faster)
        char_count = int(text_area.index('end-1c').split('.')[1])

        # Get word count more efficiently
        text_content = text_area.get(1.0, tk.END)
        word_count = len(text_content.split())

        self.char_count_label.config(text=f"Chars: {char_count}")
        self.word_count_label.config(text=f"Words: {word_count}")
