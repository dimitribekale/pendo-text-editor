import tkinter as tk
from tkinter import ttk

class LineNumbers(tk.Canvas):
    def __init__(self, master, text_widget, theme, *args, **kwargs):
        super().__init__(master, *args, **kwargs, bg=theme["line_number_bg"], highlightthickness=0)
        self.text_widget = text_widget
        self.theme = theme
        self.redraw()

    def redraw(self, *args):
        """Redraw line numbers (optimized)"""
        # Delete only old line numbers, not all canvas items
        self.delete("line_number")

        i = self.text_widget.index("@0,0")
        while True:
            dline = self.text_widget.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.create_text(self.winfo_width() - 5, y, anchor="ne", text=linenum, fill=self.theme["line_number_fg"], tag="line_number")
            i = self.text_widget.index(f"{i}+1line")
