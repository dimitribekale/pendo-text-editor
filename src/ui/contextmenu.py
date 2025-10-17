import tkinter as tk

class ContextMenu(tk.Menu):
    def __init__(self, master, text_widget):
        super().__init__(master, tearoff=0)
        self.text_widget = text_widget

        self.add_command(label="Cut", command=self._cut)
        self.add_command(label="Copy", command=self._copy)
        self.add_command(label="Paste", command=self._paste)
        self.add_separator()
        self.add_command(label="Undo", command=self._undo)
        self.add_command(label="Redo", command=self._redo)

    def _cut(self):
        self.text_widget.event_generate("<<Cut>>")

    def _copy(self):
        self.text_widget.event_generate("<<Copy>>")

    def _paste(self):
        self.text_widget.event_generate("<<Paste>>")

    def _undo(self):
        self.text_widget.event_generate("<<Undo>>")

    def _redo(self):
        self.text_widget.event_generate("<<Redo>>")

    def show(self, event):
        self.tk_popup(event.x_root, event.y_root)
