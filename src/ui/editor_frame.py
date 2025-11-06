import tkinter as tk
from tkinter import ttk
import re
import threading
from .editor.line_numbers_widget import LineNumbers
from .suggestion_box import SuggestionBox
from .editor.prediction_handler import PredictionHandler

class EditorFrame(ttk.Frame):
    def __init__(self, master, theme, change_callback, predictor, **kwargs):
        super().__init__(master, **kwargs)
        self.theme = theme
        self.change_callback = change_callback

        # Core UI Components
        self.text_area = tk.Text(
            self, 
            wrap=tk.WORD, 
            undo=True, 
            bg=self.theme["text_bg"],
            fg=self.theme["text_fg"],
            insertbackground=self.theme["cursor"],
            selectbackground=self.theme["select_bg"]
        )
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.linenumbers = LineNumbers(self, self.text_area, theme=self.theme, width=30)
        self.linenumbers.pack(side=tk.LEFT, fill=tk.Y)

        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text_area.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_area.config(yscrollcommand=self._on_scroll)

        # Prediction and Suggestion Engine
        self.predictor = predictor
        self.suggestion_box = SuggestionBox(self, self._accept_suggestion, self.text_area)
        
        # Prediction Handler
        self.prediction_handler = PredictionHandler(
            master=self,
            text_area=self.text_area,
            predictor=self.predictor,
            suggestion_box=self.suggestion_box,
            change_callback=self.change_callback
        )

        # Event Bindings
        self.text_area.bind("<<Modified>>", self.change_callback)
        # Delegate key and mouse events to the prediction handler
        self.text_area.bind("<Button-1>", self.prediction_handler._on_mouse_click)
        self.text_area.bind("<KeyRelease>", self.prediction_handler._on_key_release)

    def _on_scroll(self, *args):
        self.scrollbar.set(*args)
        self.linenumbers.redraw()

    def _accept_suggestion(self, event=None):
        # This method is called by the SuggestionBox via prediction_handler
        if self.suggestion_box.is_active():
            selection = self.suggestion_box.get_selection()
            if selection:
                cursor_index = self.text_area.index(tk.INSERT)
                line_text_before_cursor = self.text_area.get(f"{cursor_index} linestart", cursor_index)
                
                # Find the current partial word to replace it
                current_word_match = re.search(r"\b(\w*)$", line_text_before_cursor)
                if current_word_match:
                    start_of_partial_word = f"{cursor_index}-{len(current_word_match.group(1))}c"
                    self.text_area.delete(start_of_partial_word, cursor_index)

                self.text_area.insert(tk.INSERT, selection + " ") # Insert suggestion followed by a space
                
                self.suggestion_box.hide()
                return "break" # Prevents default Tab/Enter behavior

    def get_text(self):
        return self.text_area.get(1.0, tk.END)

    def set_text(self, text):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(1.0, text)

    def set_font(self, font_family, font_size):
        self.text_area.config(font=(font_family, font_size))