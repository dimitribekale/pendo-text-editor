import tkinter as tk
import re
import threading

class PredictionHandler:
    def __init__(self, master, text_area, predictor, suggestion_box, change_callback):
        self.master = master
        self.text_area = text_area
        self.predictor = predictor
        self.suggestion_box = suggestion_box
        self.change_callback = change_callback

        self.after_id = None # For debouncing
        self.prediction_thread = None # To keep track of the prediction thread

        # Bindings that were previously in EditorFrame
        self.text_area.bind("<KeyRelease>", self._on_key_release)
        self.text_area.bind("<Button-1>", self._on_mouse_click)

    def _on_mouse_click(self, event=None):
        self.suggestion_box.hide()

    def _on_key_release(self, event=None):
        self.change_callback() # Update dirty state and line numbers

        # Hide suggestions immediately for certain keys
        if event and event.keysym in ("Escape", "Return", "Tab", "Left", "Right", "Up", "Down"):
            self.suggestion_box.hide()
            if event.keysym in ("Up", "Down"): # Allow arrow keys to navigate listbox if active
                if self.suggestion_box.is_active():
                    return # Don't hide, let listbox handle
            return

        # Cancel any pending prediction
        if self.after_id:
            self.master.after_cancel(self.after_id)

        # Schedule new prediction after a short delay
        self.after_id = self.master.after(300, self._perform_prediction) # 300ms debounce

    def _perform_prediction(self):
        cursor_index = self.text_area.index(tk.INSERT)
        line_text_before_cursor = self.text_area.get(f"{cursor_index} linestart", cursor_index)

        # Extract the current partial word being typed
        current_word_match = re.search(r"\b(\w*)$", line_text_before_cursor)
        partial_word = current_word_match.group(1) if current_word_match else ""

        # Construct prompt for the predictor
        prompt_end_index = f"{cursor_index}-{len(partial_word)}c" if partial_word else cursor_index
        prompt = self.text_area.get("1.0", prompt_end_index).strip()

        if not prompt and not partial_word: # Don't predict on empty editor
            self.suggestion_box.hide()
            return

        # Store current context to check relevance later
        self.current_prediction_context = {
            "prompt": prompt,
            "partial_word": partial_word,
            "cursor_index": cursor_index
        }

        # Run prediction in a separate thread
        self.prediction_thread = threading.Thread(
            target=self._run_prediction_in_thread,
            args=(prompt, partial_word, cursor_index)
        )
        self.prediction_thread.daemon = True # Allow program to exit even if thread is running
        self.prediction_thread.start()

    def _run_prediction_in_thread(self, prompt, partial_word, cursor_index):
        suggestions = self.predictor.predict(prompt)
        # Schedule UI update on the main thread
        self.master.after(0, lambda: self._update_suggestions_ui(suggestions, prompt, partial_word, cursor_index))

    def _update_suggestions_ui(self, suggestions, prompt_at_prediction_time, partial_word_at_prediction_time, cursor_index_at_prediction_time):
        # Check if the context is still valid (user hasn't typed something else)
        current_cursor_index = self.text_area.index(tk.INSERT)
        current_line_text_before_cursor = self.text_area.get(f"{current_cursor_index} linestart", current_cursor_index)
        current_partial_word_match = re.search(r"\b(\w*)$", current_line_text_before_cursor)
        current_partial_word = current_partial_word_match.group(1) if current_partial_word_match else ""

        current_prompt_end_index = f"{current_cursor_index}-{len(current_partial_word)}c" if current_partial_word else current_cursor_index
        current_prompt = self.text_area.get("1.0", current_prompt_end_index).strip()

        if (current_prompt != prompt_at_prediction_time or 
            current_partial_word != partial_word_at_prediction_time or 
            current_cursor_index != cursor_index_at_prediction_time):
            # Context changed, predictions are outdated
            self.suggestion_box.hide()
            return

        # Filter suggestions based on the partial word
        if partial_word_at_prediction_time:
            suggestions = [s for s in suggestions if s.lower().startswith(partial_word_at_prediction_time.lower())]

        if suggestions:
            x, y, _, height = self.text_area.bbox(tk.INSERT)
            self.suggestion_box.show(self.text_area.winfo_rootx() + x, self.text_area.winfo_rooty() + y + height, suggestions)
        else:
            self.suggestion_box.hide()

    def _accept_suggestion(self, event=None):
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
