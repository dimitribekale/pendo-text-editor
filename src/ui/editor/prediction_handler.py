import tkinter as tk
import re
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PredictionHandler:
    """
    Handles text prediction and inline ghost text suggestions for the text editor.

    Manages prediction threads, displays suggestions as inline ghost text,
    and handles user interaction with suggestions (Tab to accept, typing to dismiss).
    """
    def __init__(self, master, text_area, predictor, suggestion_box, change_callback):
        """
        Initialize the prediction handler.

        Args:
            master: Parent widget
            text_area: Text widget to monitor and display suggestions in
            predictor: Prediction model instance
            suggestion_box: Suggestion box widget (legacy, kept for compatibility)
            change_callback: Callback function to trigger on text changes
        """
        self.master = master
        self.text_area = text_area
        self.predictor = predictor
        self.suggestion_box = suggestion_box
        self.change_callback = change_callback

        self.after_id = None # For debouncing
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.current_future = None

        # Configure ghost text tag for inline suggestions
        self.text_area.tag_config("ghost_text", foreground="gray")

        # Bindings that were previously in EditorFrame
        self.text_area.bind("<KeyRelease>", self._on_key_release)
        self.text_area.bind("<Button-1>", self._on_mouse_click)
        self.text_area.bind("<Tab>", self._accept_ghost_text)

    def _on_mouse_click(self, event=None):
        self.suggestion_box.hide()
        self._clear_ghost_text()

    def _on_key_release(self, event=None):
        self.change_callback() # Update dirty state and line numbers

        # Clear ghost text on any keypress except Tab, Shift, Control, Alt
        if event and event.keysym not in ("Tab", "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R"):
            self._clear_ghost_text()

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
        """Initiate prediction with proper thread management"""
        # Cancel previous prediction if still running
        if self.current_future and not self.current_future.done():
            self.current_future.cancel()

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

        # Submit to thread pool
        self.current_future = self.executor.submit(
            self._run_prediction_in_thread,
            prompt,
            partial_word,
            cursor_index
        )

    def _run_prediction_in_thread(self, prompt, partial_word, cursor_index):
        logging.debug(f"Prediction requested - prompt length: {len(prompt)}, partial: '{partial_word}'")
        suggestions = self.predictor.predict(prompt)
        logging.debug(f"Prediction completed - {len(suggestions)} suggestions received")
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
            # Use inline ghost text instead of suggestion box
            suggestion = suggestions[0]  # Take the first suggestion
            # Remove the partial word from the suggestion
            if partial_word_at_prediction_time and suggestion.lower().startswith(partial_word_at_prediction_time.lower()):
                ghost_text = suggestion[len(partial_word_at_prediction_time):]
            else:
                ghost_text = suggestion
            self._display_inline_suggestion(ghost_text)
        else:
            self._clear_ghost_text()

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

    def _display_inline_suggestion(self, suggestion):
        """Display suggestion as inline ghost text"""
        # Remove any existing ghost text
        self._clear_ghost_text()

        # Insert ghost text at cursor
        cursor_pos = self.text_area.index(tk.INSERT)
        self.text_area.insert(cursor_pos, suggestion, "ghost_text")
        self.text_area.mark_set("ghost_start", cursor_pos)
        self.text_area.mark_set("ghost_end", f"{cursor_pos}+{len(suggestion)}c")
        logging.debug(f"Ghost text displayed: '{suggestion}' at position {cursor_pos}")

    def _clear_ghost_text(self):
        """Remove ghost text"""
        if self.text_area.tag_ranges("ghost_text"):
            try:
                self.text_area.delete("ghost_start", "ghost_end")
                logging.debug("Ghost text cleared")
            except tk.TclError:
                # Marks might not exist, ignore
                pass

    def _accept_ghost_text(self, event=None):
        """Accept ghost text on Tab press"""
        if self.text_area.tag_ranges("ghost_text"):
            # Remove ghost tag, making text permanent
            self.text_area.tag_remove("ghost_text", "ghost_start", "ghost_end")
            try:
                self.text_area.mark_unset("ghost_start")
                self.text_area.mark_unset("ghost_end")
            except tk.TclError:
                # Marks might not exist, ignore
                pass
            logging.debug("Ghost text accepted via Tab key")
            return "break"  # Prevent default Tab behavior
        # If no ghost text, allow default Tab behavior
        logging.debug("Tab pressed but no ghost text present")
        return None

    def cleanup(self):
        """Cleanup resources when tab is closed"""
        if self.current_future:
            self.current_future.cancel()
        self.executor.shutdown(wait=False)
