import tkinter as tk
import re
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from ..utils.text_utils import extract_partial_word, get_prompt_before_cursor

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
        """Handle key release events and trigger predictions."""
        self.change_callback() # Update dirty state and line numbers

        # Clear ghost text on typing (except modifier keys)
        if self._should_clear_ghost_text(event):
            self._clear_ghost_text()

        # Check for keys that should stop prediction
        if self._should_skip_prediction(event):
            self.suggestion_box.hide()
            return

        # Debounce and schedule prediction
        self._schedule_debounced_prediction()

    def _should_clear_ghost_text(self, event):
        """Determine if ghost text should be cleared for this key."""
        if not event:
            return False
        modifier_keys = ("Tab", "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R")
        return event.keysym not in modifier_keys

    def _should_skip_prediction(self, event):
        """Determine if prediction should be skipped for this key."""
        if not event:
            return False
        skip_keys = ("Escape", "Return", "Tab", "Left", "Right", "Up", "Down")
        if event.keysym not in skip_keys:
            return False

        # Special case: allow arrow keys if suggestion box is active
        if event.keysym in ("Up", "Down") and self.suggestion_box.is_active():
            return False

        return True

    def _schedule_debounced_prediction(self):
        """Cancel pending prediction and schedule new one with debounce delay."""
        if self.after_id:
            self.master.after_cancel(self.after_id)
        self.after_id = self.master.after(300, self._perform_prediction)

    def _perform_prediction(self):
        """Initiate prediction with proper thread management"""
        # Cancel previous prediction if still running
        if self.current_future and not self.current_future.done():
            self.current_future.cancel()

        cursor_index = self.text_area.index(tk.INSERT)

        # Extract the current partial word being typed
        partial_word, _ = extract_partial_word(self.text_area, cursor_index)

        # Construct prompt for the predictor (excluding partial word)
        prompt = get_prompt_before_cursor(self.text_area, cursor_index, exclude_partial_word=True)

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
        """Update UI with suggestions if context is still valid."""
        # Validate context hasn't changed
        if not self._is_context_still_valid(prompt_at_prediction_time,
                                            partial_word_at_prediction_time,
                                            cursor_index_at_prediction_time):
            self.suggestion_box.hide()
            return

        # Filter and display suggestions
        filtered_suggestions = self._filter_suggestions(suggestions, partial_word_at_prediction_time)

        if filtered_suggestions:
            self._show_best_suggestion(filtered_suggestions[0], partial_word_at_prediction_time)
        else:
            self._clear_ghost_text()

    def _is_context_still_valid(self, expected_prompt, expected_partial_word, expected_cursor_index):
        """Check if context hasn't changed since prediction was requested."""
        current_cursor_index = self.text_area.index(tk.INSERT)
        current_partial_word, _ = extract_partial_word(self.text_area, current_cursor_index)
        current_prompt = get_prompt_before_cursor(self.text_area, current_cursor_index, exclude_partial_word=True)

        return (current_prompt == expected_prompt and
                current_partial_word == expected_partial_word and
                current_cursor_index == expected_cursor_index)

    def _filter_suggestions(self, suggestions, partial_word):
        """Filter suggestions based on partial word prefix matching."""
        if not partial_word:
            return suggestions
        return [s for s in suggestions if s.lower().startswith(partial_word.lower())]

    def _show_best_suggestion(self, suggestion, partial_word):
        """Display the best suggestion as inline ghost text."""
        # Remove the partial word prefix from suggestion
        if partial_word and suggestion.lower().startswith(partial_word.lower()):
            ghost_text = suggestion[len(partial_word):]
        else:
            ghost_text = suggestion
        self._display_inline_suggestion(ghost_text)

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
