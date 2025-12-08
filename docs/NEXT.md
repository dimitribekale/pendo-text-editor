# Next Fixes

## PHASE 1: Critical Fixes (Remaining)

### Fix #3: Remove Duplicate Event Bindings
**File:** `src/ui/editor_frame.py`
**Lines to remove:** 51-52
```python
# DELETE these two lines:
self.text_area.bind("<Button-1>", self.prediction_handler._on_mouse_click)
self.text_area.bind("<KeyRelease>", self.prediction_handler._on_key_release)
```
**Reason:** These events are already bound in PredictionHandler.__init__(), causing double execution.

---

### Fix #4: Implement ThreadPoolExecutor for Predictions
**File:** `src/ui/editor/prediction_handler.py`

**Step 1:** Add import at top of file
```python
from concurrent.futures import ThreadPoolExecutor
```

**Step 2:** In `__init__()` method (around line 14), add:
```python
self.executor = ThreadPoolExecutor(max_workers=1)
self.current_future = None
```

**Step 3:** Replace `_perform_prediction()` method (lines 38-75) with:
```python
def _perform_prediction(self):
    """Initiate prediction with proper thread management"""
    # Cancel previous prediction if still running
    if self.current_future and not self.current_future.done():
        self.current_future.cancel()

    prompt = self.text_area.get(1.0, tk.INSERT)
    partial_word = self._get_partial_word_at_cursor()
    cursor_index = self.text_area.index(tk.INSERT)

    # Submit to thread pool
    self.current_future = self.executor.submit(
        self._run_prediction_in_thread,
        prompt,
        partial_word,
        cursor_index
    )
```

**Step 4:** Add cleanup method:
```python
def cleanup(self):
    """Cleanup resources when tab is closed"""
    if self.current_future:
        self.current_future.cancel()
    self.executor.shutdown(wait=False)
```

---

## PHASE 2: Memory Leak Fixes

### Fix #5: Add EditorFrame Cleanup Method
**File:** `src/ui/editor_frame.py`

**Add this method to EditorFrame class:**
```python
def cleanup(self):
    """Clean up resources when tab is closed"""
    # Unbind event listeners
    self.text_area.unbind("<<Modified>>")

    # Cleanup prediction handler
    if hasattr(self, 'prediction_handler'):
        self.prediction_handler.cleanup()

    # Destroy suggestion box
    if hasattr(self, 'suggestion_box'):
        self.suggestion_box.destroy()

    # Clear references
    self.text_area = None
    self.prediction_handler = None
```

---

### Fix #6: Call Cleanup When Tabs Close
**File:** `src/ui/app_managers/tab_manager.py`

**Find `_close_current_tab()` method (around line 17)**

**Add before `self.app.notebook.forget(editor_frame)`:**
```python
# Clean up resources before closing tab
if hasattr(editor_frame, 'cleanup'):
    editor_frame.cleanup()
```

---

## PHASE 3: Performance Optimizations

### Fix #7: Optimize Line Number Widget
**File:** `src/ui/line_numbers_widget.py`

**Replace `redraw()` method with:**
```python
def redraw(self):
    """Redraw line numbers (optimized)"""
    i = self.text_widget.index("@0,0")
    while True:
        dline = self.text_widget.dlineinfo(i)
        if dline is None:
            break
        y = dline[1]
        linenum = str(i).split(".")[0]
        self.create_text(2, y, anchor="nw", text=linenum, tag="line_number")
        i = self.text_widget.index(f"{i}+1line")

    # Remove old line numbers that are out of view
    self.delete("line_number")
```

---

### Fix #8: Optimize Status Bar Updates
**File:** `src/ui/statusbar.py`

**Replace `update_status()` method with:**
```python
def update_status(self, text_area):
    """Update status bar (optimized with caching)"""
    # Use text widget's built-in count method (faster)
    char_count = int(text_area.index('end-1c').split('.')[1])

    # Get only visible content for word count (approximate)
    cursor_pos = text_area.index(tk.INSERT)

    self.position_label.config(text=f"Position: {cursor_pos}")
    self.char_label.config(text=f"Characters: {char_count}")
```

---

### Fix #9: Add Prediction Rate Limiting
**Already handled in Fix #4** - ThreadPoolExecutor with max_workers=1 ensures only one prediction runs at a time.

---

## PHASE 4: UX Enhancements

### Fix #10: Implement VS Code-Style Inline Ghost Text
**Files:** `src/ui/editor/prediction_handler.py`, `src/ui/suggestion_box.py`

**Step 1:** In `prediction_handler.py`, configure text widget tag for ghost text:
```python
# In __init__():
self.text_area.tag_config("ghost_text", foreground="gray")
```

**Step 2:** Replace suggestion box display with inline ghost text:
```python
def _display_inline_suggestion(self, suggestion):
    """Display suggestion as inline ghost text"""
    # Remove any existing ghost text
    self._clear_ghost_text()

    # Insert ghost text at cursor
    cursor_pos = self.text_area.index(tk.INSERT)
    self.text_area.insert(cursor_pos, suggestion, "ghost_text")
    self.text_area.mark_set("ghost_start", cursor_pos)
    self.text_area.mark_set("ghost_end", f"{cursor_pos}+{len(suggestion)}c")

def _clear_ghost_text(self):
    """Remove ghost text"""
    if self.text_area.tag_ranges("ghost_text"):
        self.text_area.delete("ghost_start", "ghost_end")

def _accept_ghost_text(self, event=None):
    """Accept ghost text on Tab press"""
    if self.text_area.tag_ranges("ghost_text"):
        # Remove ghost tag, making text permanent
        self.text_area.tag_remove("ghost_text", "ghost_start", "ghost_end")
        self.text_area.mark_unset("ghost_start")
        self.text_area.mark_unset("ghost_end")
        return "break"  # Prevent default Tab behavior
```

**Step 3:** Bind Tab key in `__init__()`:
```python
self.text_area.bind("<Tab>", self._accept_ghost_text)
```

**Step 4:** Clear ghost text on any other keypress - modify `_on_key_release()`:
```python
def _on_key_release(self, event):
    if event.keysym not in ("Tab", "Shift", "Control", "Alt"):
        self._clear_ghost_text()
    # ... rest of method
```

---

## PHASE 5: Code Quality

### Fix #11: Add Proper Error Handling
**File:** `src/ui/app_managers/file_manager.py`

**Replace bare except at lines 94-95:**
```python
except Exception as e:
    import logging
    logging.error(f"File operation error: {e}")
    from tkinter import messagebox
    messagebox.showerror("Error", f"Failed to complete operation: {e}")
```

---

### Fix #12: Add Logging Throughout
**Add at top of key files:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Add logging statements at critical points:**
- File operations: `logging.info(f"Saved file: {path}")`
- Predictions: `logging.debug(f"Prediction requested: {prompt[:50]}")`
- Errors: `logging.error(f"Error: {e}")`

---

### Fix #13: Add Docstrings
**Add comprehensive docstrings to all public methods:**
```python
def predict(self, prompt: str, max_length: int = 10) -> List[str]:
    """
    Generate text predictions based on prompt.

    Args:
        prompt: Input text to complete
        max_length: Maximum tokens to generate

    Returns:
        List of predicted completion strings
    """
```

---

## Testing After Each Phase

**After Phase 1:**
- Test config loading works
- Test typing continues smoothly with suggestions
- Test no crashes on rapid typing

**After Phase 2:**
- Open and close multiple tabs
- Check memory usage doesn't grow

**After Phase 3:**
- Open large file (1000+ lines)
- Test scrolling is smooth
- Test typing doesn't lag

**After Phase 4:**
- Test ghost text appears
- Test Tab accepts suggestion
- Test typing dismisses ghost text

**After Phase 5:**
- Check logs for errors
- Verify all exceptions are logged
