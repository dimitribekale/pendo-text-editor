"""Classifies key presses for the prediction handler."""

_MODIFIER_KEYS = frozenset({
    "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R",
    "Caps_Lock", "Win_L", "Win_R", "Meta_L", "Meta_R",
})

_NAVIGATION_KEYS = frozenset({
    "Escape", "Return", "Tab", "Left", "Right", "Up", "Down",
    "Home", "End", "Prior", "Next",
})


def dismisses_suggestion(keysym):
    """True for any key that should remove a visible suggestion."""
    return keysym not in _MODIFIER_KEYS


def requests_prediction(keysym):
    """True for keys that change the text and deserve a new suggestion."""
    return keysym not in _MODIFIER_KEYS and keysym not in _NAVIGATION_KEYS
