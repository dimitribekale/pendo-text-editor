"""Decides whether a prediction is worth showing to the user."""

# Accepting costs one key press (Tab), so a one-letter completion saves
# nothing and only adds visual noise.
_MIN_COMPLETION_LENGTH = 2


def choose_completion(suggestions, prefix, min_confidence):
    """Pick the text to offer after the cursor, or None to stay quiet.

    Only the top suggestion is considered. Nothing is offered when the model
    is not confident enough, when it thinks the word is already complete, or
    when accepting would not save any typing.

    Args:
        suggestions: Ranked `Suggestion` list from a predictor.
        prefix: Letters of the current word typed so far.
        min_confidence: Minimum probability required to show a suggestion.

    Returns:
        str | None: The remaining letters of the word to display.
    """
    if not suggestions:
        return None
    best = suggestions[0]
    if best.probability < min_confidence:
        return None
    if not best.word.startswith(prefix):
        return None
    completion = best.word[len(prefix):]
    return completion if len(completion) >= _MIN_COMPLETION_LENGTH else None
