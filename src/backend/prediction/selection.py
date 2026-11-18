"""Decides whether a prediction is worth showing to the user."""


def choose_completion(suggestions, prefix, min_confidence):
    """Pick the text to offer after the cursor, or None to stay quiet.

    Only the top suggestion is considered. Nothing is offered when the model
    is not confident enough, or when it thinks the word is already complete.

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
    return best.word[len(prefix):] or None
