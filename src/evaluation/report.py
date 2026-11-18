"""Formats a simulation result for the terminal."""


def format_report(title, result):
    """Return a short, aligned summary of one simulation run."""
    rows = [
        ("Characters typed", f"{result.characters:,}"),
        ("Words", f"{result.words:,}"),
        ("Keystroke savings", f"{result.keystroke_savings:.1%}"),
        ("Suggestion precision", f"{result.precision:.1%}"),
        ("Words completed", f"{result.word_coverage:.1%}"),
        ("Suggestions shown", f"{result.shown:,}"),
        ("Latency mean", f"{result.latency_mean_ms:.0f} ms"),
        ("Latency p95", f"{result.latency_p95_ms:.0f} ms"),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [title, "-" * len(title)]
    lines += [f"{label.ljust(width)}  {value}" for label, value in rows]
    return "\n".join(lines)
