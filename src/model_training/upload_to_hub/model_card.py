"""Writes the Hugging Face model card for a fine-tuned writing model."""

import json

_HEADER = """---
license: apache-2.0
language: en
base_model: {base_model}
pipeline_tag: text-generation
tags: [autocomplete, word-prediction, pendo]
---

# {title}

`{base_model}` fine-tuned for inline word completion in the
[Pendo text editor](https://github.com/dimitribekale/pendo-text-editor).

## Training

Three epochs of causal language modelling on about 200 MB of English text
from six kinds of writing: emails (AESLC), blogs (Blog Authorship Corpus),
fiction (WritingPrompts), academic abstracts (arXiv), news (CNN/DailyMail)
and encyclopedia articles (Wikipedia). Documents were split into train,
validation and test sets before chunking into 512-token examples.

## Results

Keystroke savings come from Pendo's typing simulator, which replays
held-out text one keystroke at a time and accepts a suggestion only when it
is exactly the intended word (3,000 characters per genre).
"""


def _table(genres, columns):
    names = list(columns)
    lines = ["| Genre | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for genre in genres:
        cells = " | ".join(columns[name][genre] for name in names)
        lines.append(f"| {genre.capitalize()} | {cells} |")
    return "\n".join(lines)


def build_model_card(run_dir, title, base_model):
    """Return the card text for the run stored in `run_dir`."""
    savings = json.loads((run_dir / "simulator.json").read_text())
    losses = json.loads((run_dir / "metrics.json").read_text())
    genres = list(savings["after"])
    columns = {
        "Keystroke savings before": {g: f"{savings['before'][g]:.1f}%" for g in genres},
        "Keystroke savings after": {g: f"{savings['after'][g]:.1f}%" for g in genres},
        "Validation loss before": {g: f"{losses['before'][f'eval_{g}_loss']:.2f}" for g in genres},
        "Validation loss after": {g: f"{losses['after'][f'eval_{g}_loss']:.2f}" for g in genres},
    }
    header = _HEADER.format(title=title, base_model=base_model)
    return header + "\n" + _table(genres, columns) + "\n"
