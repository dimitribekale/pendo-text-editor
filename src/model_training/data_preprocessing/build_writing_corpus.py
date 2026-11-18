"""Builds the multi-genre writing corpus used to fine-tune the predictor.

Run from the `src` directory:
    python -m model_training.data_preprocessing.build_writing_corpus

Writes train/val/test JSON Lines files (one document per line, split by
document so no text is shared between splits) and one plain-text test file
per genre for the typing simulator.
"""

import json
import random
from pathlib import Path

from .source_reader import read_documents
from .writing_sources import SOURCES

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "writing"
_CHARS_PER_GENRE = 40_000_000
_HELD_OUT_DOCUMENTS = 300  # per genre, for each of val and test
_SIMULATOR_CHARS = 20_000
_SEED = 42


def _collect(source):
    """Read documents until the genre's character budget is filled."""
    documents, total = [], 0
    for text in read_documents(source):
        documents.append(text)
        total += len(text)
        if total >= _CHARS_PER_GENRE:
            break
    random.Random(_SEED).shuffle(documents)
    return documents


def _split(documents):
    held_out = min(_HELD_OUT_DOCUMENTS, len(documents) // 10)
    return {
        "test": documents[:held_out],
        "val": documents[held_out: 2 * held_out],
        "train": documents[2 * held_out:],
    }


def _append(split, genre, documents):
    with (CORPUS_DIR / f"{split}.jsonl").open("a", encoding="utf-8") as file:
        for text in documents:
            file.write(json.dumps({"genre": genre, "text": text}, ensure_ascii=False) + "\n")


def _write_simulator_text(genre, documents):
    text = "\n\n".join(documents)[:_SIMULATOR_CHARS]
    (CORPUS_DIR / "simulator").mkdir(exist_ok=True)
    (CORPUS_DIR / "simulator" / f"{genre}.txt").write_text(text, encoding="utf-8")


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CORPUS_DIR.glob("*.jsonl"):
        stale.unlink()
    for source in SOURCES:
        splits = _split(_collect(source))
        for split, documents in splits.items():
            _append(split, source.genre, documents)
        _write_simulator_text(source.genre, splits["test"])
        sizes = {name: f"{sum(map(len, docs)) / 1e6:.1f}M chars" for name, docs in splits.items()}
        print(f"{source.genre}: {len(splits['train']):,} training documents, {sizes}", flush=True)


if __name__ == "__main__":
    main()
