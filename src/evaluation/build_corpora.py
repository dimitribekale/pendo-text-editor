"""Builds the held-out evaluation texts under `evaluation/data/`.

Run from the `src` directory:
    python -m evaluation.build_corpora

Neither source overlaps the training data: training used the first 60% of
the Wikipedia dump, and never used Dolly.
"""

import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).parent / "data"
_SEED = 42
_SAMPLES = 60
_MIN_CHARS = 300
_MAX_CHARS = 1200


def _wikipedia_texts():
    """Article openings from the last shard of the dump (held out)."""
    shard = hf_hub_download(
        repo_id="wikimedia/wikipedia",
        filename="20231101.en/train-00040-of-00041.parquet",
        repo_type="dataset",
    )
    return load_dataset("parquet", data_files=shard, split="train")["text"]


def _general_texts():
    """Human-written answers: closer to everyday writing than Wikipedia."""
    return load_dataset("databricks/databricks-dolly-15k", split="train")["response"]


def _sample(texts):
    usable = [text.strip() for text in texts if len(text.strip()) >= _MIN_CHARS]
    chosen = random.Random(_SEED).sample(usable, _SAMPLES)
    return "\n\n".join(text[:_MAX_CHARS] for text in chosen)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    sources = {"wiki.txt": _wikipedia_texts, "general.txt": _general_texts}
    for filename, load_texts in sources.items():
        corpus = _sample(load_texts())
        (DATA_DIR / filename).write_text(corpus, encoding="utf-8")
        print(f"Wrote {len(corpus):,} characters to {DATA_DIR / filename}")


if __name__ == "__main__":
    main()
