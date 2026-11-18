"""Loads the writing corpus as tokenized training and validation sets."""

from functools import partial

from datasets import concatenate_datasets, load_dataset

from .chunking import chunk_documents
from .config import CORPUS_DIR

_OVERSAMPLED_GENRE = "emails"


def _load_chunks(split, tokenizer, config, limit=None):
    documents = load_dataset("json", data_files=str(CORPUS_DIR / f"{split}.jsonl"), split="train")
    if limit:
        documents = documents.shuffle(seed=config.seed).select(range(min(limit, len(documents))))
    chunker = partial(
        chunk_documents, tokenizer=tokenizer,
        block_size=config.block_size, min_tokens=config.min_tokens,
    )
    return documents.map(
        chunker, batched=True, remove_columns=documents.column_names,
        num_proc=None if limit else config.workers, desc=f"Tokenizing {split}",
    )


def _of_genre(chunks, genre):
    return chunks.filter(lambda example: example["genre"] == genre)


def load_training_set(tokenizer, config):
    """Shuffled training examples, with the smallest genre repeated."""
    chunks = _load_chunks("train", tokenizer, config, config.train_limit)
    extra = [_of_genre(chunks, _OVERSAMPLED_GENRE)] * (config.email_oversample - 1)
    combined = concatenate_datasets([chunks] + extra)
    return combined.remove_columns("genre").shuffle(seed=config.seed)


def load_validation_sets(tokenizer, config):
    """Validation examples as {"all": ..., "<genre>": ...}.

    Scoring each genre separately shows whether a gain on one kind of
    writing is being paid for by a loss on another.
    """
    chunks = _load_chunks("val", tokenizer, config, limit=200 if config.train_limit else None)
    sets = {"all": chunks}
    for genre in sorted(set(chunks["genre"])):
        sets[genre] = _of_genre(chunks, genre)
    return {name: subset.remove_columns("genre") for name, subset in sets.items()}
