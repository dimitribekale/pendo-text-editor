"""Training examples are cut the way the predictor will see text."""

from model_training.finetune.chunking import chunk_documents


class WordTokenizer:
    """One token per word; ids are the word lengths."""

    bos_token_id = None
    eos_token_id = 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[len(word) for word in text.split()] for text in texts]}


def _chunk(texts, genres, block_size=4, min_tokens=2):
    batch = {"text": texts, "genre": genres}
    return chunk_documents(batch, WordTokenizer(), block_size, min_tokens)


def test_every_chunk_starts_with_the_start_token_and_fits_the_block():
    chunks = _chunk(["a bb ccc dddd eeeee ffffff"], ["blogs"])
    assert chunks["input_ids"] == [[0, 1, 2, 3], [0, 4, 5, 6]]


def test_chunks_never_mix_documents():
    chunks = _chunk(["a bb", "ccc dddd"], ["emails", "news"])
    assert chunks["input_ids"] == [[0, 1, 2], [0, 3, 4]]
    assert chunks["genre"] == ["emails", "news"]


def test_short_leftovers_are_dropped():
    chunks = _chunk(["a bb ccc dddd"], ["blogs"])
    assert chunks["input_ids"] == [[0, 1, 2, 3]]
