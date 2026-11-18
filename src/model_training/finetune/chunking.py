"""Cuts tokenized documents into fixed-size training examples."""


def start_token_id(tokenizer):
    """The token the predictor puts in front of every context."""
    return tokenizer.bos_token_id or tokenizer.eos_token_id


def chunk_documents(batch, tokenizer, block_size, min_tokens):
    """Turn a batch of documents into examples of at most `block_size` tokens.

    Chunks never span two documents, and each begins with the start token
    followed by text from anywhere in a document. That mirrors inference,
    where the model sees a start token and then a window of recent text.

    Args:
        batch: {"genre": [...], "text": [...]} as given by `Dataset.map`.

    Returns:
        {"genre": [...], "input_ids": [...]} with one entry per chunk.
    """
    start = start_token_id(tokenizer)
    body_size = block_size - 1
    encoded = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
    chunks = {"genre": [], "input_ids": []}
    for genre, token_ids in zip(batch["genre"], encoded):
        for offset in range(0, len(token_ids), body_size):
            body = token_ids[offset: offset + body_size]
            if len(body) >= min_tokens:
                chunks["genre"].append(genre)
                chunks["input_ids"].append([start] + body)
    return chunks
