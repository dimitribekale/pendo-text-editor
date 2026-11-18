"""Streams cleaned documents out of one writing source."""

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .text_cleaning import is_usable, normalize_whitespace, truncate_at_paragraph


def _clean(text, source):
    for cleaner in source.cleaners:
        text = cleaner(text)
    return truncate_at_paragraph(normalize_whitespace(text))


def read_documents(source):
    """Yield usable, cleaned documents from `source`, one in every `stride`.

    Documents matching the source's `reject` pattern are left out.

    The file is downloaded once into the Hugging Face cache and then read
    as a stream, so even the multi-gigabyte sources need little memory.
    """
    path = hf_hub_download(source.repo, source.filename, repo_type="dataset")
    rows = load_dataset(source.builder, data_files=path, split="train", streaming=True)
    for position, row in enumerate(rows):
        if position % source.stride:
            continue
        raw = row[source.column] or ""
        if source.reject and source.reject.search(raw):
            continue
        text = _clean(raw, source)
        if is_usable(text):
            yield text
