"""The public datasets that make up the writing corpus, one per genre."""

import re
from dataclasses import dataclass
from typing import Optional

from .text_cleaning import (
    detokenize,
    remove_blog_markers,
    strip_news_dateline,
    unwrap_lines,
)


@dataclass(frozen=True)
class WritingSource:
    """Where one genre comes from and how to read it.

    `stride` keeps one document in every N so the sample spans the whole
    file instead of only its beginning. Documents matching `reject` are
    skipped: they carry artifacts that would teach the model bad habits.
    """

    genre: str
    repo: str
    filename: str
    builder: str
    column: str
    cleaners: tuple = ()
    stride: int = 1
    reject: Optional[re.Pattern] = None


_MAIL_ENCODING_DEBRIS = re.compile(r"=[0-9A-F]{2}|=\s*$", re.MULTILINE)
_CHAT_SPELLING = re.compile(r"\b(?:im|dont|cant|didnt|ive|lol|ur)\b|\.{4,}")
_LATEX = re.compile(r"\\[A-Za-z]|(?:\$[^$]*){4}")


SOURCES = (
    WritingSource(
        "emails", "Yale-LILY/aeslc", "data/train-00000-of-00001.parquet",
        "parquet", "email_body", cleaners=(unwrap_lines,), reject=_MAIL_ENCODING_DEBRIS,
    ),
    WritingSource(
        "blogs", "tasksource/blog_authorship_corpus", "blogtext.csv",
        "csv", "text", cleaners=(remove_blog_markers,), stride=5, reject=_CHAT_SPELLING,
    ),
    WritingSource(
        "fiction", "euclaise/writingprompts",
        "data/train-00000-of-00002-105e07cb0d199464.parquet",
        "parquet", "story", cleaners=(detokenize,), stride=6,
    ),
    WritingSource(
        "academic", "gfissore/arxiv-abstracts-2021", "arxiv-abstracts.jsonl.gz",
        "json", "abstract", cleaners=(unwrap_lines,), stride=12, reject=_LATEX,
    ),
    WritingSource(
        "news", "abisee/cnn_dailymail", "1.0.0/validation-00000-of-00001.parquet",
        "parquet", "article", cleaners=(strip_news_dateline,),
    ),
    # Shard 37 is outside both the original training slice (first 60%) and
    # the evaluation text (shard 40).
    WritingSource(
        "encyclopedia", "wikimedia/wikipedia",
        "20231101.en/train-00037-of-00041.parquet",
        "parquet", "text", stride=12,
    ),
)
