"""Indexes the tokenizer vocabulary by the text each token produces."""

from bisect import bisect_left
from collections import defaultdict

import torch

_WORD_EXTRAS = "'’"


def is_word_text(text):
    """True when every character can be part of a word (empty counts)."""
    return all(char.isalnum() or char in _WORD_EXTRAS for char in text)


class TokenVocabulary:
    """Answers "which tokens can come next?" for word-level constraints."""

    def __init__(self, tokenizer, device):
        self._device = device
        ids = range(len(tokenizer))
        self._pieces = tokenizer.batch_decode([[i] for i in ids])
        self.word_ids = self._ids(
            i for i, piece in enumerate(self._pieces) if piece and is_word_text(piece)
        )
        self.boundary_ids = self._ids(
            i for i, piece in enumerate(self._pieces) if self._ends_word(i, piece, tokenizer)
        )
        self._match_cache = {}
        self._sorted = sorted((piece, i) for i, piece in enumerate(self._pieces) if piece)
        self._ids_by_piece = defaultdict(list)
        for piece, token_id in self._sorted:
            self._ids_by_piece[piece].append(token_id)

    def piece(self, token_id):
        """Text produced by a single token."""
        return self._pieces[token_id]

    def matching(self, remaining):
        """Tokens compatible with the not-yet-generated part of the target.

        Returns:
            (partial_ids, cover_ids): tokens that spell only the start of
            `remaining`, and tokens that spell all of it and may continue
            the word.
        """
        if remaining not in self._match_cache:
            self._match_cache[remaining] = self._find_matches(remaining)
        return self._match_cache[remaining]

    def _find_matches(self, remaining):
        partial = [
            token_id
            for length in range(1, len(remaining))
            for token_id in self._ids_by_piece.get(remaining[:length], ())
        ]
        cover = []
        # Pieces sharing a prefix are adjacent in the sorted vocabulary.
        for position in range(bisect_left(self._sorted, (remaining,)), len(self._sorted)):
            piece, token_id = self._sorted[position]
            if not piece.startswith(remaining):
                break
            if is_word_text(piece[len(remaining):]):
                cover.append(token_id)
        return self._ids(partial), self._ids(cover)

    def _ends_word(self, token_id, piece, tokenizer):
        if token_id == tokenizer.eos_token_id:
            return True
        return bool(piece) and not is_word_text(piece[0]) and "�" not in piece

    def _ids(self, token_ids):
        return torch.tensor(list(token_ids), dtype=torch.long, device=self._device)
