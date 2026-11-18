"""Scores beam-search steps incrementally, reusing work on the context."""

import copy

import torch


class DecodingSession:
    """One word search over a context that has already been processed.

    The context's attention cache is shared and never modified; the session
    works on its own copy, so each step only runs the newly added tokens
    through the model instead of the whole text again.
    """

    def __init__(self, forward, context_cache, first_log_probs):
        self._forward = forward
        self._context_cache = context_cache
        self._first_log_probs = first_log_probs
        self._cache = None

    def first(self):
        """Next-token log-probabilities right after the context: [1, vocab]."""
        return self._first_log_probs

    def advance(self, parents, token_ids):
        """Extend hypotheses by one token each.

        Args:
            parents: For each new hypothesis, its index in the previous step.
            token_ids: The token each hypothesis adds.

        Returns:
            Tensor [len(token_ids), vocab] of next-token log-probabilities.
        """
        device = self._first_log_probs.device
        if self._cache is None:
            self._cache = copy.deepcopy(self._context_cache)
            self._cache.batch_repeat_interleave(len(token_ids))
        else:
            self._cache.batch_select_indices(torch.tensor(parents, device=device))
        tokens = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(1)
        log_probs, self._cache = self._forward(tokens, self._cache)
        return log_probs
