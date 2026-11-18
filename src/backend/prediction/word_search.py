"""Beam search for whole words that start with what the user typed."""

import math
from collections import defaultdict

import torch

from .hypothesis import Hypothesis

_MIN_PROBABILITY = 1e-4  # drop words too unlikely to ever be shown


class WordSearch:
    """Finds the most probable words consistent with a target prefix.

    The model works in sub-word tokens, so a word may take several steps.
    Each step keeps the best `beam_width` partial words; a word is finished
    when the model would next emit a space, punctuation or end of text.
    """

    def __init__(self, model, vocabulary, beam_width, max_word_tokens):
        self._model = model
        self._vocabulary = vocabulary
        self._beam_width = beam_width
        self._max_word_tokens = max_word_tokens

    def search(self, context_ids, target):
        """Return {generated_text: probability given the target prefix}."""
        beams = [Hypothesis()]
        finished = defaultdict(float)
        # Probability that the text continues with `target` at all; used to
        # condition word probabilities on what the user has already typed.
        target_mass = 0.0 if target else 1.0

        session = self._model.begin(context_ids)
        rows = session.first()
        for step in range(self._max_word_tokens):
            candidates = []
            for position, (beam, row) in enumerate(zip(beams, rows)):
                if beam.covers(target):
                    self._record_finished(beam, row, finished)
                    allowed = self._vocabulary.word_ids
                else:
                    partial, cover = self._vocabulary.matching(target[len(beam.text):])
                    target_mass += self._mass(beam, row, cover)
                    allowed = torch.cat([partial, cover])
                candidates.extend(self._expand(beam, row, allowed, position))
            beams = sorted(candidates, key=lambda h: h.log_prob, reverse=True)
            beams = beams[: self._beam_width]
            out_of_steps = step == self._max_word_tokens - 1
            if not beams or out_of_steps or self._is_decided(beams, finished):
                break
            rows = session.advance(
                [beam.parent for beam in beams], [beam.token_ids[-1] for beam in beams]
            )

        if target_mass <= 0.0:
            return {}
        words = {text: min(1.0, prob / target_mass) for text, prob in finished.items()}
        return {text: prob for text, prob in words.items() if prob >= _MIN_PROBABILITY}

    @staticmethod
    def _is_decided(beams, finished):
        """True when the unfinished words can no longer change the winner."""
        unfinished = sum(math.exp(beam.log_prob) for beam in beams)
        ranked = sorted(finished.values(), reverse=True) + [0.0, 0.0]
        return ranked[0] > ranked[1] + unfinished

    def _record_finished(self, beam, row, finished):
        if beam.text.strip():
            finished[beam.text] += self._mass(beam, row, self._vocabulary.boundary_ids)

    def _expand(self, beam, row, allowed, position):
        if allowed.numel() == 0:
            return []
        count = min(self._beam_width, allowed.numel())
        scores, positions = torch.topk(row[allowed], count)
        token_ids = allowed[positions].tolist()
        return [
            beam.extended(token_id, self._vocabulary.piece(token_id), score, position)
            for token_id, score in zip(token_ids, scores.tolist())
        ]

    @staticmethod
    def _mass(beam, row, token_ids):
        """Joint probability of reaching `beam` and then any of `token_ids`."""
        if token_ids.numel() == 0:
            return 0.0
        return math.exp(beam.log_prob + torch.logsumexp(row[token_ids], dim=0).item())
