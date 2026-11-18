"""Tiny stand-ins for the tokenizer and language model, for fast tests."""

import math

import torch

from backend.prediction.settings import PredictionSettings

PIECES = [" the", " there", " th", "ere", " ", ".", "e", "<eos>", " cat"]
EOS_ID = PIECES.index("<eos>")
_IMPOSSIBLE = -1e9


class FakeTokenizer:
    eos_token_id = EOS_ID

    def __len__(self):
        return len(PIECES)

    def batch_decode(self, batches):
        return [PIECES[ids[0]] for ids in batches]


class FakeLanguageModel:
    """Returns scripted next-token probabilities keyed by generated pieces."""

    device = torch.device("cpu")
    tokenizer = FakeTokenizer()

    def __init__(self, script):
        self._script = script

    def encode(self, text):
        return (EOS_ID,)

    def begin(self, context_ids):
        return FakeSession(self._row)

    def _row(self, tokens):
        generated = tuple(PIECES[token] for token in tokens)
        probabilities = self._script.get(generated, {".": 1.0})
        row = torch.full((len(PIECES),), _IMPOSSIBLE)
        for piece, probability in probabilities.items():
            row[PIECES.index(piece)] = math.log(probability)
        return row


class FakeSession:
    """Tracks each hypothesis's tokens the way the real cache does."""

    def __init__(self, row_for):
        self._row_for = row_for
        self._sequences = [()]

    def first(self):
        return torch.stack([self._row_for(())])

    def advance(self, parents, token_ids):
        self._sequences = [
            self._sequences[parent] + (token,) for parent, token in zip(parents, token_ids)
        ]
        return torch.stack([self._row_for(tokens) for tokens in self._sequences])


class FakePredictor:
    """Predictor that answers from a {(prompt, prefix): suggestions} table."""

    def __init__(self, answers=None, settings=None):
        self.settings = settings or PredictionSettings(debounce_ms=0)
        self._answers = answers or {}
        self.calls = []

    def predict(self, prompt, prefix=""):
        self.calls.append((prompt, prefix))
        return self._answers.get((prompt, prefix), [])
