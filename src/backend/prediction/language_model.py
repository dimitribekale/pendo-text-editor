"""Loads the causal language model and scores next tokens."""

import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .decoding_session import DecodingSession

_MAX_CONTEXT_TOKENS = 256


def _pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LanguageModel:
    """Thin wrapper exposing only what word prediction needs."""

    def __init__(self, model_path, device="auto"):
        self.device = _pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        self._model.to(self.device).eval()
        self._start_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        self._lock = threading.Lock()
        self._last_context = None  # (context_ids, cache, log_probs)

    def encode(self, text):
        """Token ids for the context, keeping only the most recent tokens."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return tuple([self._start_id] + ids[-_MAX_CONTEXT_TOKENS:])

    def begin(self, context_ids):
        """Start a word search after `context_ids`.

        The context is processed once and remembered, so every keystroke
        inside the same word reuses it instead of re-reading the text.
        """
        if self._last_context is None or self._last_context[0] != context_ids:
            tokens = torch.tensor([context_ids], dtype=torch.long, device=self.device)
            log_probs, cache = self._forward(tokens, None)
            self._last_context = (context_ids, cache, log_probs)
        _, cache, log_probs = self._last_context
        return DecodingSession(self._forward, cache, log_probs)

    def _forward(self, tokens, cache):
        """Run new tokens through the model; return (log_probs, updated cache)."""
        with self._lock, torch.inference_mode():
            output = self._model(input_ids=tokens, past_key_values=cache, use_cache=True)
            log_probs = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)
        return log_probs, output.past_key_values
