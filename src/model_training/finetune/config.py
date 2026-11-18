"""Fine-tuning settings and their command line overrides."""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

MODEL_TRAINING_DIR = Path(__file__).resolve().parents[1]
CORPUS_DIR = MODEL_TRAINING_DIR / "data" / "writing"
OUTPUT_DIR = MODEL_TRAINING_DIR / "output"


@dataclass(frozen=True)
class FinetuneConfig:
    """One training run. Defaults suit a single large GPU."""

    model: str = "HuggingFaceTB/SmolLM2-135M"
    run_name: str = "smollm2-135m-writing"
    block_size: int = 512        # tokens per training example
    min_tokens: int = 32         # shorter leftovers are dropped
    batch_size: int = 128
    learning_rate: float = 5e-5
    epochs: float = 3.0
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    eval_steps: int = 100
    email_oversample: int = 3    # emails are the smallest genre
    workers: int = 16            # tokenization and data loading processes
    train_limit: Optional[int] = None  # documents; for quick trial runs
    max_steps: int = -1          # stop early; for quick trial runs
    seed: int = 42

    @property
    def output_dir(self):
        return OUTPUT_DIR / self.run_name


def parse_config():
    """Build a config from `--field value` command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a model on the writing corpus")
    defaults = FinetuneConfig()
    for field in fields(FinetuneConfig):
        default = getattr(defaults, field.name)
        kind = int if field.name in ("train_limit",) else type(default)
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=kind, default=default)
    return FinetuneConfig(**vars(parser.parse_args()))
