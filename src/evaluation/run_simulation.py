"""Command line entry point: measure a model on a text file.

Run from the `src` directory:
    python -m evaluation.run_simulation evaluation/data/general.txt
"""

import argparse
from dataclasses import replace
from pathlib import Path

from backend.app_config import load_config
from backend.prediction.settings import PredictionSettings
from backend.prediction.word_predictor import WordPredictor

from .report import format_report
from .typing_simulator import TypingSimulator


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("text_file", type=Path, help="UTF-8 text to replay")
    parser.add_argument("--model", help="model path or hub id (default: config.yaml)")
    parser.add_argument("--min-confidence", type=float, help="display threshold")
    parser.add_argument("--max-chars", type=int, default=6000, help="text length to replay")
    return parser.parse_args()


def _settings_from(args):
    settings = PredictionSettings.from_config(load_config())
    if args.model:
        settings = replace(settings, model_path=args.model)
    if args.min_confidence is not None:
        settings = replace(settings, min_confidence=args.min_confidence)
    return settings


def main():
    args = _parse_args()
    settings = _settings_from(args)
    text = args.text_file.read_text(encoding="utf-8")[: args.max_chars]
    predictor = WordPredictor.from_settings(settings)
    result = TypingSimulator(predictor, settings.min_confidence).run(text)
    title = f"{settings.model_path} on {args.text_file.name} (threshold {settings.min_confidence})"
    print(format_report(title, result))


if __name__ == "__main__":
    main()
