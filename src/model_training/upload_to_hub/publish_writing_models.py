"""Uploads the fine-tuned writing models to the logged-in Hugging Face account.

Run from the `src` directory (after `hf auth login`):
    python -m model_training.upload_to_hub.publish_writing_models
Repositories are created private unless `--public` is given.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from .model_card import build_model_card

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


@dataclass(frozen=True)
class TrainedModel:
    run_name: str
    repo_name: str
    title: str
    base_model: str


MODELS = (
    TrainedModel(
        "smollm2-135m-writing", "pendo-smollm2-135m-writing",
        "Pendo SmolLM2-135M (writing)", "HuggingFaceTB/SmolLM2-135M",
    ),
    TrainedModel(
        "distilgpt2-writing", "pendo-distilgpt2-writing",
        "Pendo DistilGPT-2 (writing)", "bekalebendong/pendo-distilgpt2-wikitext",
    ),
)


def publish(api, model, owner, private):
    """Create the repository if needed and upload weights, tokenizer and card."""
    run_dir = OUTPUT_DIR / model.run_name
    repo_id = f"{owner}/{model.repo_name}"
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, folder_path=str(run_dir / "final"),
        ignore_patterns=["training_args.bin"],
    )
    card = build_model_card(run_dir, model.title, model.base_model)
    api.upload_file(
        repo_id=repo_id, path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
    )
    return repo_id


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--public", action="store_true", help="make the repositories public")
    args = parser.parse_args()
    api = HfApi()
    owner = api.whoami()["name"]
    for model in MODELS:
        repo_id = publish(api, model, owner, private=not args.public)
        print(f"Uploaded https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
