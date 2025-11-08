"""
Re-upload models to HuggingFace Hub with updated model cards

This script:
1. Downloads models from HuggingFace Hub
2. Generates comprehensive model cards with YAML metadata
3. Re-uploads models with updated documentation

Usage:
    python re_upload.py
    python re_upload.py --model baseline  # Upload only baseline
    python re_upload.py --model teacher   # Upload only teacher
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, create_repo

import config
from utils import generate_causal_lm_model_card


def re_upload_model(model_config: dict, token: str, dry_run: bool = False):
    """
    Download a model from HuggingFace, update its model card, and re-upload.

    Args:
        model_config: Dictionary containing model configuration
        token: HuggingFace API token
        dry_run: If True, only generate and display model card without uploading

    Returns:
        bool: True if successful, False otherwise
    """
    repo_name = model_config['repo_name']
    display_name = model_config['display_name']

    print("\n" + "="*70)
    print(f"RE-UPLOADING: {display_name}")
    print("="*70)
    print(f"Repository: {repo_name}")
    print()

    try:
        # Step 1: Download model from HuggingFace Hub
        print("Step 1: Downloading model from HuggingFace Hub...")
        print("-" * 70)

        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        model = AutoModelForCausalLM.from_pretrained(repo_name)

        print(f"✓ Model downloaded: {repo_name}")
        print()

        # Step 2: Generate updated model card
        print("Step 2: Generating updated model card...")
        print("-" * 70)

        model_card = generate_causal_lm_model_card(model_config)

        print("✓ Model card generated with comprehensive YAML metadata")
        print()

        # Display model card preview
        print("Model Card Preview (first 15 lines):")
        print("-" * 70)
        for i, line in enumerate(model_card.split('\n')[:15], 1):
            print(f"{i:2d} | {line}")
        print("...")
        print()

        if dry_run:
            print("[DRY RUN MODE] Skipping upload")
            print()
            print("Full model card saved to: model_card_preview.md")
            with open("model_card_preview.md", "w", encoding="utf-8") as f:
                f.write(model_card)
            return True

        # Step 3: Create/verify repository
        print("Step 3: Creating/verifying repository...")
        print("-" * 70)

        api = HfApi()
        create_repo(
            repo_id=repo_name,
            token=token,
            private=config.PRIVATE,
            exist_ok=True,
            repo_type="model"
        )

        print(f"✓ Repository ready: {repo_name}")
        print()

        # Step 4: Upload model, tokenizer, and model card
        print("Step 4: Uploading to HuggingFace Hub...")
        print("-" * 70)
        print("This may take several minutes for larger models...")
        print()

        # Upload model
        print("→ Uploading model...")
        model.push_to_hub(
            repo_id=repo_name,
            token=token,
            commit_message=f"Re-upload {display_name} with updated model card"
        )
        print("✓ Model uploaded")

        # Upload tokenizer
        print("→ Uploading tokenizer...")
        tokenizer.push_to_hub(
            repo_id=repo_name,
            token=token,
            commit_message="Re-upload tokenizer"
        )
        print("✓ Tokenizer uploaded")

        # Upload model card
        print("→ Uploading model card...")
        temp_card_path = f"temp_card_{model_config['base_model']}.md"

        with open(temp_card_path, "w", encoding="utf-8") as f:
            f.write(model_card)

        api.upload_file(
            path_or_fileobj=temp_card_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
            token=token,
            commit_message="Update model card with comprehensive documentation"
        )

        # Cleanup
        os.remove(temp_card_path)
        print("✓ Model card uploaded")
        print()

        # Success message
        print("="*70)
        print("✓ RE-UPLOAD COMPLETE!")
        print("="*70)
        print(f"View your updated model at:")
        print(f"  https://huggingface.co/{repo_name}")
        print()
        print("Features enabled:")
        print("  ✓ Interactive widget for testing")
        print("  ✓ Searchable by tags")
        print("  ✓ Metrics displayed on model page")
        print("  ✓ YAML metadata for discoverability")
        print("  ✓ Comprehensive usage examples")
        print("="*70)
        print()

        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to re-upload model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your token has write permissions")
        print("2. Verify repository exists and you have access")
        print("3. Ensure stable internet connection")
        print("4. Try: pip install --upgrade transformers huggingface_hub")
        return False


def main():
    """Main function to orchestrate model re-uploads"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Re-upload models to HuggingFace Hub with updated model cards"
    )
    parser.add_argument(
        '--model',
        choices=['baseline', 'teacher', 'both'],
        default='both',
        help='Which model to upload (default: both)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate model cards without uploading'
    )

    args = parser.parse_args()

    # Header
    print("\n" + "="*70)
    print("PENDO TEXT EDITOR - MODEL RE-UPLOAD SYSTEM")
    print("="*70)
    print()
    print("This script will:")
    print("  1. Download models from your HuggingFace account")
    print("  2. Generate updated model cards with YAML metadata")
    print("  3. Re-upload models with comprehensive documentation")
    print()

    if args.dry_run:
        print("[DRY RUN MODE] - Model cards will be generated but not uploaded")
        print()

    # Get authentication token
    try:
        token = config.get_token()
        print("✓ Authentication token validated")
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return

    # Determine which models to upload
    models_to_upload = []

    if args.model in ['baseline', 'both']:
        models_to_upload.append(config.BASELINE_CONFIG)

    if args.model in ['teacher', 'both']:
        models_to_upload.append(config.TEACHER_CONFIG)

    print(f"\nModels to upload: {len(models_to_upload)}")
    for model_cfg in models_to_upload:
        print(f"  - {model_cfg['display_name']}")
    print()

    # Confirm before proceeding (skip in dry-run mode)
    if not args.dry_run:
        response = input("Proceed with re-upload? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Aborted by user.")
            return

    # Upload each model
    success_count = 0
    for model_cfg in models_to_upload:
        if re_upload_model(model_cfg, token, dry_run=args.dry_run):
            success_count += 1

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Successfully processed: {success_count}/{len(models_to_upload)} models")

    if success_count == len(models_to_upload):
        print("\n🎉 All models re-uploaded successfully!")
    elif success_count > 0:
        print(f"\n⚠️  {len(models_to_upload) - success_count} model(s) failed")
    else:
        print("\n❌ All uploads failed")

    print("="*70)
    print()


if __name__ == "__main__":
    main()
