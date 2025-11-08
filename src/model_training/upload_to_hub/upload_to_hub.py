"""
================================================================
================================================================
AI-Generated code. It worked just fine for me, but you might
want to double check before using it.
================================================================
================================================================

"""

import torch
import config
from class_model import Roberta4TextEntailment
from utils import push_to_huggingface_hub


def main():
    """Upload the trained model to Hugging Face Hub."""

    print("\n" + "="*60)
    print("HUGGING FACE HUB UPLOAD SCRIPT")
    print("="*60)

    # Configuration check
    if config.HF_REPO_NAME is None:
        print("\n[ERROR] HF_REPO_NAME not configured in config.py")
        print("\nPlease set:")
        print("  config.HF_REPO_NAME = 'your-username/model-name'")
        print("\nExample:")
        print("  HF_REPO_NAME = 'john-doe/xlm-roberta-large-en-ko-nli'")
        return

    if config.HF_TOKEN is None:
        import os
        if os.environ.get('HF_TOKEN') is None:
            print("\n[ERROR] HF_TOKEN not configured")
            print("\nPlease either:")
            print("  1. Set config.HF_TOKEN = 'hf_xxxx' in config.py")
            print("  2. Set HF_TOKEN environment variable")
            print("\nGet your token from:")
            print("  https://huggingface.co/settings/tokens")
            print("  (Make sure it has WRITE permission!)")
            return

    # Model configuration
    MODEL_PATH = 'best_model.bin'  # Path to your saved checkpoint

    print(f"\nConfiguration:")
    print(f"  Repository: {config.HF_REPO_NAME}")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Private: {config.HF_PRIVATE}")

    # Confirm upload
    response = input("\nProceed with upload? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("Upload cancelled.")
        return

    # Load model
    print("\n→ Loading model...")
    model = Roberta4TextEntailment(config.N_CLASSES)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print("[OK] Model loaded successfully")
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("\nMake sure you have trained the model first:")
        print("  python train.py")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Get metrics (optional - you can update these manually)
    # These will be shown in the model card
    print("\n→ Enter model metrics (press Enter to skip):")

    try:
        f1_input = input("  F1-score (e.g., 0.88): ").strip()
        f1_score = float(f1_input) if f1_input else None

        acc_input = input("  Accuracy (e.g., 0.88): ").strip()
        accuracy = float(acc_input) if acc_input else None
    except ValueError:
        print("[WARNING] Invalid metrics, skipping...")
        f1_score = None
        accuracy = None

    # Optional: Add classification report
    classification_report_text = None
    add_report = input("\n  Include classification report? (yes/no): ").lower().strip()
    if add_report in ['yes', 'y']:
        print("\nPaste classification report (press Ctrl+D/Ctrl+Z+Enter when done):")
        try:
            import sys
            classification_report_text = sys.stdin.read()
        except:
            print("[WARNING] Could not read report, skipping...")

    # Upload to Hub
    print("\n" + "="*60)
    print("STARTING UPLOAD")
    print("="*60)

    success = push_to_huggingface_hub(
        model=model,
        tokenizer=config.TOKENIZER,
        model_path=MODEL_PATH,
        repo_name=config.HF_REPO_NAME,
        token=config.HF_TOKEN,
        private=config.HF_PRIVATE,
        f1_score=f1_score,
        accuracy=accuracy,
        classification_report=classification_report_text
    )

    if success:
        print("\n✓ Upload completed successfully!")
        print(f"\nView your model at:")
        print(f"  https://huggingface.co/{config.HF_REPO_NAME}")
    else:
        print("\n✗ Upload failed. Check the error messages above.")


if __name__ == "__main__":
    main()
