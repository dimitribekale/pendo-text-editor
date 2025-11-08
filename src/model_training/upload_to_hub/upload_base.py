import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'output', 'final_model')
REPO_NAME = "bekalebendong/pendo-distilgpt2-wikitext"
HF_TOKEN = None
PRIVATE = False


def upload_model():
    print("="*60)
    print("HUGGING FACE HUB UPLOAD - BASELINE MODEL")
    print("="*60)
    print()

    # Step 1: Token validation
    print("Step 1: Validating HuggingFace token...")
    print("-" * 60)

    token = HF_TOKEN
    if token is None:
        token = os.environ.get('HF_TOKEN')

    if token is None:
        print("\n[ERROR] HuggingFace token not found!")
        print("\nPlease either:")
        print("  1. Set HF_TOKEN variable in this script")
        print("  2. Set HF_TOKEN environment variable")
        print("\nGet your token from:")
        print("  https://huggingface.co/settings/tokens")
        print("  (Make sure it has WRITE permission!)")
        return False

    print("✓ Token found")
    print()

    # Step 2: Create repository
    print("Step 2: Creating repository...")
    print("-" * 60)
    try:
        create_repo(
            repo_id=REPO_NAME,
            token=token,
            private=PRIVATE,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository ready: {REPO_NAME}")
    except Exception as e:
        print(f"[ERROR] Failed to create repository: {e}")
        return False
    print()

    model_card = f"""---
  language: en
  license: mit
  tags:
  - text-generation
  - distilgpt2
  - wikitext
  datasets:
  - wikitext
  metrics:
  - perplexity
  model-index:
  - name: {REPO_NAME}
    results:
    - task:
        type: text-generation
      dataset:
        name: WikiText-103
        type: wikitext
      metrics:
      - type: loss
        value: 3.206
        name: Validation Loss
  ---

  # Pendo DistilGPT2 - Fine-tuned on WikiText-103

  ## Model Description

  This is a DistilGPT2 model fine-tuned on the full WikiText-103 dataset for text prediction in the Pendo Text Editor.

  **Training Details:**
  - Base Model: `distilgpt2` (82M parameters)
  - Dataset: WikiText-103 (full training set)
  - Training Hardware: 2x NVIDIA H100 80GB
  - Training Time: ~42 minutes
  - Block Size: 512 tokens
  - Batch Size: 128 (effective)
  - Mixed Precision: bf16
  - Optimizer: AdamW with cosine learning rate schedule

  **Performance:**
  - Final Training Loss: 3.379
  - Final Validation Loss: 3.206
  - Model shows good generalization (no overfitting)

  **Training Configuration:**
  - Learning Rate: 5e-5 with warmup
  - Epochs: 3
  - Gradient Accumulation Steps: 2
  - LR Scheduler: Cosine with 500 warmup steps

  ## Intended Use

  This model is designed for real-time text prediction in the Pendo text editor, providing contextual word suggestions
  as users type.

  ## Usage

  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM

  tokenizer = AutoTokenizer.from_pretrained("{REPO_NAME}")
  model = AutoModelForCausalLM.from_pretrained("{REPO_NAME}")

  prompt = "The history of"
  inputs = tokenizer(prompt, return_tensors="pt")
  outputs = model.generate(**inputs, max_new_tokens=20)
  print(tokenizer.decode(outputs[0]))

  Limitations

  - Trained only on Wikipedia-style text
  - May not perform well on code or informal text
  - Optimized for inference on consumer hardware (M2, RTX 4060Ti)

  Training Details

  Trained using optimized hyperparameters:
  - Multi-GPU training with Distributed Data Parallel
  - Proper train/validation split (90/10)
  - Gradient clipping for stability
  - TensorBoard logging for monitoring

  Citation

  @misc{{pendo-distilgpt2-wikitext,
    author = {{Your Name}},
    title = {{Pendo DistilGPT2 Fine-tuned on WikiText-103}},
    year = {{2025}},
    publisher = {{HuggingFace}},
    howpublished = {{\\url{{https://huggingface.co/{REPO_NAME}}}}}
  }}
  """
    print("Step 3: Loading model...")
    print("-" * 60)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        print("✓ Model loaded!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False
    print()

    # Step 4: Push to hub
    print("Step 4: Uploading to HuggingFace Hub...")
    print("-" * 60)
    print(f"Repository: {REPO_NAME}")
    print()

    try:
        # Push model with commit message
        model.push_to_hub(
            repo_id=REPO_NAME,
            token=token,
            commit_message="Upload baseline model (Val Loss: 3.206)"
        )
        print("✓ Model uploaded!")

        # Push tokenizer
        tokenizer.push_to_hub(
            repo_id=REPO_NAME,
            token=token,
            commit_message="Upload tokenizer"
        )
        print("✓ Tokenizer uploaded!")

        # Push model card
        api = HfApi()
        temp_card_path = "temp_model_card.md"

        with open(temp_card_path, "w", encoding="utf-8") as f:
            f.write(model_card)

        api.upload_file(
            path_or_fileobj=temp_card_path,
            path_in_repo="README.md",
            repo_id=REPO_NAME,
            repo_type="model",
            token=token,
            commit_message="Add model card with YAML metadata"
        )
        print("✓ Model card uploaded!")

        # Cleanup
        os.remove(temp_card_path)

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return False

    print()
    print("="*60)
    print("✓ UPLOAD COMPLETE!")
    print("="*60)
    print(f"Your model is now available at:")
    print(f"https://huggingface.co/{REPO_NAME}")
    print()
    print("Others can now use your model with:")
    print(f'model = AutoModelForCausalLM.from_pretrained("{REPO_NAME}")')
    print("="*60)

    return True

if __name__ == "__main__":
    upload_model()