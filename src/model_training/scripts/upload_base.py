import os
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'output', 'final_model')
REPO_NAME = "bekalebendong/pendo-distilgpt2-wikitext"

def upload_model():
    login()
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    print("✓ Model loaded!")

    # Step 4: Push to hub
    print("Step 4: Uploading to HuggingFace Hub...")
    print("-" * 60)
    print(f"Repository: {REPO_NAME}")
    print()

    # Push model
    model.push_to_hub(REPO_NAME)
    tokenizer.push_to_hub(REPO_NAME)

    # Push model card
    api = HfApi()
    with open("temp_model_card.md", "w") as f:
        f.write(model_card)

    api.upload_file(
        path_or_fileobj="temp_model_card.md",
        path_in_repo="README.md",
        repo_id=REPO_NAME,
        repo_type="model"
    )

    # Cleanup
    os.remove("temp_model_card.md")

    print("="*60)
    print("✓ Upload complete!")
    print("="*60)
    print(f"Your model is now available at:")
    print(f"https://huggingface.co/{REPO_NAME}")
    print()
    print("Others can now use your model with:")
    print(f'model = AutoModelForCausalLM.from_pretrained("{REPO_NAME}")')
    print("="*60)

if __name__ == "__main__":
    upload_model()