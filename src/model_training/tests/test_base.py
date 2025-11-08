import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'output', 'final_model')

def test_model():
    print("="*60)
    print("Loading your fine-tuned model...")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"✓ Model loaded on {device}")
    print(f"✓ Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print()

    # Test prompts
    test_prompts = [
        "The history of",
        "In the field of science,",
        "The human brain is",
        "Machine learning is a method of",
        "Wikipedia is an online",
    ]

    print("="*60)
    print("Testing predictions...")
    print("="*60)
    print()

    for prompt in test_prompts:
        print(f"Prompt: '{prompt}'")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                num_return_sequences=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
            )
        for i, sequence in enumerate(output, 1):
            generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
            print(f"    Prediction {i}: {generated_text}")

        print()
if __name__ == "__main__":
    test_model()

