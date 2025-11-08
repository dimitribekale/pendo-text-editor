from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
import os

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TEACHER_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'teacher_model')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'teacher_checkpoints')

MODEL_NAME = "gpt2-medium"

def test_teacher_generation(model, tokenizer):
    """Quick generation test to verify teacher quality"""
    print("\n" + "="*70)
    print("🧪 QUICK TEACHER GENERATION TEST")
    print("="*70)

    test_prompts = [
        "The history of",
        "Machine learning is",
        "According to the"
    ]

    model.eval()
    import torch

    for prompt in test_prompts:
        print(f"\nPrompt: \"{prompt}\"")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=30,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Output: {generated}")

    print("="*70)
    model.train()

def main():
    os.makedirs(TEACHER_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("="*70)
    print("NEW TEACHER MODEL TRAINING - MIXED DATASET")
    print("="*70)
    print(f"Model: {MODEL_NAME} (355M parameters)")
    print(f"Dataset: WikiText-103 (70%) + Wikipedia 2023 (30%)")
    print(f"Purpose: Teacher for knowledge distillation")
    print(f"GPUs: 3x H100 80GB")
    print("="*70)
    print()

    print(f"Loading processed dataset from {PROCESSED_DATA_DIR}...")
    lm_dataset = load_from_disk(PROCESSED_DATA_DIR)

    print(f"✓ Training samples: {len(lm_dataset['train']):,}")
    print(f"✓ Validation samples: {len(lm_dataset['test']):,}")
    print()

    

    print(f"Loading tokenizer and model for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    
    print("Dataset verification:")
    sample = lm_dataset['train'][0]
    sample_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"  Sample text (first 100 chars): {sample_text[:100]}")
    print()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Model loaded: {num_params:.1f}M parameters")
    print()

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=100,              
        save_strategy="steps",
        save_steps=100,              
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Training schedule
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,

        # Optimizer settings
        learning_rate=3e-5,          
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Learning rate scheduler
        lr_scheduler_type="cosine",
        warmup_steps=100,            

        # Performance
        bf16=True,                   # ✅ GOOD for H100
        dataloader_num_workers=8,    # ✅ GOOD for 3 GPUs

        # Logging
        logging_dir="./logs/teacher",
        logging_steps=50,
        logging_first_step=True,
        report_to=["tensorboard"],

        # Multi-GPU
        ddp_find_unused_parameters=False,
  )
    print("Initializing Trainer...")
    trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=lm_dataset["train"],
          eval_dataset=lm_dataset["test"],
          tokenizer=tokenizer,
    )
    print("="*70)
    print("Starting teacher model training...")
    print("Training on EXACT mixed dataset for perfect distillation!")
    print(f"Estimated time: ~2.5-3 hours on 3x H100")
    print("="*70)
    print()
    trainer.train()
    test_teacher_generation(model, tokenizer)
    
    print()
    print("="*70)
    print("✓ NEW TEACHER MODEL TRAINING COMPLETE!")
    print("="*70)
    trainer.save_model(TEACHER_MODEL_DIR)
    tokenizer.save_pretrained(TEACHER_MODEL_DIR)
    print(f"Model saved to: {TEACHER_MODEL_DIR}")
    print(f"Dataset: Mixed (WikiText-103 + Wikipedia 2023)")
    print()
    print("Next steps:")
    print("  1. Update distill_model.py to load this local teacher")
    print("  2. Run: python distill_model.py")
    print("="*70)

if __name__ == "__main__":
    main()

