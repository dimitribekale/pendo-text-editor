from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
import os

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TEACHER_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'teacher_model')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'teacher_checkpoints')

MODEL_NAME = "gpt2-medium"

def main():
    os.makedirs(TEACHER_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("="*60)
    print("TEACHER MODEL TRAINING")
    print("="*60)
    print(f"Model: {MODEL_NAME} (355M parameters)")
    print(f"Purpose: Large teacher for knowledge distillation")
    print("="*60)
    print()

    print(f"Loading processed dataset from {PROCESSED_DATA_DIR}...")
    lm_dataset = load_from_disk(PROCESSED_DATA_DIR)

    print(f"✓ Training samples: {len(lm_dataset['train']):,}")
    print(f"✓ Validation samples: {len(lm_dataset['test']):,}")
    print()

    print(f"Loading tokenizer and model for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Model loaded: {num_params:.1f}M parameters")
    print()

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=500,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Logging
        logging_dir="./logs/teacher",
        logging_steps=50,
        logging_first_step=True,
        report_to=["tensorboard"],
        # Multi-GPU optimization
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,
    )
    print("Initializing Trainer...")
    trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=lm_dataset["train"],
          eval_dataset=lm_dataset["test"],  # Proper validation set!
          tokenizer=tokenizer,
    )
    print("="*60)
    print("Starting teacher model training...")
    print("This will take approximately 3-4 hours on 2x H100")
    print("="*60)
    print()

    trainer.train()

    print()
    print("="*60)
    print(f"Saving teacher model to {TEACHER_MODEL_DIR}...")
    trainer.save_model(TEACHER_MODEL_DIR)
    tokenizer.save_pretrained(TEACHER_MODEL_DIR)

    print("="*60)
    print("✓ TEACHER MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {TEACHER_MODEL_DIR}")
    print(f"Next step: Train student via distillation")
    print("="*60)

if __name__ == "__main__":
    main()

