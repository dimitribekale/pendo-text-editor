from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_from_disk
import os

# Define paths
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
FINAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'final_model')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'checkpoints')

# Model name for tokenizer and base model
MODEL_NAME = "distilgpt2"

def main():
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Loading processed dataset from {PROCESSED_DATA_DIR}...")
    lm_dataset = load_from_disk(PROCESSED_DATA_DIR)

    print(f"Loading tokenizer and model for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Set pad_token_id for generation if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_step=500,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        max_grad_norm=1.0,
        bf16=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir="./logs",
        logging_steps=50,
        logging_first_step=True,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,    
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned model and tokenizer to {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("Fine-tuning complete and model saved.")

if __name__ == "__main__":
    main()
