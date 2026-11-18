"""Fine-tunes one model on one GPU.

Run from the `src` directory, choosing the GPU with CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=0 python -m model_training.finetune.train \
        --model HuggingFaceTB/SmolLM2-135M --run-name smollm2-135m-writing
"""

import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .config import parse_config
from .dataset import load_training_set, load_validation_sets


def _training_arguments(config):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return TrainingArguments(
        output_dir=str(config.output_dir / "checkpoints"),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_all_loss",
        greater_is_better=False,
        logging_steps=max(1, config.eval_steps // 5),
        dataloader_num_workers=0 if config.train_limit else min(config.workers, 8),
        report_to="none",
        seed=config.seed,
    )


def main():
    config = parse_config()
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model)

    trainer = Trainer(
        model=model,
        args=_training_arguments(config),
        train_dataset=load_training_set(tokenizer, config),
        eval_dataset=load_validation_sets(tokenizer, config),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    before = trainer.evaluate()
    trainer.train()
    after = trainer.evaluate()

    final_dir = config.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    report = {"before": before, "after": after}
    (config.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"Saved best model to {final_dir}")
    for name in sorted(key for key in after if key.endswith("_loss")):
        print(f"{name}: {before[name]:.4f} -> {after[name]:.4f}")


if __name__ == "__main__":
    main()
