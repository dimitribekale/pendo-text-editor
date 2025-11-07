import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TEACHER_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'teacher_model')
STUDENT_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'student_model')

TEACHER_NAME = "gpt2-medium"
STUDENT_NAME = "gpt2"

BATCH_SIZE = 32       
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
TEMPERATURE = 2.0           
ALPHA = 0.5      
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Compute combined distillation loss with PROPER masking.

    FIXED: Masks padding tokens, proper normalization

    Args:
        student_logits: Student model output logits [batch, seq_len, vocab]
        teacher_logits: Teacher model output logits [batch, seq_len, vocab]
        labels: True labels [batch, seq_len]
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss

    Returns:
        total_loss, distill_loss, task_loss
    """

    # Create mask for non-padding tokens
    # labels == -100 means padding/ignore token
    mask = (labels != -100).float()
    num_tokens = mask.sum()

    if num_tokens == 0:
        # Safety: if all tokens are padding, return zero loss
        return torch.tensor(0.0, device=student_logits.device), \
                torch.tensor(0.0, device=student_logits.device), \
                torch.tensor(0.0, device=student_logits.device)


    # === DISTILLATION LOSS (with masking!) ===

    # Reshape: [batch, seq_len, vocab] -> [batch * seq_len, vocab]
    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
    mask_flat = mask.view(-1)

    # Apply temperature scaling
    student_soft = F.log_softmax(student_logits_flat / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits_flat / temperature, dim=-1)

    # Compute KL divergence for each position
    kl_div = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='none'  # Don't reduce yet - we need to mask first!
    ).sum(dim=-1)  # Sum over vocabulary dimension

    # Apply mask and average only over non-padding tokens
    masked_kl = (kl_div * mask_flat).sum() / num_tokens

    # Scale by temperature^2 to balance gradient magnitude
    distill_loss = masked_kl * (temperature ** 2)


    # === TASK LOSS (already handles padding via ignore_index) ===
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100  # This already ignores padding!
    )


    # === COMBINE BOTH LOSSES ===
    total_loss = alpha * distill_loss + (1 - alpha) * task_loss

    return total_loss, distill_loss, task_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Teacher: {TEACHER_NAME} (355M) - Frozen")
    print(f"Student: {STUDENT_NAME} (124M) - Learning")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Alpha: {ALPHA}")
    print("="*60)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading teacher model from {TEACHER_MODEL_DIR}...")
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_DIR)
    teacher.to(device)
    teacher.eval()

    # Freeze all teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False # Don't update the teacher

    print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M")
    print()

    student = AutoModelForCausalLM.from_pretrained(STUDENT_NAME)
    student.to(device)
    student.train()

    print(f"Loading dataset from {PROCESSED_DATA_DIR}...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"✓ Training samples: {len(train_dataset):,}")
    print(f"✓ Validation samples: {len(eval_dataset):,}")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * NUM_EPOCHS

    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            # Warmup: Linear increase from 0 to 1
            return float(current_step) / float(max(1, WARMUP_STEPS))
        # Cosine decay after warmup
        progress = float(current_step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("="*60)
    print("Starting distillation training...")
    print(f"Total epochs: {NUM_EPOCHS}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total steps: {total_steps}")
    print(f"Estimated time: 4-5 hours on 2x H100")
    print("="*60)
    print()

    best_eval_loss = float('inf')
    global_step = 0

    # === MAIN TRAINING LOOP ===
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        # Training phase
        student.train()
        total_loss = 0
        total_distill_loss = 0
        total_task_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # === FORWARD PASS: Both teacher and student ===

            # Teacher forward pass (no gradient computation)
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

            # Student forward pass (compute gradients)
            student_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits

            loss, distill_loss, task_loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                TEMPERATURE,
                ALPHA
            )
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent explosions)
            torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()

            # Track losses
            total_loss += loss.item()
            total_distill_loss += distill_loss.item()
            total_task_loss += task_loss.item()
            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'distill': f'{distill_loss.item():.4f}',
                'task': f'{task_loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

            # Logging every 50 steps
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"\nStep {global_step}: Avg Loss = {avg_loss:.4f}")

        # Epoch statistics
        avg_train_loss = total_loss / len(train_loader)
        avg_distill_loss = total_distill_loss / len(train_loader)
        avg_task_loss = total_task_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1} Training Summary:")
        print(f"  Avg Total Loss:   {avg_train_loss:.4f}")
        print(f"  Avg Distill Loss: {avg_distill_loss:.4f}")
        print(f"  Avg Task Loss:    {avg_task_loss:.4f}")

        # === EVALUATION PHASE ===
        print("\nEvaluating...")
        student.eval()
        eval_loss = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Only compute task loss for evaluation (standard practice)
                outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                eval_loss += outputs.loss.item()

        avg_eval_loss = eval_loss / len(eval_loader)
        perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()

        print(f"\nEpoch {epoch + 1} Evaluation:")
        print(f"  Eval Loss:  {avg_eval_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")

        # Save best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"\n✓ New best model! Saving to {STUDENT_MODEL_DIR}...")
            os.makedirs(STUDENT_MODEL_DIR, exist_ok=True)
            student.save_pretrained(STUDENT_MODEL_DIR)
            tokenizer.save_pretrained(STUDENT_MODEL_DIR)

    # === TRAINING COMPLETE ===
    print("\n" + "="*60)
    print("✓ DISTILLATION COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_eval_loss:.4f}")
    print(f"Best perplexity: {torch.exp(torch.tensor(best_eval_loss)):.2f}")
    print(f"Model saved to: {STUDENT_MODEL_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()

