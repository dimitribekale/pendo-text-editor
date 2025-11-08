import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

# Data paths
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
STUDENT_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'student_model')

# Model configuration
TEACHER_NAME = "bekalebendong/pendo-gpt2-medium-teacher"  # ✅ Load from HuggingFace Hub
STUDENT_NAME = "gpt2"  # 124M parameters

# Training hyperparameters (optimized for 3x H100)
BATCH_SIZE = 32           # Per GPU
GRADIENT_ACCUMULATION = 2 # Effective batch = 32 × 3 GPUs × 2 = 192
LEARNING_RATE = 15e-5
NUM_EPOCHS = 3

# Distillation hyperparameters (FIXED for better generation!)
TEMPERATURE = 1.5         # ✅ Lower temp = sharper distributions
ALPHA = 0.3               # ✅ 80% task loss, 20% distillation 

# Optimization
MAX_GRAD_NORM = 0.5
WARMUP_STEPS = 1000

# Generation testing prompts
TEST_PROMPTS = [
    "The history of",
    "In the field of science,",
    "Machine learning is",
    "The main reason for",
    "According to the"
]

"""def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    # Compute combined distillation loss with PROPER masking.

    # FIXED: Masks padding tokens, proper normalization

    # Args:
    #     student_logits: Student model output logits [batch, seq_len, vocab]
    #     teacher_logits: Teacher model output logits [batch, seq_len, vocab]
    #     labels: True labels [batch, seq_len]
    #     temperature: Temperature for softening distributions
    #     alpha: Weight for distillation loss

    # Returns:
    #     total_loss, distill_loss, task_loss
"""
    assert student_logits.shape == teacher_logits.shape, "Logit shapes don't match!"
    assert student_logits.shape[:-1] == labels.shape, "Label shape mismatch!"

    # Create mask for non-padding tokens
    # labels == -100 means padding/ignore token
    mask = (labels != -100).float()
    num_tokens = mask.sum()

    if num_tokens == 0:
        # Safety: if all tokens are padding, return zero loss
        return torch.tensor(0.0, device=student_logits.device), \
                torch.tensor(0.0, device=student_logits.device), \
                torch.tensor(0.0, device=student_logits.device)


    # === DISTILLATION LOSS ===

    # Reshape: [batch, seq_len, vocab] -> [batch * seq_len, vocab]
    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
    teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
    #mask_flat = mask.view(-1)

    # Apply temperature scaling
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL divergence for each position
    distill_loss = F.kl_div(
        student_log_soft.view(-1, student_logits.size(-1)),
        teacher_soft.view(-1, teacher_logits.size(-1)),
        reduction='none'
    ).sum(dim=-1)

    # Apply mask and average only over non-padding tokens
    #masked_kl = (kl_div * mask_flat).sum() / num_tokens

    # Scale by temperature^2 to balance gradient magnitude
    #distill_loss = masked_kl * (temperature ** 2)


    # === TASK LOSS ===
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # Mask and average
    distill_loss = (distill_loss * mask.view(-1)).sum() / num_tokens
    distill_loss = distill_loss * (temperature ** 2)

    # === COMBINE BOTH LOSSES ===
    #total_loss = alpha * distill_loss + (1 - alpha) * task_loss
    total_loss = (1 - alpha) * task_loss + alpha * distill_loss

    return total_loss, distill_loss, task_loss"""

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Create mask
    mask = (labels != -100).float()
    num_tokens = mask.sum()
    
    if num_tokens == 0:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    
    # Task loss
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    
    # Distillation loss (vectorized, not loops!)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    distill_loss = F.kl_div(
        student_log_soft.view(-1, student_logits.size(-1)),
        teacher_soft.view(-1, teacher_logits.size(-1)),
        reduction='none'
    ).sum(dim=-1)
    
    # Mask and average
    distill_loss = (distill_loss * mask.view(-1)).sum() / num_tokens
    distill_loss = distill_loss * (temperature ** 2)
    
    # Combine - FIXED FORMULA!
    total_loss = (1 - alpha) * task_loss + alpha * distill_loss
    
    return total_loss, distill_loss, task_loss

def test_generation(model, tokenizer, device, prompts, epoch, max_length=30):
    """
    Test model generation quality to catch mode collapse EARLY!

    Why this matters:
    - Metrics (loss/perplexity) can look perfect while generation is broken
    - Catches "of of of" repetition immediately
    - Shows actual progress each epoch
    """
    print("\n" + "="*70)
    print(f"🧪 GENERATION TEST - Epoch {epoch}")
    print("="*70)

    model.eval()

    # Test on primary GPU only (cleaner output)
    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get the underlying model (unwrap DataParallel if needed)
    test_model = model.module if hasattr(model, 'module') else model

    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")

        inputs = tokenizer(prompt, return_tensors="pt").to(test_device)

        with torch.no_grad():
            outputs = test_model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                repetition_penalty=1.2,  # ✅ Prevent "of of of" loops
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check for repetition (mode collapse indicator)
        words = generated_text.split()
        has_repetition = False

        if len(words) > 3:
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    has_repetition = True
                    break

        print(f"  Output: {generated_text}")

        if has_repetition:
            print("  ⚠️  WARNING: REPETITION DETECTED (possible mode collapse!)")
        else:
            print("  ✓ Looks good!")

    print("="*70)
    model.train()


def main():
    # Setup device - use DataParallel for all 3 GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print("="*70)
    print("KNOWLEDGE DISTILLATION TRAINING - MULTI-GPU")
    print("="*70)
    print(f"Device: {device}")
    print(f"GPUs available: {n_gpus}")
    if n_gpus > 0:
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"\nTeacher: {TEACHER_NAME} (355M) - Frozen")
    print(f"Student: {STUDENT_NAME} (124M) - Learning")
    print(f"\nHyperparameters:")
    print(f"  Temperature: {TEMPERATURE} (lower = sharper)")
    print(f"  Alpha: {ALPHA} (70% task, 30% distillation)")
    print(f"  Batch size per GPU: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch size: {BATCH_SIZE * max(1, n_gpus) * GRADIENT_ACCUMULATION}")
    print("="*70)
    print()

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load teacher from HuggingFace Hub
    print(f"Loading teacher model from HuggingFace Hub...")
    print(f"  Repository: {TEACHER_NAME}")
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_NAME)
    print("✓ Teacher loaded successfully!")

    # Wrap teacher in DataParallel for multi-GPU inference
    if n_gpus > 1:
        teacher = nn.DataParallel(teacher)
    teacher.to(device)
    teacher.eval()

    # Freeze all teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M")
    print()

    # Load student
    print(f"Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(STUDENT_NAME)

    # Wrap student in DataParallel for multi-GPU training
    if n_gpus > 1:
        student = nn.DataParallel(student)
    student.to(device)
    student.train()

    print(f"  Parameters: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")
    print()

    print(f"Loading dataset from {PROCESSED_DATA_DIR}...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"✓ Training samples: {len(train_dataset):,}")
    print(f"✓ Validation samples: {len(eval_dataset):,}")
    print()
    # After loading dataset, check a sample
    sample = train_dataset[0]
    decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"Sample text: {decoded[:200]}")
    print(f"Sample labels: {sample['labels'][:50]}")

    # DataLoaders with more workers for 3 GPUs
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # More workers for 3 GPUs
        pin_memory=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Optimizer (for student parameters only)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION

    # Learning rate scheduler with warmup + cosine decay
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        progress = float(current_step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("="*70)
    print("Starting distillation training...")
    print(f"Total epochs: {NUM_EPOCHS}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Estimated time: ~2 hours on 3x H100 (faster than before!)")
    print("="*70)
    print()

    best_eval_loss = float('inf')
    global_step = 0

    # === MAIN TRAINING LOOP ===
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*70}")

        student.train()
        total_loss = 0
        total_distill_loss = 0
        total_task_loss = 0

        optimizer.zero_grad()  # Zero gradients at start

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

            # Compute distillation loss
            loss, distill_loss, task_loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                TEMPERATURE,
                ALPHA
            )

            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()

            # Track losses (unscaled)
            total_loss += loss.item() * GRADIENT_ACCUMULATION
            total_distill_loss += distill_loss.item()
            total_task_loss += task_loss.item()

            # Update weights every GRADIENT_ACCUMULATION steps
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Update progress bar
            total_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
            progress_bar.set_postfix({
                'loss': f'{loss.item() * GRADIENT_ACCUMULATION:.4f}',
                'grad_norm': f'{total_norm:.2f}',
                'distill': f'{distill_loss.item():.4f}',
                'task': f'{task_loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

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
                eval_loss += outputs.loss.mean().item()

        avg_eval_loss = eval_loss / len(eval_loader)
        perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()

        print(f"\nEpoch {epoch + 1} Evaluation:")
        print(f"  Eval Loss:  {avg_eval_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")

        # ✅ TEST GENERATION (catch mode collapse early!)
        test_generation(student, tokenizer, device, TEST_PROMPTS, epoch + 1)

        # Save best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"\n✓ New best model! Saving to {STUDENT_MODEL_DIR}...")
            os.makedirs(STUDENT_MODEL_DIR, exist_ok=True)

            # Save the underlying model (unwrap DataParallel)
            model_to_save = student.module if hasattr(student, 'module') else student
            model_to_save.save_pretrained(STUDENT_MODEL_DIR)
            tokenizer.save_pretrained(STUDENT_MODEL_DIR)

    # === TRAINING COMPLETE ===
    print("\n" + "="*70)
    print("✓ DISTILLATION COMPLETE!")
    print("="*70)
    print(f"Best validation loss: {best_eval_loss:.4f}")
    print(f"Best perplexity: {torch.exp(torch.tensor(best_eval_loss)):.2f}")
    print(f"Model saved to: {STUDENT_MODEL_DIR}")
    print(f"\nKey improvements in this training:")
    print(f"  ✓ Temperature: {TEMPERATURE} (prevents over-smoothing)")
    print(f"  ✓ Alpha: {ALPHA} (balanced task/distillation)")
    print(f"  ✓ Generation tested every epoch (caught mode collapse)")
    print(f"  ✓ Multi-GPU training on {n_gpus}x H100")
    print("="*70)

if __name__ == "__main__":
    main()

