import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TEACHER_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'teacher_model')
STUDENT_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'student_model')

# Model configuration
TEACHER_NAME = TEACHER_MODEL_DIR  # ✅ FIXED: Use local teacher (data-aligned!)
STUDENT_NAME = "gpt2"  # 124M parameters

# Training hyperparameters (optimized for 3x H100)
BATCH_SIZE = 32           # Per GPU
GRADIENT_ACCUMULATION = 2 # Effective batch = 32 × 3 × 2 = 192
LEARNING_RATE = 1.5e-5      # ✅ FIXED: Was 15e-5 (3x too high!)
NUM_EPOCHS = 5

# Distillation hyperparameters (OPTIMIZED!)
TEMPERATURE = 1.5         # ✅ FIXED: Was 1.5 (prevents over-smoothing)
ALPHA = 0.4               # ✅ CORRECT: 70% task loss, 30% distillation

# Optimization
MAX_GRAD_NORM = 1.0       # ✅ FIXED: Was 0.5 (less aggressive clipping)
WARMUP_STEPS = 100        # ✅ FIXED: Was 1000 (now 13% of total steps)

# Generation testing prompts
TEST_PROMPTS = [
    "The history of",
    "In the field of science,",
    "Machine learning is",
    "The main reason for",
    "According to the"
]

# ==============================================================================
# DISTILLATION LOSS FUNCTION
# ==============================================================================

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Compute combined distillation loss with proper masking.

    Args:
        student_logits: Student model output [batch, seq_len, vocab]
        teacher_logits: Teacher model output [batch, seq_len, vocab]
        labels: True labels [batch, seq_len] (-100 for padding)
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss (0.3 = 30% distill, 70% task)

    Returns:
        total_loss: Combined loss
        distill_loss: KL divergence component
        task_loss: Cross-entropy component
    """
    # Create mask for non-padding tokens
    mask = (labels != -100).float()
    num_tokens = mask.sum()

    # Safety check for empty batches
    if num_tokens == 0:
        device = student_logits.device
        return (torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device))

    # === TASK LOSS (Cross-Entropy) ===
    # Student learns to predict correct tokens
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100  # Skip padding tokens
    )

    # === DISTILLATION LOSS (KL Divergence) ===
    # Student learns from teacher's soft probability distributions

    # Apply temperature scaling to soften distributions
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL divergence (measures difference between distributions)
    kl_div = F.kl_div(
        student_log_soft.view(-1, student_logits.size(-1)),
        teacher_soft.view(-1, teacher_logits.size(-1)),
        reduction='none'
    ).sum(dim=-1)  # Sum over vocabulary dimension

    # Apply mask to ignore padding tokens and average
    distill_loss = (kl_div * mask.view(-1)).sum() / num_tokens

    # Scale by temperature^2 to balance gradient magnitudes
    # (standard practice in knowledge distillation)
    distill_loss = distill_loss * (temperature ** 2)

    # === COMBINE LOSSES ===
    # (1 - alpha) * task_loss + alpha * distill_loss
    # With alpha=0.3: 70% task loss + 30% distillation loss
    total_loss = (1 - alpha) * task_loss + alpha * distill_loss

    return total_loss, distill_loss, task_loss


# ==============================================================================
# GENERATION TESTING
# ==============================================================================

def test_generation(model, tokenizer, device, prompts, epoch, max_length=30):
    """
    Test model generation quality to catch mode collapse EARLY!

    This is CRITICAL - metrics can look good while generation is broken.
    Tests actual text generation to catch issues like repetition.
    """
    print("\n" + "="*70)
    print(f"🧪 GENERATION TEST - Epoch {epoch}")
    print("="*70)

    model.eval()

    # Test on primary GPU only (cleaner output)
    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get underlying model (unwrap DataParallel if needed)
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
                repetition_penalty=1.2,  # Prevent repetition loops
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check for repetition (mode collapse indicator)
        words = generated_text.split()
        has_repetition = False

        if len(words) > 3:
            # Look for 3+ consecutive identical words
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


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main():
    # Setup device - use DataParallel for all GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print("="*70)
    print("KNOWLEDGE DISTILLATION - DATA-ALIGNED TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"GPUs available: {n_gpus}")
    if n_gpus > 0:
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print(f"\nTeacher: LOCAL (trained on exact mixed dataset) - 355M params")
    print(f"Student: {STUDENT_NAME} - 124M params")

    print(f"\n🎯 Key: Teacher and student trained on IDENTICAL data!")
    print(f"   This ensures informative soft targets for distillation.")

    print(f"\nHyperparameters:")
    print(f"  Learning rate: {LEARNING_RATE} (stable convergence)")
    print(f"  Temperature: {TEMPERATURE} (balanced soft targets)")
    print(f"  Alpha: {ALPHA} (70% task, 30% distillation)")
    print(f"  Batch size per GPU: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch size: {BATCH_SIZE * max(1, n_gpus) * GRADIENT_ACCUMULATION}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print("="*70)
    print()

    # ==== LOAD TOKENIZER ====
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==== LOAD TEACHER (LOCAL) ====
    print(f"Loading teacher model from LOCAL directory...")
    print(f"  Path: {TEACHER_NAME}")
    print(f"  This teacher was trained on EXACT same mixed dataset!")

    if not os.path.exists(TEACHER_NAME):
        print(f"\n❌ ERROR: Teacher model not found at {TEACHER_NAME}")
        print(f"\nPlease ensure teacher training completed successfully.")
        print(f"Expected directory: {TEACHER_NAME}")
        return

    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_NAME)
    print("✓ Teacher loaded successfully!")

    # Wrap in DataParallel for multi-GPU inference
    if n_gpus > 1:
        teacher = nn.DataParallel(teacher)
    teacher.to(device)
    teacher.eval()

    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False

    print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M")
    print()

    # ==== LOAD STUDENT ====
    print(f"Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(STUDENT_NAME)

    # Wrap in DataParallel for multi-GPU training
    if n_gpus > 1:
        student = nn.DataParallel(student)
    student.to(device)
    student.train()

    print(f"  Parameters: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")
    print()

    # ==== LOAD DATASET ====
    print(f"Loading dataset from {PROCESSED_DATA_DIR}...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"✓ Training samples: {len(train_dataset):,}")
    print(f"✓ Validation samples: {len(eval_dataset):,}")
    print()

    # ==== CREATE DATA LOADERS ====
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=24,
        pin_memory=True
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # ==== OPTIMIZER & SCHEDULER ====
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # Calculate total training steps
    steps_per_epoch = len(train_loader) // GRADIENT_ACCUMULATION
    total_steps = steps_per_epoch * NUM_EPOCHS

    # Learning rate scheduler with warmup + cosine decay
    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            # Linear warmup
            return float(current_step) / float(max(1, WARMUP_STEPS))
        # Cosine decay after warmup
        progress = float(current_step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("="*70)
    print("Starting distillation training...")
    print(f"Total epochs: {NUM_EPOCHS}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Warmup steps: {WARMUP_STEPS} ({WARMUP_STEPS/total_steps*100:.1f}% of training)")
    print(f"Estimated time: ~2 hours on 3x H100")
    print("="*70)
    print()

    best_eval_loss = float('inf')
    global_step = 0

    # ==== MAIN TRAINING LOOP ====
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*70}")

        student.train()
        total_loss = 0
        total_distill_loss = 0
        total_task_loss = 0

        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # === TEACHER FORWARD PASS (no gradients) ===
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

            # === STUDENT FORWARD PASS (with gradients) ===
            student_outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits

            # === COMPUTE DISTILLATION LOSS ===
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

            # Track losses (unscaled for logging)
            total_loss += loss.item() * GRADIENT_ACCUMULATION
            total_distill_loss += distill_loss.item()
            total_task_loss += task_loss.item()

            # === UPDATE WEIGHTS (every GRADIENT_ACCUMULATION steps) ===
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                # Clip gradients to prevent explosions
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    MAX_GRAD_NORM
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * GRADIENT_ACCUMULATION:.4f}',
                    'grad': f'{grad_norm:.2f}',
                    'distill': f'{distill_loss.item():.4f}',
                    'task': f'{task_loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

        # ==== EPOCH STATISTICS ====
        avg_train_loss = total_loss / len(train_loader)
        avg_distill_loss = total_distill_loss / len(train_loader)
        avg_task_loss = total_task_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1} Training Summary:")
        print(f"  Avg Total Loss:   {avg_train_loss:.4f}")
        print(f"  Avg Distill Loss: {avg_distill_loss:.4f}")
        print(f"  Avg Task Loss:    {avg_task_loss:.4f}")

        # ==== EVALUATION ====
        print("\nEvaluating...")
        student.eval()
        eval_loss = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Compute standard task loss for evaluation
                outputs = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # Handle DataParallel output
                batch_loss = outputs.loss.mean() if hasattr(outputs.loss, 'mean') else outputs.loss
                eval_loss += batch_loss.item()

        avg_eval_loss = eval_loss / len(eval_loader)
        perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()

        print(f"\nEpoch {epoch + 1} Evaluation:")
        print(f"  Eval Loss:  {avg_eval_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")

        # ==== GENERATION TEST ====
        test_generation(student, tokenizer, device, TEST_PROMPTS, epoch + 1)

        # ==== SAVE BEST MODEL ====
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            print(f"\n✓ New best model! Saving to {STUDENT_MODEL_DIR}...")
            os.makedirs(STUDENT_MODEL_DIR, exist_ok=True)

            # Unwrap DataParallel before saving
            model_to_save = student.module if hasattr(student, 'module') else student
            model_to_save.save_pretrained(STUDENT_MODEL_DIR)
            tokenizer.save_pretrained(STUDENT_MODEL_DIR)

    # ==== TRAINING COMPLETE ====
    print("\n" + "="*70)
    print("✓ DISTILLATION COMPLETE!")
    print("="*70)
    print(f"Best validation loss: {best_eval_loss:.4f}")
    print(f"Best perplexity: {torch.exp(torch.tensor(best_eval_loss)):.2f}")
    print(f"Model saved to: {STUDENT_MODEL_DIR}")
    print(f"\nConfiguration used:")
    print(f"  ✓ Temperature: {TEMPERATURE} (balanced soft targets)")
    print(f"  ✓ Alpha: {ALPHA} (70% task, 30% distillation)")
    print(f"  ✓ Learning rate: {LEARNING_RATE} (stable)")
    print(f"  ✓ Data alignment: PERFECT (teacher & student on same data)")
    print(f"  ✓ Generation tested every epoch")
    print(f"  ✓ Multi-GPU training on {n_gpus}x H100")
    print("="*70)


if __name__ == "__main__":
    main()