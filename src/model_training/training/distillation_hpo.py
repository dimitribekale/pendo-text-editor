import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import optuna 
from functools import partial

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TEACHER_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'teacher_model')
BEST_STUDENT_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'student_model_best_hpo')


TEACHER_NAME = TEACHER_MODEL_DIR
STUDENT_NAME = "gpt2"

BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 2
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 1000  

# Generation testing prompts
TEST_PROMPTS = [
    "The history of",
    "In the field of science,",
    "Machine learning is",
    "The main reason for",
    "According to the"
]

# ==============================================================================
# DISTILLATION LOSS FUNCTION (Unchanged)
# ==============================================================================

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    mask = (labels != -100).float()
    num_tokens = mask.sum()
    if num_tokens == 0:
        device = student_logits.device
        return (torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device))
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kl_div = F.kl_div(
        student_log_soft.view(-1, student_logits.size(-1)),
        teacher_soft.view(-1, teacher_logits.size(-1)),
        reduction='none'
    ).sum(dim=-1)
    distill_loss = (kl_div * mask.view(-1)).sum() / num_tokens
    distill_loss = distill_loss * (temperature ** 2)
    total_loss = (1 - alpha) * task_loss + alpha * distill_loss
    return total_loss, distill_loss, task_loss

# ==============================================================================
# GENERATION TESTING (Unchanged)
# ==============================================================================

def test_generation(model, tokenizer, device, prompts, epoch, max_length=30):

    print("\n" + "="*70)
    print(f"🧪 GENERATION TEST - Epoch {epoch}")
    print("="*70)
    model.eval()
    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                temperature=0.7,
                repetition_penalty=1.7,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        words = generated_text.split()
        has_repetition = False
        if len(words) > 3:
            for i in range(len(words) - 1):
                if words[i] == words[i+1]:
                    has_repetition = True
                    break
        print(f"  Output: {generated_text}")
        if has_repetition:
            print("  ⚠️  WARNING: REPETITION DETECTED (possible mode collapse!)")
        else:
            print("  ✓ Looks good!")
    print("="*70)
    model.train()

# ==============================================================================
# HPO OBJECTIVE FUNCTION
# This is the function that Optuna will call.
# ==============================================================================

def run_trial(trial, device, n_gpus, train_loader, eval_loader, tokenizer, teacher):
    """
    Run a single training and evaluation trial for Optuna.
    """
    
    # --- 1. Define Hyperparameter Search Space ---
    # Optuna will pick values from these ranges for each trial.
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
    alpha = trial.suggest_float("alpha", 0.3, 0.7)
    temperature = trial.suggest_float("temperature", 0.8, 2.0)
    
    # --- 2. Create Model and Optimizer for this Trial ---
    # A *new* student model and optimizer must be created for *every* trial
    # to ensure the weights are reset.
    student = AutoModelForCausalLM.from_pretrained(STUDENT_NAME)
    if n_gpus > 1:
        student = nn.DataParallel(student)
    student.to(device)
    student.train()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # --- 3. Create Scheduler ---
    NUM_EPOCHS_PER_TRIAL = 1 
    steps_per_epoch = len(train_loader) // GRADIENT_ACCUMULATION
    total_steps = steps_per_epoch * NUM_EPOCHS_PER_TRIAL

    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        progress = float(current_step - WARMUP_STEPS) / float(max(1, total_steps - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*70}")
    print(f"STARTING TRIAL {trial.number}:")
    print(f"  LR: {learning_rate:.2e}, Alpha: {alpha:.2f}, Temp: {temperature:.2f}")
    print(f"{'='*70}")
    
    # --- 4. Run Training Loop (for 1 Epoch) ---
    student.train()
    total_loss = 0
    total_distill_loss = 0
    total_task_loss = 0
    
    optimizer.zero_grad()
    progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} Training")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits

        student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        loss, distill_loss, task_loss = distillation_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature,
            alpha
        )
        
        loss = loss / GRADIENT_ACCUMULATION
        loss.backward()

        total_loss += loss.item() * GRADIENT_ACCUMULATION
        total_distill_loss += distill_loss.item()
        total_task_loss += task_loss.item()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({
                'loss': f'{loss.item() * GRADIENT_ACCUMULATION:.4f}',
                'distill': f'{distill_loss.item():.4f}',
                'task': f'{task_loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

    avg_train_loss = total_loss / len(train_loader)
    print(f"\nTrial {trial.number} Training Summary:")
    print(f"  Avg Total Loss:   {avg_train_loss:.4f}")
    
    # --- 5. Run Evaluation Loop ---
    print(f"\nEvaluating Trial {trial.number}...")
    student.eval()
    eval_loss = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Trial {trial.number} Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            batch_loss = outputs.loss.mean() if hasattr(outputs.loss, 'mean') else outputs.loss
            eval_loss += batch_loss.item()

    avg_eval_loss = eval_loss / len(eval_loader)
    perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()

    print(f"\nTrial {trial.number} Evaluation:")
    print(f"  Eval Loss:  {avg_eval_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    # --- 6. Run Generation Test ---
    test_generation(student, tokenizer, device, TEST_PROMPTS, epoch=f"Trial {trial.number}")
    
    # --- 7. Return the Metric to Optuna ---

    return avg_eval_loss



def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print("="*70)
    print("KNOWLEDGE DISTILLATION - OPTUNA HPO")
    print("="*70)
    print(f"Device: {device}, GPUs: {n_gpus}")
    
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading teacher model from {TEACHER_NAME}...")
    if not os.path.exists(TEACHER_NAME):
        print(f"\n❌ ERROR: Teacher model not found at {TEACHER_NAME}")
        return
    
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_NAME)
    if n_gpus > 1:
        teacher = nn.DataParallel(teacher)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("✓ Teacher loaded successfully!")

    print(f"Loading dataset from {PROCESSED_DATA_DIR}...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"✓ Training samples: {len(train_dataset):,}")
    print(f"✓ Validation samples: {len(eval_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=32, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=32, pin_memory=True
    )

    # --- Setup Optuna Study ---
    print("="*70)
    print("🚀 STARTING OPTUNA STUDY...")
    print("="*70)
    
    # We want to *minimize* the evaluation loss
    study = optuna.create_study(direction="minimize")

    # We use `partial` to pass our loaded data (teacher, loaders, etc.)
    # to the `run_trial` function, since Optuna only expects a
    # function that takes one argument: `trial`.
    objective_func = partial(
        run_trial,
        device=device,
        n_gpus=n_gpus,
        train_loader=train_loader,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        teacher=teacher
    )

    # --- 7. Run the HPO Study ---
    N_TRIALS = 20
    study.optimize(objective_func, n_trials=N_TRIALS)

    # --- 8. Print the Best Results ---
    print("\n" + "="*70)
    print("🎉 HPO STUDY COMPLETE!")
    print("="*70)
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("1. Take the 'Best Hyperparameters' from above.")
    print("2. Plug them back into your *original* 5-epoch training script.")
    print("3. Run that script ONCE to train your final, optimized model.")
    print("="*70)


if __name__ == "__main__":
    main()