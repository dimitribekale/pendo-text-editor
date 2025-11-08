"""
Configuration for re-uploading models to HuggingFace Hub
"""
import os

# ==============================================================================
# AUTHENTICATION
# ==============================================================================

# HuggingFace token - set this or use HF_TOKEN environment variable
HF_TOKEN = None # Will check environment if None

# Repository visibility
PRIVATE = False  # Set to True for private repositories

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

# Baseline Model (DistilGPT2 fine-tuned on WikiText-103)
BASELINE_CONFIG = {
    "repo_name": "bekalebendong/pendo-distilgpt2-wikitext",
    "model_type": "AutoModelForCausalLM",
    "display_name": "Pendo DistilGPT2 - Fine-tuned on WikiText-103",

    # Model specs
    "base_model": "distilgpt2",
    "parameters": "82M",
    "layers": 6,
    "hidden_size": 768,
    "attention_heads": 12,
    "context_length": 512,

    # Training details
    "dataset": "WikiText-103",
    "dataset_size": "~100M tokens",
    "training_hardware": "2x NVIDIA H100 80GB",
    "training_time": "~42 minutes",
    "epochs": 3,
    "batch_size": 32,
    "effective_batch_size": 128,
    "gradient_accumulation_steps": 2,
    "learning_rate": "5e-5",
    "lr_scheduler": "cosine with 500 warmup steps",
    "mixed_precision": "bf16",
    "block_size": 512,

    # Performance metrics
    "final_train_loss": 3.379,
    "final_val_loss": 3.206,
    "perplexity": "~25",

    # Tags
    "tags": [
        "text-generation",
        "distilgpt2",
        "wikitext",
        "causal-lm",
        "pytorch"
    ],

    # Metadata
    "author": "Dimitri Bekale",
    "license": "mit",
    "language": "en",
    "year": 2025,
    "project_url": "https://github.com/dimitribekale/pendo-text-editor"
}

# Teacher Model (GPT-2 Medium fine-tuned on WikiText-103)
TEACHER_CONFIG = {
    "repo_name": "bekalebendong/pendo-gpt2-medium-teacher",
    "model_type": "AutoModelForCausalLM",
    "display_name": "Pendo GPT-2 Medium Teacher Model",

    # Model specs
    "base_model": "gpt2-medium",
    "parameters": "355M",
    "layers": 24,
    "hidden_size": 1024,
    "attention_heads": 16,
    "context_length": 1024,

    # Training details
    "dataset": "WikiText-103 (full) + Wikipedia EN (20231101, 30%)",
    "dataset_size": "~100M+ tokens",
    "training_hardware": "2x NVIDIA H100 80GB",
    "training_time": "~3 hours",
    "epochs": 3,
    "batch_size": 16,
    "effective_batch_size": 128,
    "gradient_accumulation_steps": 4,
    "learning_rate": "3e-5",
    "lr_scheduler": "cosine with 1000 warmup steps",
    "mixed_precision": "bf16",
    "block_size": 512,

    # Performance metrics
    "final_train_loss": 2.822,
    "final_val_loss": 2.706,
    "perplexity": "~15",
    "improvement": "16% improvement over baseline",

    # Tags
    "tags": [
        "text-generation",
        "gpt2",
        "knowledge-distillation",
        "teacher-model",
        "wikitext",
        "causal-lm",
        "pytorch"
    ],

    # Metadata
    "author": "Dimitri Bekale",
    "license": "mit",
    "language": "en",
    "year": 2025,
    "project_url": "https://github.com/dimitribekale/pendo-text-editor"
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_token():
    """Get HuggingFace token from config or environment"""
    token = HF_TOKEN
    if token is None:
        token = os.environ.get('HF_TOKEN')

    if token is None:
        raise ValueError(
            "HuggingFace token not found!\n"
            "Please either:\n"
            "  1. Set HF_TOKEN in config.py\n"
            "  2. Set HF_TOKEN environment variable\n"
            "\n"
            "Get your token from:\n"
            "  https://huggingface.co/settings/tokens\n"
            "  (Make sure it has WRITE permission!)"
        )

    return token

def get_model_configs():
    """Get list of all model configurations"""
    return [BASELINE_CONFIG, TEACHER_CONFIG]
