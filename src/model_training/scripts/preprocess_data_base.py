from transformers import AutoTokenizer
from datasets import load_dataset
import os

RAW_DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'wikitext_full.txt')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


MODEL_NAME = "distilgpt2"
BLOCK_SIZE = 512 


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("="*70)
    print("BASELINE MODEL PREPROCESSING (DistilGPT-2)")
    print("="*70)
    print(f"Model tokenizer: {MODEL_NAME}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Dataset: WikiText-103 only (wikitext_full.txt)")
    print(f"Input file: {RAW_DATA_FILE}")
    print(f"Output directory: {PROCESSED_DATA_DIR}")
    print("="*70)
    print()


    if not os.path.exists(RAW_DATA_FILE):
        print(f"[ERROR] Input file not found: {RAW_DATA_FILE}")
        print("\nPlease run download_data_base.py first:")
        print("  python download_data_base.py")
        return

    
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set pad_token_id for generation 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Tokenizer loaded")
    print()

    print(f"Loading raw data from {RAW_DATA_FILE}...")
    dataset = load_dataset("text", data_files={"train": RAW_DATA_FILE})
    print(f"✓ Loaded {len(dataset['train']):,} text samples")
    print()

    
    print("Tokenizing dataset...")
    print(f"  This will convert text to token IDs using {MODEL_NAME} tokenizer")

    def tokenize_function(examples):
        """
        Tokenize text examples.

        Args:
            examples: Batch of text examples

        Returns:
            Tokenized examples with input_ids and attention_mask
        """
        return tokenizer(examples["text"], truncation=True, max_length=BLOCK_SIZE)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),  # Use all CPU cores for faster processing
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )

    print("✓ Tokenization complete")
    print()

    # Group texts into blocks
    print(f"Grouping texts into {BLOCK_SIZE}-token blocks...")
    print(f"  This creates fixed-length sequences for efficient training")

    def group_texts(examples):
        """
        Group tokenized texts into fixed-length blocks.

        This function:
        1. Concatenates all tokenized texts
        2. Splits into blocks of exactly BLOCK_SIZE tokens
        3. Drops any remainder smaller than BLOCK_SIZE
        4. Creates labels (copy of input_ids for causal LM)

        Args:
            examples: Batch of tokenized examples

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Drop the small remainder (could add padding instead if model supports it)
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE

        # Split by chunks of BLOCK_SIZE
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }

        # Create labels (for causal language modeling, labels = input_ids)
        result["labels"] = result["input_ids"].copy()

        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        desc="Grouping texts"
    )

    print("✓ Grouping complete")
    print()

    
    print("="*70)
    print("Creating train/validation split (90/10)...")
    print("="*70)
    print(f"Using seed=42 for reproducibility")
    print()

    split_dataset = lm_dataset["train"].train_test_split(
        test_size=0.1,
        seed=42 
    )

    print(f"✓ Training samples: {len(split_dataset['train']):,}")
    print(f"✓ Validation samples: {len(split_dataset['test']):,}")
    print()

    
    print(f"Saving processed dataset to {PROCESSED_DATA_DIR}...")
    split_dataset.save_to_disk(PROCESSED_DATA_DIR)

if __name__ == "__main__":
    main()
