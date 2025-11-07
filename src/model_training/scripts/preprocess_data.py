from transformers import AutoTokenizer
from datasets import load_dataset
import os

# Define paths
RAW_DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'wikitext_full.txt')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Model name for tokenizer
MODEL_NAME = "distilgpt2"
BLOCK_SIZE = 512 # Max sequence length for the model

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set pad_token_id for generation if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading raw data from {RAW_DATA_FILE}...")
    # Load the text file as a dataset
    dataset = load_dataset("text", data_files={"train": RAW_DATA_FILE})

    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=BLOCK_SIZE)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(), # Use multiple processes for faster tokenization
        remove_columns=dataset["train"].column_names
    )

    print("Grouping texts into blocks...")
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
    )

    print("\n" + "="*60)
    print("Creating train/validation split...")
    print("="*60)

    split_dataset = lm_dataset["train"].train_test_split(
        test_size=0.1,
        seed=42
    )

    print(f"Training samples: {len(split_dataset["train"]):,}")
    print(f"Validation samples: {len(split_dataset["test"]):,}")
    print("="*60 + "\n")

    print(f"Saving processed dataset to {PROCESSED_DATA_DIR}...")
    lm_dataset.save_to_disk(PROCESSED_DATA_DIR)
    print("Processed data saved successfully.")

if __name__ == "__main__":
    main()
