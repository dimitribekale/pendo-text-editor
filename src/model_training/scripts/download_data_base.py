import os
from tqdm import tqdm
from datasets import load_dataset


RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, 'wikitext_full.txt')

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("Downloading Wikitext-103 dataset (FULL training set)...")

    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
    print(f"✓ Downloaded {len(dataset)} samples.")

    print(f"Saving raw data to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(tqdm(dataset, desc="Saving raw data")):
            # Filter out empty lines
            text = sample['text'].strip()
            if text:  # Only write non-empty text
                f.write(text + '\n\n')  # Add double newline to separate documents
    print("✓ Raw data saved successfully.")

if __name__ == "__main__":
    main()

