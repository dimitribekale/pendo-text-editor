from datasets import load_dataset
import os

# Define paths
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, 'openwebtext_subset.txt')

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    print("Downloading OpenWebText dataset (subset)... This may take a while.")
    # Load a small subset for demonstration and initial fine-tuning
    # Adjust the split as needed, e.g., "train[:10%]" for 10% of the training data
    dataset = load_dataset("openwebtext", split="train[:1%]")
    print(f"Downloaded {len(dataset)} samples.")

    print(f"Saving raw data to {OUTPUT_FILE}...")
    from tqdm import tqdm
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(tqdm(dataset, desc="Saving raw data")):
            f.write(sample['text'] + '\n\n') # Add double newline to separate documents
    print("Raw data saved successfully.")

if __name__ == "__main__":
    main()

