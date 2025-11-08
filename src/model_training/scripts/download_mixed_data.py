from datasets import load_dataset
from tqdm import tqdm
import os
import random

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, 'mixed_dataset.txt')


DATASET_MIX = {
    'wikitext': 0.70,      
    'wikipedia': 0.30,     
}

TARGET_SAMPLES = 500000  # Increased for improved teacher model

def download_and_sample_dataset(dataset_name, proportion, total_target_samples=100000):
    """
    Download a dataset and sample the appropriate proportion.
    """
    num_samples = int(total_target_samples * proportion)

    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"Target samples: {num_samples:,} ({proportion*100:.0f}% of total)")
    print(f"{'='*60}")

    if dataset_name == 'wikitext':
        print("Loading WikiText-103 (full training set)...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        texts = [sample['text'].strip() for sample in dataset if sample['text'].strip()]

    elif dataset_name == 'wikipedia':
        print("Loading Wikipedia (English, 20231101 version)...")
        print("This may take 10-15 minutes (loading 60% of dataset)...")

        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:60%]")

        texts = [sample['text'].strip() for sample in dataset if sample['text'].strip() and len(sample['text']) > 100]

    print(f"✓ Downloaded {len(texts):,} samples from {dataset_name}")

    # Sample the desired proportion
    if len(texts) > num_samples:
        print(f"Sampling {num_samples:,} from {len(texts):,} available samples...")
        random.seed(42)
        texts = random.sample(texts, num_samples)

    print(f"✓ Selected {len(texts):,} samples for final mix")

    return texts

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("="*70)
    print("MULTI-DATASET DOWNLOADER FOR KNOWLEDGE DISTILLATION")
    print("="*70)
    print("\nDataset Mix Strategy (matches teacher training):")
    for name, prop in DATASET_MIX.items():
        print(f"  - {name}: {prop*100:.0f}%")
    print(f"\nTarget total samples: {TARGET_SAMPLES:,}")
    print("="*70)
    print()

    all_texts = []

    
    for dataset_name, proportion in DATASET_MIX.items():
        texts = download_and_sample_dataset(dataset_name, proportion, TARGET_SAMPLES)
        all_texts.extend(texts)

   
    print(f"\n{'='*60}")
    print("Mixing and shuffling datasets...")
    print(f"{'='*60}")
    random.seed(42)
    random.shuffle(all_texts)
    print(f"✓ Total samples in mixed dataset: {len(all_texts):,}")

    # Save to file
    print(f"\nSaving mixed dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for text in tqdm(all_texts, desc="Writing to file"):
            f.write(text + '\n\n')

    print(f"\n{'='*70}")
    print("✓ DOWNLOAD COMPLETE!")
    print(f"{'='*70}")
    print(f"Mixed dataset saved to: {OUTPUT_FILE}")
    print(f"Total samples: {len(all_texts):,}")
    print(f"\nDataset composition:")
    print(f"  - WikiText-103:      ~{int(len(all_texts) * 0.70):,} samples (70%)")
    print(f"  - Wikipedia 2023:    ~{int(len(all_texts) * 0.30):,} samples (30%)")
    print(f"\nEstimated tokens: ~120M tokens (5x larger than previous)")
    print(f"\nNext step:")
    print(f"  Run: python preprocess_mixed_data.py")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()