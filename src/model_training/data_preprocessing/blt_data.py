from datasets import load_dataset
import numpy as np
import torch

DATASET = "Na0s/Next_Token_Prediction_dataset"

dataset = load_dataset(DATASET)
DATA_PATH = "blt_data"
dataset.save_to_disk(DATA_PATH)