import torch
from torch.utils.data import Dataset

import torch
import pandas as pd
from torch.utils.data import Dataset
from preprocess import preprocess_tweet

def load_data(file_path):
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)
    texts = data['tweets'].tolist()  # Extract the tweet text
    # Label the classes: 1 for "Figurative", "Sarcasm", or "Irony", otherwise 0
    labels = [1 if label in ["figurative", "sarcasm", "irony"] else 0 for label in data['class']]
    return texts, labels

def create_dataset(texts, labels, tokenizer, max_length=128):
    class SarcasmDataset(Dataset):
        def __init__(self, texts, labels):
            self.texts = [preprocess_tweet(text) for text in texts]
            self.labels = labels

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
            inputs['labels'] = torch.tensor(label, dtype=torch.long)
            return {key: val.squeeze() for key, val in inputs.items()}

    return SarcasmDataset(texts, labels)
