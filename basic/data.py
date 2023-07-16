from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

def parse_line(line):
    utterance_data, intent_label = line.split(" <=> ")
    items = utterance_data.split()
    words = [item.rsplit(":", 1)[0] for item in items]
    word_labels = [item.rsplit(":", 1)[1]for item in items]
    return {
        "intent_label": intent_label,
        "words": " ".join(words),
        "word_labels": " ".join(word_labels),
        "length": len(words),
    }

def load_data(data_path:str):
    lines = Path(data_path).read_text("utf-8").strip().splitlines()
    data = [parse_line(line) for line in lines]

    return pd.DataFrame(data)

def encode_dataset(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length),
                         dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    return {"input_ids":torch.from_numpy(token_ids), 
            "attention_mask": torch.from_numpy(attention_masks)}

def load_intent(data_path:str):
    df = load_data(data_path)
    intent_label = df["intent_label"].unique()
    intent_dict = {label: i for i, label in enumerate(intent_label)}
    return intent_dict