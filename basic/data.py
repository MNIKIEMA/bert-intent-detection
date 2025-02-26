import typing as t
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class Task(Enum):
    INTENT = "intent"
    JOINT = "joint"


class Split(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


def parse_line(line):
    utterance_data, intent_label = line.split(" <=> ")
    items = utterance_data.split()
    words = [item.rsplit(":", 1)[0] for item in items]
    word_labels = [item.rsplit(":", 1)[1] for item in items]
    return {
        "intent_label": intent_label,
        "words": " ".join(words),
        "word_labels": " ".join(word_labels),
        "length": len(words),
    }


def load_lines(data_path: str):
    lines = Path(data_path).read_text("utf-8").strip().splitlines()
    return lines


def load_data(data_path: str):
    lines = load_lines(data_path)
    data = [parse_line(line) for line in lines]
    return pd.DataFrame(data)


def get_intent_map(data_path: str):
    intent_names = load_lines(data_path)
    return {label: i for i, label in enumerate(intent_names)}


def get_slot_map(data_path: str):
    slot_names = ["[PAD]"]
    slot_names += load_lines(data_path)
    slot_map: t.Dict[str, int] = {}
    for label in slot_names:
        slot_map[label] = len(slot_map)
    return slot_map


def encode_dataset(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded  # fmt: skip
    attention_masks = (token_ids != 0).astype(np.int32)
    return {
        "input_ids": torch.from_numpy(token_ids),
        "attention_mask": torch.from_numpy(attention_masks),
    }


def encode_token_labels(text_sequences, slot_names, tokenizer, slot_map, max_length):
    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate(zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if expand_label not in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        try:
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels  # fmt: skip
        except:
            print(len(tokenizer.tokenize(text_sequence)), len(encoded_labels))
            print(text_sequence)
            print(word_labels)
    return encoded


def load_intent(data_path: str):
    intent2ids = get_intent_map(data_path)
    return intent2ids


def load_slot(data_path: str) -> t.Dict[str, int]:
    slot_label = get_slot_map(data_path)
    return slot_label


def encode_texts_and_intent(
    tokenizer, df: pd.DataFrame, intent_dict: t.Dict[str, int], max_length: int
):
    intent_labels = torch.from_numpy(df["intent_label"].map(intent_dict).values)
    encoded_texts = encode_dataset(tokenizer, df["words"], max_length)
    return TensorDataset(
        encoded_texts["input_ids"],
        encoded_texts["attention_mask"],
        intent_labels,
    )


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloaders(
    data_path,
    batch_size,
    tokenizer,
    max_length=42,
    shuffle=True,
    task: str = "intent",
    split: str = "train",
):
    task_type = Task(task)
    split_type = Split(split)
    df = load_data(data_path)
    intent_dict = load_intent(data_path)
    dataset = encode_texts_and_intent(tokenizer, df, intent_dict, max_length)
    if task_type == Task.INTENT:
        return create_dataloader(dataset, batch_size, shuffle)
    else:
        slot_dict = load_slot(data_path)
        slot_dataset = encode_token_labels(
            df["words"], df["word_labels"], tokenizer, slot_dict, max_length
        )
        dataset = TensorDataset(
            dataset[0],
            dataset[1],
            dataset[2],
            torch.from_numpy(slot_dataset),
        )
        return create_dataloader(dataset, batch_size, shuffle)
