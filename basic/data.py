from pathlib import Path
import pandas as pd
import numpy as np
import torch


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


def load_data(data_path: str):
    lines = Path(data_path).read_text("utf-8").strip().splitlines()
    data = [parse_line(line) for line in lines]

    return pd.DataFrame(data)


def encode_dataset(tokenizer, text_sequences, max_length):
    token_ids = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    return {
        "input_ids": torch.from_numpy(token_ids),
        "attention_mask": torch.from_numpy(attention_masks),
    }


def load_intent(data_path: str):
    df = load_data(data_path)
    intent_label = df["intent_label"].unique()
    ids2intent = {label: i for i, label in enumerate(intent_label)}
    intent_ids = df["intent_label"].map(ids2intent).values
    return ids2intent, intent_ids


def load_slot(data_path: str) -> dict:
    df = load_data(data_path)
    slot_label = df["word_labels"].unique()
    slot_dict = {label: i for i, label in enumerate(slot_label)}
    return slot_dict


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
            encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
        except:
            print(len(tokenizer.tokenize(text_sequence)), len(encoded_labels))
            print(text_sequence)
            print(word_labels)
    return encoded


def get_dataloaders(data_path, batch_size, tokenizer):
    intent_dict = load_intent(data_path)
    slot_dict = load_slot(data_path)
    df_train = load_data(data_path)
    df_valid = load_data(data_path)
    df_test = load_data(data_path)

    slot_train = encode_token_labels(
    df_train["words"], df_train["word_labels"], tokenizer, slot_dict, 42
    )
    slot_valid = encode_token_labels(
        df_valid["words"], df_valid["word_labels"], tokenizer, slot_dict, 42
    )
    slot_test = encode_token_labels(
        df_test["words"], df_test["word_labels"], tokenizer, slot_dict, 42
    )
