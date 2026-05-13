
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

def load_dataset_json(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}")
    return data

def format_example(sample: dict[str, Any]) -> str:
    if str(sample.get("input", "")).strip():
        return PROMPT_WITH_INPUT.format(
            instruction=sample["instruction"],
            input=sample["input"],
            output=sample["output"],
        )
    return PROMPT_NO_INPUT.format(instruction=sample["instruction"], output=sample["output"])

def format_prompt(sample: dict[str, Any]) -> str:
    return format_example({**sample, "output": ""})

class CommonsenseDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt_ids = self.tokenizer(
            format_prompt(sample), add_special_tokens=True, truncation=False, return_attention_mask=False
        )["input_ids"]
        response_ids = self.tokenizer(
            str(sample["output"]), add_special_tokens=False, truncation=False, return_attention_mask=False
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]
        if len(response_ids) >= self.max_length:
            prompt_ids = []
            response_ids = response_ids[: self.max_length]
        else:
            prompt_ids = prompt_ids[-(self.max_length - len(response_ids)):]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids.copy()
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def _left_pad_1d(sequences, padding_value):
    max_length = max(seq.size(0) for seq in sequences)
    padded = sequences[0].new_full((len(sequences), max_length), padding_value)
    for row, seq in enumerate(sequences):
        padded[row, -seq.size(0):] = seq
    return padded

def collate_fn(batch):
    return {
        "input_ids": _left_pad_1d([x["input_ids"] for x in batch], 0),
        "attention_mask": _left_pad_1d([x["attention_mask"] for x in batch], 0),
        "labels": _left_pad_1d([x["labels"] for x in batch], -100),
    }
