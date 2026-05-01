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


def format_example(sample: dict[str, Any]) -> str:
    """Format one commonsense JSON sample as an Alpaca-style prompt."""
    required = {"instruction", "output"}
    missing = required - sample.keys()
    if missing:
        raise KeyError(f"sample is missing required keys: {sorted(missing)}")

    if str(sample.get("input", "")).strip():
        return PROMPT_WITH_INPUT.format(
            instruction=sample["instruction"],
            input=sample["input"],
            output=sample["output"],
        )

    return PROMPT_NO_INPUT.format(
        instruction=sample["instruction"],
        output=sample["output"],
    )


def format_prompt(sample: dict[str, Any]) -> str:
    """Format a sample up through the response marker, without the answer."""
    return format_example({**sample, "output": ""})


class CommonsenseDataset(Dataset):
    """Tokenized commonsense instruction dataset with response-only labels."""

    def __init__(self, samples: list[dict[str, Any]], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        prompt = format_prompt(sample)
        response = str(sample["output"])

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            response_ids = response_ids + [eos_token_id]

        if len(response_ids) >= self.max_length:
            response_ids = response_ids[: self.max_length]
            prompt_ids = []
        else:
            max_prompt_length = self.max_length - len(response_ids)
            prompt_ids = prompt_ids[-max_prompt_length:]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids.copy()

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _left_pad_1d(sequences: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    max_length = max(seq.size(0) for seq in sequences)
    padded = sequences[0].new_full((len(sequences), max_length), padding_value)

    for row, seq in enumerate(sequences):
        padded[row, -seq.size(0) :] = seq

    return padded


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Left-pad a batch and ignore padded labels with -100."""
    return {
        "input_ids": _left_pad_1d([item["input_ids"] for item in batch], 0),
        "attention_mask": _left_pad_1d([item["attention_mask"] for item in batch], 0),
        "labels": _left_pad_1d([item["labels"] for item in batch], -100),
    }


def load_dataset_json(path: str | Path) -> list[dict[str, Any]]:
    """Load a commonsense JSON file containing a list of samples."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"expected a list of samples in {path}")

    return data
