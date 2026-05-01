import re
from pathlib import Path
from typing import Any

import torch

try:
    from .data import format_prompt, load_dataset_json
except ImportError:
    from data import format_prompt, load_dataset_json


TASK_FILES = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Easy",
    "ARC-Challenge",
    "openbookqa",
]

LETTER_CHOICES = {"A", "B", "C", "D", "E"}
TRUE_FALSE = {"TRUE", "FALSE"}


def normalize_answer(text: str) -> str:
    """Normalize generated text or gold output for exact answer matching."""
    answer = text.strip().upper()
    answer = re.sub(r"^(ANSWER|RESPONSE)\s*:\s*", "", answer).strip()
    answer = answer.strip(" .,:;\"'`()[]{}")

    if answer in {"YES", "Y"}:
        return "TRUE"
    if answer in {"NO", "N"}:
        return "FALSE"

    if answer.startswith("TRUE"):
        return "TRUE"
    if answer.startswith("FALSE"):
        return "FALSE"

    match = re.search(r"\b([A-E])\b", answer)
    if match:
        return match.group(1)

    return answer


def answers_match(prediction: str, gold: str) -> bool:
    """Compare normalized prediction and gold answers."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if gold_norm in LETTER_CHOICES:
        return pred_norm == gold_norm
    if gold_norm in TRUE_FALSE:
        return pred_norm == gold_norm

    return pred_norm == gold_norm


def _model_input_device(model) -> torch.device:
    if hasattr(model, "device"):
        device = model.device
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _batch_prompts(samples: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


@torch.no_grad()
def evaluate_task(
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    batch_size: int = 1,
    max_new_tokens: int = 4,
) -> float:
    """
    Greedy generation evaluation.

    Returns accuracy as a float in [0, 1].
    """
    if not samples:
        raise ValueError("evaluate_task received an empty sample list")

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    correct = 0
    device = _model_input_device(model)

    try:
        for batch in _batch_prompts(samples, batch_size):
            prompts = [format_prompt(sample) for sample in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            prompt_length = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
            )

            for prediction, sample in zip(decoded, batch):
                if answers_match(prediction, str(sample["output"])):
                    correct += 1
    finally:
        tokenizer.padding_side = original_padding_side

    return correct / len(samples)


def run_all_tasks(
    model,
    tokenizer,
    task_dir: str | Path,
    batch_size: int = 1,
) -> dict[str, float]:
    """Evaluate all 8 commonsense tasks and print a compact summary."""
    task_dir = Path(task_dir)
    results = {}

    for task in TASK_FILES:
        samples = load_dataset_json(task_dir / task / "test.json")
        acc = evaluate_task(model, tokenizer, samples, batch_size=batch_size)
        results[task] = round(acc * 100, 2)
        print(f"{task:20s}: {results[task]:6.2f}%")

    average = sum(results.values()) / len(results)
    print(f"{'Average':20s}: {average:6.2f}%")

    return results
