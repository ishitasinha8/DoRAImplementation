
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import format_prompt, load_dataset_json

TASK_FILES = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
TRUE_FALSE = {"TRUE", "FALSE"}
INDEX_TO_LETTER = {str(i): chr(ord("A") + i - 1) for i in range(1, 6)}

def normalize_answer(text: str) -> str:
    answer = str(text).strip().upper()
    answer = re.sub(r"^(ANSWER|RESPONSE)\s*:\s*", "", answer).strip()
    answer = re.sub(r"^(THE\s+)?CORRECT\s+ANSWER\s+IS\s+", "", answer).strip()
    answer = answer.strip(" .,:;\"'`()[]{}")
    indexed = re.search(r"\b(OPTION|ANSWER|ENDING|SOLUTION)\s*([1-5])\b", answer)
    if indexed:
        return INDEX_TO_LETTER[indexed.group(2)]
    if answer in {"YES", "Y"}:
        return "TRUE"
    if answer in {"NO", "N"}:
        return "FALSE"
    if answer.startswith("TRUE"):
        return "TRUE"
    if answer.startswith("FALSE"):
        return "FALSE"
    match = re.search(r"\b([A-E])\b", answer)
    return match.group(1) if match else answer

def _gold_for_eval(sample: dict[str, Any]) -> str:
    return str(sample["answer"]) if str(sample.get("answer", "")).strip() else str(sample["output"])

def _infer_prefix(sample: dict[str, Any]) -> str | None:
    text = " ".join(str(sample.get(k, "")) for k in ("instruction", "input", "output", "answer"))
    for prefix in ("option", "answer", "ending", "solution"):
        if re.search(rf"\b{prefix}\s*[1-5]\b", text, re.IGNORECASE):
            return prefix
    return None

def _num_choices(sample: dict[str, Any], prefix: str | None) -> int:
    text = " ".join(str(sample.get(k, "")) for k in ("instruction", "input", "output", "answer"))
    prefixes = [prefix] if prefix else ["option", "answer", "ending", "solution"]
    max_seen = 0
    for item_prefix in prefixes:
        for match in re.finditer(rf"\b{item_prefix}\s*([1-5])\b", text, re.IGNORECASE):
            max_seen = max(max_seen, int(match.group(1)))
    return max_seen or 5

def _candidate_groups(sample: dict[str, Any]) -> dict[str, list[str]]:
    gold_norm = normalize_answer(_gold_for_eval(sample))
    if gold_norm in TRUE_FALSE:
        labels = ["true", "false"]
        return {normalize_answer(label): [label, f"the correct answer is {label}"] for label in labels}
    prefix = _infer_prefix(sample)
    if prefix:
        labels = [f"{prefix}{idx}" for idx in range(1, _num_choices(sample, prefix) + 1)]
    else:
        labels = [chr(ord("A") + idx) for idx in range(_num_choices(sample, None))]
    return {normalize_answer(label): [label, f"the correct answer is {label}"] for label in labels}

def _build_entries(samples, tokenizer, max_prompt_tokens):
    entries = []
    for sample_idx, sample in enumerate(samples):
        prompt_ids = tokenizer(format_prompt(sample), add_special_tokens=True, truncation=False, return_attention_mask=False)["input_ids"]
        if max_prompt_tokens is not None:
            prompt_ids = prompt_ids[-max_prompt_tokens:]
        for label, variants in _candidate_groups(sample).items():
            for variant in variants:
                cand_ids = tokenizer(variant, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
                if cand_ids:
                    entries.append({"sample_idx": sample_idx, "label": label, "input_ids": prompt_ids + cand_ids, "prompt_len": len(prompt_ids), "candidate_len": len(cand_ids)})
    return entries

@torch.no_grad()
def _score_entry_batch(model, entries, pad_token_id):
    device = next(model.parameters()).device
    max_len = max(len(e["input_ids"]) for e in entries)
    input_ids = torch.full((len(entries), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, entry in enumerate(entries):
        ids = torch.tensor(entry["input_ids"], dtype=torch.long, device=device)
        input_ids[row, -ids.numel():] = ids
        attention_mask[row, -ids.numel():] = 1
    logprobs = F.log_softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, dim=-1)
    scored = []
    for row, entry in enumerate(entries):
        pad_len = max_len - len(entry["input_ids"])
        start = pad_len + entry["prompt_len"]
        end = start + entry["candidate_len"]
        target = input_ids[row, start:end]
        token_lp = logprobs[row, start - 1:end - 1, :]
        score = token_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item()
        scored.append((entry["sample_idx"], entry["label"], score))
    return scored

@torch.no_grad()
def evaluate_task(model, tokenizer, samples, batch_size=32, max_prompt_tokens=512, show_progress=True, progress_desc="eval"):
    if not samples:
        raise ValueError("evaluate_task received an empty sample list")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    old_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        entries = _build_entries(samples, tokenizer, max_prompt_tokens)
        best = [{} for _ in samples]
        iterator = range(0, len(entries), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=progress_desc, leave=False)
        for start in iterator:
            batch = entries[start:start + batch_size]
            for sample_idx, label, score in _score_entry_batch(model, batch, tokenizer.pad_token_id):
                best[sample_idx][label] = max(best[sample_idx].get(label, float("-inf")), score)
        correct = 0
        for sample, scores in zip(samples, best):
            pred = max(scores, key=scores.get) if scores else ""
            correct += int(pred == normalize_answer(_gold_for_eval(sample)))
        return correct / len(samples)
    finally:
        tokenizer.padding_side = old_side

def run_all_tasks(model, tokenizer, task_dir: str | Path, batch_size=32, max_prompt_tokens=512, show_progress=True):
    task_dir = Path(task_dir)
    results = {}
    task_iter = tqdm(TASK_FILES, desc="tasks") if show_progress else TASK_FILES
    for task in task_iter:
        samples = load_dataset_json(task_dir / task / "test.json")
        acc = evaluate_task(model, tokenizer, samples, batch_size=batch_size, max_prompt_tokens=max_prompt_tokens, show_progress=show_progress, progress_desc=task)
        results[task] = round(acc * 100, 2)
        print(f"{task:20s}: {results[task]:6.2f}%")
    print(f"{'Average':20s}: {sum(results.values()) / len(results):6.2f}%")
    return results
