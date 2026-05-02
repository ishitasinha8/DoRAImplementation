"""
DoRA fine-tuning script for Llama 3.1 8B on commonsense reasoning tasks.

Usage
-----
    python train.py \
        --model_name meta-llama/Llama-3.1-8B \
        --data_path  ./data/train.json \
        --output_dir ./checkpoints \
        --r 16 --alpha 32 \
        --epochs 3 --lr 2e-4 --batch_size 4
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from data import CommonsenseDataset, collate_fn, load_dataset_json
from inject import inject_dora, print_trainable_params, get_trainable_params
from evaluate import evaluate_task, load_dataset_json as load_eval_json


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DoRA fine-tuning for Llama")

    # model / data
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--data_path", type=str, required=True,
                    help="Path to training JSON (list of instruction samples)")
    p.add_argument("--eval_path", type=str, default=None,
                    help="Path to eval JSON for end-of-epoch accuracy check")
    p.add_argument("--output_dir", type=str, default="./checkpoints")

    # DoRA hyper-params
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--target_names", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"],
                    help="Linear layer name suffixes to replace with DoRA")

    # training hyper-params
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)

    # hardware
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device", type=str, default=None,
                    help="Force device (default: auto-detect)")

    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ADAPTER_KEYWORDS = {"lora_A", "lora_B", "magnitude"}


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_adapter_checkpoint(model, path: Path, epoch: int, step: int):
    """Save only DoRA adapter weights (lora_A, lora_B, magnitude)."""
    adapter_state = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if any(kw in k for kw in ADAPTER_KEYWORDS)
    }
    ckpt = {
        "adapter_state_dict": adapter_state,
        "epoch": epoch,
        "step": step,
    }
    path.mkdir(parents=True, exist_ok=True)
    save_path = path / f"adapter_epoch{epoch}_step{step}.pt"
    torch.save(ckpt, save_path)
    print(f"[checkpoint] Saved {len(adapter_state)} tensors → {save_path}")


def load_adapter_checkpoint(model, ckpt_path: str | Path):
    """Load DoRA adapter weights back into the model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(
        ckpt["adapter_state_dict"], strict=False
    )
    print(f"[checkpoint] Loaded adapter from {ckpt_path} "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = resolve_device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    print(f"Device: {device}  |  dtype: {dtype}")
    print(f"Model : {args.model_name}")
    print(f"Rank  : {args.r}  |  Alpha: {args.alpha}")

    # ── tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=device if device.type == "cuda" else None,
    )

    # ── inject DoRA adapters (uses Llama-specific MHA swap) ──────────────
    model = inject_dora(
        model,
        target_names=args.target_names,
        r=args.r,
        lora_alpha=args.alpha,
        use_llama_mha=True,
    )
    print_trainable_params(model)

    if device.type != "cuda":
        model.to(device)

    # ── dataset & dataloader ─────────────────────────────────────────────
    raw_samples = load_dataset_json(args.data_path)
    dataset = CommonsenseDataset(raw_samples, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── optimizer & scheduler ────────────────────────────────────────────
    trainable = [p for _, p in get_trainable_params(model)]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = (len(dataloader) // args.grad_accum_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nDataset  : {len(dataset):,} samples")
    print(f"Batches  : {len(dataloader):,} per epoch")
    print(f"Steps    : {total_steps:,} total ({args.epochs} epochs)")
    print()

    # ── training loop ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(dataloader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += outputs.loss.item()

            if step % args.log_interval == 0:
                avg = epoch_loss / step
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch} | step {step}/{len(dataloader)} | "
                    f"loss {outputs.loss.item():.4f} | avg {avg:.4f} | "
                    f"lr {lr_now:.2e} | {elapsed:.0f}s"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} done — avg loss: {avg_loss:.4f}  "
              f"({time.time() - t0:.0f}s)")

        save_adapter_checkpoint(model, output_dir, epoch, global_step)

        # ── optional end-of-epoch eval ───────────────────────────────────
        if args.eval_path:
            eval_samples = load_eval_json(args.eval_path)
            acc = evaluate_task(model, tokenizer, eval_samples, batch_size=1)
            print(f"  eval accuracy: {acc * 100:.2f}%")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
