
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from data import CommonsenseDataset, collate_fn, load_dataset_json
from dora import adapter_state_dict
from inject import get_trainable_params, inject_dora, print_trainable_params

DTYPE = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=list(DTYPE), default="bfloat16")
    p.add_argument("--no_adapt_mlp", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=DTYPE[args.dtype], device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad = False

    model = inject_dora(
        model,
        target_names=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=args.r,
        lora_alpha=args.alpha,
        use_llama_mha=True,
        adapt_mlp=not args.no_adapt_mlp,
    )
    print_trainable_params(model)

    samples = load_dataset_json(args.data_path)
    dataset = CommonsenseDataset(samples, tokenizer, max_length=args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate_fn)

    trainable = [p for _, p in get_trainable_params(model)]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(loader) // args.grad_accum_steps) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    meta = vars(args) | {"adapter_format": "custom_dora_pt", "target_modules": ["q/k/v/o", "up/down if adapt_mlp"]}
    Path(args.output_dir, "adapter_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    global_step = 0
    device = next(model.parameters()).device
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for micro_step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running += outputs.loss.item()
            if micro_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if micro_step % 10 == 0:
                pbar.set_postfix(loss=f"{outputs.loss.item():.4f}", avg=f"{running / micro_step:.4f}")

        ckpt = {"adapter_state_dict": adapter_state_dict(model), "epoch": epoch, "global_step": global_step, "config": meta}
        path = Path(args.output_dir) / f"adapter_epoch{epoch}.pt"
        torch.save(ckpt, path)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved custom DoRA adapter to {path}")

if __name__ == "__main__":
    main()
