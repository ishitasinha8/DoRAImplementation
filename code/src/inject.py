
from typing import List

import torch.nn as nn

from dora import DoRALinear, dora_param_count

ADAPTER_KEYWORDS = ("lora_A", "lora_B", "magnitude")

def _name_matches(name: str, target_names: List[str]) -> bool:
    return any(name == pattern or name.endswith(f".{pattern}") for pattern in target_names)

def _set_submodule(model: nn.Module, key: str, new_module: nn.Module):
    parent = model
    parts = key.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)

def inject_dora(
    model: nn.Module,
    target_names: List[str],
    r: int = 16,
    lora_alpha: float = 32.0,
    dropout: float = 0.0,
    use_llama_mha: bool = False,
    adapt_mlp: bool = True,
) -> nn.Module:
    # Keep the old function signature. Instead of replacing full attention,
    # this clean version wraps the target Linear projections directly.
    if use_llama_mha:
        target_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if adapt_mlp:
            target_names += ["up_proj", "down_proj"]

    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _name_matches(name, target_names):
            _set_submodule(model, name, DoRALinear(module, r=r, alpha=lora_alpha))
            replaced.append(name)
    if not replaced:
        raise ValueError(f"No nn.Linear modules matched {target_names}")
    print(f"[inject_dora] Replaced {len(replaced)} layers")
    for name in replaced[:20]:
        print("  -", name)
    if len(replaced) > 20:
        print(f"  ... {len(replaced) - 20} more")
    return model

def get_trainable_params(model: nn.Module):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ADAPTER_KEYWORDS)]

def print_trainable_params(model: nn.Module):
    counts = dora_param_count(model)
    print(
        f"Trainable params: {counts['trainable']:,} / {counts['total']:,} "
        f"({counts['trainable_pct']}%) | Frozen: {counts['frozen']:,}"
    )
