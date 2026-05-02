import re
from typing import List, Optional

import torch.nn as nn

from dora import DoRALinear, apply_dora_to_llama, dora_param_count


def _name_matches(name: str, target_names: List[str]) -> bool:
    """Return True if the module name ends with any of the target patterns."""
    for pattern in target_names:
        if name == pattern or name.endswith(f".{pattern}"):
            return True
    return False


def _set_submodule(model: nn.Module, key: str, new_module: nn.Module):
    tokens = key.split(".")
    parent = model
    for tok in tokens[:-1]:
        parent = getattr(parent, tok)
    setattr(parent, tokens[-1], new_module)


def inject_dora(
    model: nn.Module,
    target_names: List[str],
    r: int = 16,
    lora_alpha: float = 32.0,
    dropout: float = 0.0,
    use_llama_mha: bool = False,
) -> nn.Module:
    """
    Replace matching nn.Linear layers with DoRALinear adapters.

    Parameters
    ----------
    model : nn.Module
        The pretrained model to modify in-place.
    target_names : list[str]
        Suffixes to match against module names (e.g. ["q_proj", "v_proj"]).
    r : int
        LoRA rank.
    lora_alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout rate applied after the adapted linear (unused by current
        DoRALinear — reserved for future extension).
    use_llama_mha : bool
        If True, use the full DoRALlamaMHA attention replacement instead
        of individual Linear swaps.  Recommended for Llama models.
    """
    if use_llama_mha:
        return apply_dora_to_llama(model, r=r, alpha=lora_alpha)

    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _name_matches(name, target_names):
            dora_layer = DoRALinear(module, r=r, alpha=lora_alpha)
            _set_submodule(model, name, dora_layer)
            replaced.append(name)

    if not replaced:
        raise ValueError(
            f"No nn.Linear modules matched target_names={target_names}. "
            "Check the model architecture and target names."
        )

    print(f"[inject_dora] Replaced {len(replaced)} layers:")
    for name in replaced:
        print(f"  - {name}")

    return model


ADAPTER_KEYWORDS = {"lora_A", "lora_B", "magnitude"}


def get_trainable_params(model: nn.Module) -> list:
    """Return (name, param) pairs for DoRA adapter parameters only."""
    return [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and any(kw in n for kw in ADAPTER_KEYWORDS)
    ]


def print_trainable_params(model: nn.Module) -> None:
    """Print count and percentage of trainable vs total parameters."""
    counts = dora_param_count(model)
    print(
        f"Trainable params: {counts['trainable']:,} / {counts['total']:,} "
        f"({counts['trainable_pct']}%) | Frozen: {counts['frozen']:,}"
    )
