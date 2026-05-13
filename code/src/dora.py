
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ADAPTER_KEYWORDS = ("lora_A", "lora_B", "magnitude")

class DoRALinear(nn.Module):
    """Custom DoRA adapter around a frozen nn.Linear layer.

    W' = m * normalize(W + scale * B @ A), with row-wise normalization.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("DoRALinear expects nn.Linear")
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.in_features = base.in_features
        self.out_features = base.out_features

        for p in self.base.parameters():
            p.requires_grad = False

        dtype = base.weight.dtype
        device = base.weight.device
        a = torch.empty((self.r, self.in_features), dtype=torch.float32, device=device)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(a.to(dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.r), dtype=dtype, device=device))
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1).clone().to(dtype=dtype, device=device))

    def adapted_weight(self):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        weight = self.base.weight + delta.to(dtype=self.base.weight.dtype)
        direction = F.normalize(weight.float(), p=2, dim=1).to(dtype=weight.dtype)
        return direction * self.magnitude[:, None]

    def forward(self, x):
        return F.linear(x, self.adapted_weight(), self.base.bias)

def dora_param_count(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0.0,
    }

def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if any(key in k for key in ADAPTER_KEYWORDS)
    }
