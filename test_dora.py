
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/content")
from dora import DoRALinear, DoRALlamaMHA, apply_dora_to_llama, dora_param_count

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Running on {DEVICE} ({DTYPE})\n")

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def test_trainable_params():
    base  = nn.Linear(64, 128, bias=True)
    layer = DoRALinear(base, r=4, alpha=8.0)

    frozen    = [n for n, p in layer.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in layer.named_parameters() if p.requires_grad]

    ok = (set(frozen) == {"base.weight", "base.bias"}) and (set(trainable) == {"lora_A", "lora_B", "magnitude"})
    print(f"TEST 1 — Trainable/frozen split       {PASS if ok else FAIL}")
    if not ok:
        print(f"  frozen={frozen}  trainable={trainable}")

test_trainable_params()


def test_identity_at_init():
    torch.manual_seed(0)
    base     = nn.Linear(64, 128, bias=False)
    layer    = DoRALinear(base, r=4, alpha=8.0).to(DTYPE)
    x        = torch.randn(2, 10, 64, dtype=DTYPE)
    out_dora = layer(x)
    out_base = F.linear(x, base.weight.to(DTYPE))
    ok       = torch.allclose(out_dora, out_base, atol=1e-4)
    print(f"TEST 2 — Identity at init             {PASS if ok else FAIL}")
    if not ok:
        print(f"  max_diff={(out_dora - out_base).abs().max().item():.6f}")

test_identity_at_init()


def test_magnitude_init():
    base          = nn.Linear(64, 128, bias=False)
    layer         = DoRALinear(base, r=4, alpha=8.0)
    expected_vals = base.weight.detach().norm(p=2, dim=1, keepdim=True)
    ok = (layer.magnitude.shape == (128, 1)) and torch.allclose(layer.magnitude.data, expected_vals, atol=1e-6)
    print(f"TEST 3 — Magnitude init shape/value   {PASS if ok else FAIL}")

test_magnitude_init()


def test_output_shape():
    B, S, IN, OUT = 2, 7, 64, 128
    base  = nn.Linear(IN, OUT)
    layer = DoRALinear(base, r=8).to(DTYPE)
    x     = torch.randn(B, S, IN, dtype=DTYPE)
    out   = layer(x)
    ok    = (out.shape == (B, S, OUT))
    print(f"TEST 4 — Output shape {str(tuple(out.shape)):<20} {PASS if ok else FAIL}")
    
test_output_shape()


def test_gradients():
    base  = nn.Linear(32, 64)
    layer = DoRALinear(base, r=4).to(torch.float32)
    x     = torch.randn(2, 5, 32)
    layer(x).sum().backward()
    ok = (layer.lora_A.grad is not None and
          layer.lora_B.grad is not None and
          layer.magnitude.grad is not None and
          base.weight.grad is None)
    print(f"TEST 5 — Gradients (train/frozen)     {PASS if ok else FAIL}")

test_gradients()


def test_unit_direction():
    torch.manual_seed(42)
    base  = nn.Linear(64, 128, bias=False)
    layer = DoRALinear(base, r=4, alpha=8.0).to(torch.float32)
    with torch.no_grad():
        layer.lora_B.normal_(0, 0.01)
    W         = layer.base.weight
    W_adapted = W + (layer.lora_B @ layer.lora_A) * layer.scaling
    dir_norms = (W_adapted / W_adapted.norm(p=2, dim=1, keepdim=True)).norm(p=2, dim=1)
    ok        = torch.allclose(dir_norms, torch.ones_like(dir_norms), atol=1e-5)
    print(f"TEST 6 — Unit-direction per neuron    {PASS if ok else FAIL}")

test_unit_direction()


def test_llama_swap():
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=64,
            vocab_size=1000,
        )
        model  = LlamaForCausalLM(cfg)
        model  = apply_dora_to_llama(model, r=4, alpha=8.0)
        ok     = all(isinstance(layer.self_attn, DoRALlamaMHA) for layer in model.model.layers)
        counts = dora_param_count(model)
        print(f"TEST 7 — Llama attn layer swap        {PASS if ok else FAIL}")
        print(f"         trainable {counts['trainable']:,} / {counts['total']:,} params  ({counts['trainable_pct']}%)")
    except ImportError:
        print("TEST 7 — SKIPPED (transformers not installed)")

test_llama_swap()


def test_llama_forward():
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=64,
            vocab_size=1000,
        )
        model     = LlamaForCausalLM(cfg)
        model     = apply_dora_to_llama(model, r=4, alpha=8.0)
        model.eval()
        input_ids = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            out = model(input_ids)
        ok = out.logits.shape == (1, 16, 1000)
        print(f"TEST 8 — Llama forward pass shape     {PASS if ok else FAIL}  {tuple(out.logits.shape)}")
    except ImportError:
        print("TEST 8 — SKIPPED (transformers not installed)")

test_llama_forward()

print("\nAll tests complete.")
