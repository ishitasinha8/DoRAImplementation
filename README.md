# DoRA: Weight-Decomposed Low-Rank Adaptation (Re-implementation)

## 1. Introduction

This repository is a **course project re-implementation** of **DoRA: Weight-Decomposed Low-Rank Adaptation** (Liu et al., 2024). The goal is to reproduce the paper’s core idea from scratch for a **Llama-style** causal LM: split each frozen pretrained weight into **magnitude** and **direction**, apply a **low-rank update to the direction**, and learn a **trainable magnitude vector**, aiming to match or beat **LoRA**-style PEFT with similar adapter parameter counts and **no extra inference cost after merging**.

## 2. Chosen result

We targeted the paper’s **main commonsense reasoning comparison**: **DoRA vs. LoRA** on eight benchmarks (**BoolQ, PIQA, Social IQa, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA**), reported as accuracy in **Table 1** of the paper (LLaMA / commonsense section). That table is the central evidence that **decomposing magnitude and direction** improves adaptation versus standard LoRA under comparable efficiency.

> **Figure (paper):** Table 1 — commonsense accuracies (LoRA vs DoRA across datasets).  
> **Equation (paper):** adapted weight \(W' = m \odot \frac{W + BA}{\|W + BA\|}\) (column-wise normalization as in the paper; our code uses the same **row-wise \(\ell_2\)** convention on weight rows for `nn.Linear`).

## 3. GitHub contents

Typical layout: **`dora.py`** (`DoRALinear`, Llama MHA integration), **`inject.py`** (adapter injection), **`train.py`** (fine-tuning loop, checkpoints), **`data.py`** (Alpaca-style prompts, response-only loss masking), **`evaluate.py`** (per-task accuracy; likelihood-based **multiple-choice** scoring preferred), plus small helpers (**`metrics_logger.py`**, **`training_checkpoint.py`**, tests). Place **commonsense JSON** data under a folder with per-task `test.json` files matching the evaluator’s expected names.

## 4. Re-implementation details

- **Approach:** Custom **`DoRALinear`** freezes base weights, trains **`lora_A` / `lora_B`** and **`magnitude`**; **`inject_dora(..., use_llama_mha=True)`** swaps Llama attention (and optionally **`mlp.up_proj` / `mlp.down_proj`**, not `gate_proj`) for DoRA-aligned modules, matching the paper’s LLaMA setup.  
- **Models:** **`meta-llama/Llama-3.1-8B`** and **`meta-llama/Llama-3.1-8B-Instruct`** via Hugging Face `transformers`.  
- **Data / metrics:** Same eight commonsense tasks as the paper; **accuracy** on held-out `test.json` per task. Training uses **Alpaca-style** instruction formatting and **labels masked to response tokens** (`-100` on prompt positions).  
- **Evaluation:** Switched from brittle **greedy generation + exact match** to **`method="choice"`** in `evaluate_task`: **average log-likelihood** of each answer span, with **label normalization** (`true/false`, `option1/2`, `solution1/2`, `A`–`E`, etc.).  
- **Challenges:** Most “failure to reproduce” traced to **eval/prompt mismatch**, not the core DoRA math; **instruction-tuned** base models aligned better with answer-format instructions than the raw base model.

## 5. Reproduction steps

1. **Environment:** Python 3.10+ recommended; **CUDA GPU** strongly recommended (8B weights in **bfloat16** need roughly **24GB+** VRAM for comfortable single-GPU fine-tuning; adjust batch size / gradient accumulation if using less).  
2. **Dependencies (example):**  
   `pip install torch transformers accelerate safetensors tqdm numpy`  
   (versions aligned with your CUDA build of PyTorch).  
3. **Hugging Face:** Accept the **Llama 3.1** license on the Hub and run `huggingface-cli login` (token with read access).  
4. **Data:** Prepare instruction JSON (fields: `instruction`, `input`, `output`, and compact `answer` where used) and per-task eval folders (e.g. `data/boolq/test.json`, …) consistent with `evaluate.TASK_FILES`.  
5. **Train (example):** from the repo root (where `train.py` lives):

```bash
python train.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_path ./data/train.json \
  --output_dir ./checkpoints \
  --r 16 --alpha 32 \
  --epochs 3 --lr 2e-4 --batch_size 4 \
  --max_length 256 \
  --dtype bfloat16
```

- **Attention-only ablation:** add `--no_adapt_mlp`.  
- **Resume:** `--resume_ckpt path/to/step_checkpoint.pt` (see script help).  
6. **Evaluate:** Load base model + inject DoRA + load adapter weights, then call `evaluate.run_all_tasks(model, tokenizer, task_dir="./data", method="choice")` from a short script or notebook, or use your existing evaluation driver.

## 6. Results / insights

Qualitatively we match the paper’s headline pattern: **DoRA improves over the same base model without DoRA** on the commonsense suite, with **macro-average** gains on the order of **~10 accuracy points** for both **base** and **Instruct** Llama-3.1-8B in our runs. **Absolute** numbers differ from the paper’s LoRA/DoRA table due to **model generation (3.1 vs paper LLaMA)**, **prompting**, and **likelihood-based eval**; **Instruct + DoRA** was closest to the paper’s reference on several tasks. Expect stable, interpretable **per-dataset accuracies** once **choice scoring** and **label normalization** are used—not reliable scores from unconstrained greedy decoding.

## 7. Conclusion

DoRA’s idea is compact, but **end-to-end reproducibility** depends heavily on **where adapters are injected**, **training prompts vs. eval prompts**, and **how multiple-choice tasks are scored**. The main lesson: **verify the evaluation pipeline first** when PEFT results look broken.

## 8. References

1. S.-Y. Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, arXiv:2402.09353, 2024.  
2. E. J. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.  
3. Meta AI, *Llama 3.1 Model Card*, 2024.  
4. Wolf et al., *HuggingFace Transformers*, EMNLP Demo 2020.  
5. HuggingFace, *PEFT* library (reference only; this repo’s core path is custom DoRA).  
6. Dataset papers: BoolQ (NAACL 2019), PIQA (AAAI 2020), Social IQa (EMNLP 2019), HellaSwag (ACL 2019), WinoGrande (AAAI 2020), ARC (arXiv:1803.05457), OpenBookQA (EMNLP 2018).

## 9. Acknowledgements

This work was carried out as part of **Cornell CS 4/5782 (Spring 2026)** — thanks to the course staff and peers for feedback and for the reproducibility-focused project framing.
