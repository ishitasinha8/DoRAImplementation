# DoRA: Weight-Decomposed Low-Rank Adaptation (Re-implementation)

## 1. Introduction

This repository is our CS 5782 course project : a from-scratch re-implementation and experimental study of a chosen PEFT paper, packaged so others can inspect the code, training flow, and evaluation we used.

We chose [**DoRA: Weight-Decomposed Low-Rank Adaptation**](https://arxiv.org/abs/2402.09353) (Liu et al., 2024). Its main idea is to decompose each frozen pretrained weight into magnitude and direction, apply low-rank adaptation to the direction while learning a separate magnitude vector, aiming to match or beat LoRA-style efficiency while moving closer to full fine-tuning quality, with no extra inference cost after merging adapters into the base weights.

## 2. Chosen Result

We targeted the paper’s primary quantitative result on commonsense reasoning - under the shared multi-task fine-tuning setup, DoRA should improve over LoRA on the same LLaMA-family while keeping trainable parameters and inference-time behavior in line with LoRA adapters. That result is the main empirical support for the paper’s claim that decoupling magnitude and directional adaptation increases learning capacity without adding inference overhead after merge.

The original paper reports this comparison in Table 1 (eight datasets: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA; averaged accuracy and parameter counts across PEFT methods including LoRA and DoRA).

## 3. GitHub Contents

```
DoRAImplementation/
├── README.md                         
├── code/                             # runnable code
│   ├── DoRA_run_experiments.ipynb    # notebook driver for experiments
│   └── src/                          # core Python modules
│       ├── data.py                   # prompts, dataset loading, label handling
│       ├── dora.py                   # DoRALinear and Llama MHA wiring
│       ├── evaluate.py               # per-task accuracy / MC scoring
│       ├── inject.py                 # swap linear layers for DoRA adapters
│       └── train.py                  # fine-tuning loop and checkpoints
├── data/                             # training JSON and eval splits
│   ├── data/
│   │   └── train.json                # combined commonsense fine-tuning split
│   └── eval_data/                    # subfolder per benchmark task (held-out test.json)
│       ├── ARC-Challenge/            
│       ├── ARC-Easy/                 
│       ├── boolq/                    
│       ├── hellaswag/                
│       ├── openbookqa/               
│       ├── piqa/                     
│       ├── social_i_qa/              
│       └── winogrande/               
├── poster/                           # poster 
├── report/                           # report
└── results/                          # tables, plots, and run logs
```

## 4. Re-implementation Details

- **Approach**: custom `DoRALinear` on a Llama-style causal LM; frozen backbone; train LoRA + magnitude; merge for inference like LoRA.
- **Injection**: `q_proj`, `k_proj`, `v_proj`, `up_proj`, `down_proj`; optional attention-only run without MLP adapters.
- **Models / tools**: `meta-llama/Llama-3.1-8B` and `Llama-3.1-8B-Instruct` with Hugging Face `transformers` and PyTorch (GPU); tuned rank, LR, batch size, epochs, max length.
- **Training data**: `commonsense_170k.json` merged commonsense split over BoolQ, PIQA, Social IQa, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA; Alpaca-style prompts; response-only loss (`-100` on prompts).
- **Evaluation**: per-task accuracy and macro average; switched from greedy exact-match to likelihood-based multiple-choice with label normalization across datasets.
- **Challenges vs paper**: brittle scores were mostly eval/prompt issues; instruct helped formatting; Llama 3.1 + our scorer means absolute numbers differ from the paper even when DoRA-over-base trends match.

## 5. Reproduction Steps

- Open `code/DoRA_run_experiments.ipynb` in Google Colab (upload from GitHub or `File > Open notebook > GitHub`) and select a GPU runtime (`Runtime > Change runtime type > GPU`, ideally A100 / L4 high-RAM).
- Accept the Llama 3.1 license on Hugging Face, then authenticate in the notebook so `meta-llama/Llama-3.1-8B` / `Llama-3.1-8B-Instruct` can be downloaded.
- Set `ROOT_DIR` to Drive mount and get the code and data into the runtime from `%cd DoRAImplementation/code/src` point paths there. Training data goes at `{ROOT_DIR}/data/train.json`; per-task eval files at `{ROOT_DIR}/eval_data/<task>/test.json`.
- Run the training cells in the notebook (they call `inject_dora` + the `train.py` loop with defaults `--r 32 --alpha 64 --epochs 3 --lr 1e-5 --batch_size 16 --max_length 256 --dtype bfloat16`). Save adapter checkpoints to Drive so they survive disconnects.
- Run the evaluation cells next: they load the base model, inject DoRA the same way, load the adapter weights, and call `run_all_tasks` over evaluation data to print per-task accuracy and the macro average.
- Compute: a single Colab GPU with enough VRAM for 8B weights in `bfloat16` (A100 40 or 80 GB is comfortable; on L4 24GB / T4 16 GB reduce `batch_size` and raise `grad_accum_steps` in the notebook’s training config).

## 6. Results / Insights

Per-task accuracy (%) on the eight commonsense benchmarks, base vs DoRA for both Llama-3.1-8B and Llama-3.1-8B-Instruct:

| Task          | Llama-3.1-8B | + DoRA | Llama-3.1-8B-Instruct | + DoRA |
| ------------- | ------------ | ------ | --------------------- | ------ |
| BoolQ         | 44.37        | 51.83  | 56.15                 | 72.26  |
| PIQA          | 66.76        | 74.91  | 81.94                 | 86.94  |
| Social IQa    | 42.22        | 56.42  | 68.63                 | 78.15  |
| HellaSwag     | 28.73        | 45.78  | 69.00                 | 83.84  |
| WinoGrande    | 52.25        | 58.34  | 60.69                 | 82.48  |
| ARC-Easy      | 69.61        | 78.65  | 92.21                 | 93.39  |
| ARC-Challenge | 53.50        | 63.82  | 78.84                 | 83.19  |
| OpenBookQA    | 45.20        | 57.13  | 75.60                 | 84.00  |
| Macro Avg     | 50.33        | 60.86  | 72.88                 | 83.03  |

- **Headline trend matches the paper**: DoRA improves every base model on every task; macro average climbs +10.53 on Llama-3.1-8B and +10.15 on Llama-3.1-8B-Instruct.
- **Biggest gains on harder / format-sensitive tasks**: HellaSwag (+17.0 base / +14.8 instruct), Social IQa (+14.2 / +9.5), WinoGrande on instruct (+21.8).

Three-way comparison against the paper’s LLaMA3-8B numbers (LoRA and DoRA from the paper’s Table 1) and our Llama-3.1-8B-Instruct + DoRA run:

| Task          | LoRA (paper) | DoRA (paper) | Our DoRA |
| ------------- | ------------ | ------------ | -------- |
| BoolQ         | 70.80        | 74.60        | 72.26    |
| PIQA          | 85.20        | 89.30        | 86.94    |
| Social IQa    | 79.90        | 79.90        | 78.15    |
| HellaSwag     | 91.70        | 95.50        | 83.84    |
| WinoGrande    | 84.30        | 85.60        | 82.48    |
| ARC-Easy      | 84.20        | 90.50        | 93.39    |
| ARC-Challenge | 71.20        | 80.40        | 83.19    |
| OpenBookQA    | 79.00        | 85.80        | 84.00    |
| Macro Avg     | 80.79        | 85.20        | 83.03    |

## 7. Conclusion

Our from-scratch DoRA implementation improved every base model on the eight commonsense tasks for both Llama-3.1-8B and Llama-3.1-8B-Instruct, and reached numbers close to the paper’s LoRA reference on the Instruct backbone.

Most of the difficulty was in the evaluation and prompting pipeline rather than the DoRA module itself; with more time we would add a matched LoRA baseline, run rank / learning-rate / target-module ablations, and compare against the PEFT library’s DoRA.

## 8. References

1. S.-Y. Liu et al., *[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)*, arXiv:2402.09353, 2024.
2. E. J. Hu et al., *[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)*, ICLR 2022.
3. Meta AI, *[Llama 3.1 Model Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)*, 2024.
4. Wolf et al., *[HuggingFace Transformers](https://arxiv.org/abs/1910.03771)*, EMNLP Demo 2020.
5. HuggingFace, *[PEFT](https://github.com/huggingface/peft)* library (reference only; this repo’s core path is custom DoRA).
6. Dataset papers: [BoolQ](https://arxiv.org/abs/1905.10044) (NAACL 2019), [PIQA](https://arxiv.org/abs/1911.11641) (AAAI 2020), [Social IQa](https://arxiv.org/abs/1904.09728) (EMNLP 2019), [HellaSwag](https://arxiv.org/abs/1905.07830) (ACL 2019), [WinoGrande](https://arxiv.org/abs/1907.10641) (AAAI 2020), [ARC](https://arxiv.org/abs/1803.05457) (arXiv:1803.05457), [OpenBookQA](https://arxiv.org/abs/1809.02789) (EMNLP 2018).

## 9. Acknowledgements

This work was carried out as part of **Cornell CS 5782 (Spring 2026)**; thanks to the course staff and peers for feedback and for the reproducibility-focused project framing.
