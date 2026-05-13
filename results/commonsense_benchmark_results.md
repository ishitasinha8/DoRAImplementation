# Commonsense benchmark results (Llama 3.1 8B)

Performance comparison across **Llama-3.1-8B** base, **DoRA** on base, **Llama-3.1-8B-Instruct**, and **Instruct + DoRA**. Values are **accuracy (%)**.

| Task | Llama-3.1-8B | Llama-3.1-8B (DoRA) | Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct (DoRA) |
|------|-------------:|--------------------:|----------------------:|-----------------------------:|
| boolq | 44.37 | 51.83 | 56.15 | 72.26 |
| piqa | 66.76 | 74.91 | 81.94 | 86.94 |
| social_i_qa | 42.22 | 56.42 | 68.63 | 78.15 |
| hellaswag | 28.73 | 45.78 | 69.00 | 83.84 |
| winogrande | 52.25 | 58.34 | 60.69 | 82.48 |
| ARC-Easy | 69.61 | 78.65 | 92.21 | 93.39 |
| ARC-Challenge | 53.50 | 63.82 | 78.84 | 83.19 |
| openbookqa | 45.20 | 57.13 | 75.60 | 84.00 |
| **Macro Avg** | **50.33** | **60.86** | **72.88** | **83.03** |

*Caption (from report): Performance comparison across Llama-3.1-8B base, DoRA, instruct, and instruct DoRA variants.*

Performance comparison across **LoRA (paper)**, **DoRA (paper)** and **DoRA (Our Implementation)**. Values are **accuracy (%)**.

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

*Graphed in report*
