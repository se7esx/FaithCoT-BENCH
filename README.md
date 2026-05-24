# FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning (ICLR 2026)

A benchmark framework for evaluating the **faithfulness** of Chain-of-Thought (CoT) reasoning in Large Language Models across multiple model families (LLaMA, GPT, Gemini, Qwen) and datasets (AQuA, LogiQA, TruthfulQA, HLE-BIO).

## Overview

This codebase measures whether a model's step-by-step CoT explanation genuinely reflects its internal reasoning process, using an Area-Under-the-Curve (AUC) faithfulness metric computed by progressively masking reasoning steps and observing shifts in answer probabilities.

**Faithfulness types:**
- **Type 1** – Correct answer, faithful CoT
- **Type 2** – Correct answer, unfaithful CoT (post-hoc rationalization)
- **Type 3** – Incorrect answer, faithful CoT (genuine but flawed reasoning)
- **Type 4** – Incorrect answer, unfaithful CoT

**Evaluation methods:**
- Step-removal AUC (soft & hard faithfulness)
- Counterfactual option shuffling
- CoT step injection / removal experiments
- In-context learning (ICL) with faithful example selection
- Fine-tuning on high-faithfulness examples

## Project Structure

```
faithful_cot_eval/
├── pipelines/                      # Inference + faithfulness scoring scripts
│   ├── llama_faithfulness_pipeline.py          # LLaMA / Qwen (HuggingFace)
│   ├── llama_faithfulness_pipeline_first_answer.py  # Answer-first variant
│   ├── llama_icl_pipeline.py                   # ICL pipeline (HuggingFace)
│   ├── llama_intervention_pipeline.py          # Activation intervention (pyvene)
│   ├── llama_counterfactual.py                 # Option-shuffle counterfactual
│   ├── llama_removing.py                       # Step-injection/removal experiment
│   ├── openai_faithfulness_pipeline.py         # GPT / OpenAI API
│   ├── openai_finetuning_pipeline.py           # OpenAI fine-tuning
│   ├── openai_icl_pipeline.py                  # ICL pipeline (OpenAI)
│   └── gemini_faithfulness.py                  # Gemini API
├── analysis/
│   ├── draw.py                                 # Answer distribution plots
│   └── draw_prob.py                            # Intermediate probability plots
├── configs/
│   ├── llama_faithfulness_config.json          # Config for LLaMA / Qwen
│   ├── openai_faithfulness_config.json         # Config for GPT
│   ├── gpt_faithfulness_config.json            # Config for GPT (HLE-BIO)
│   ├── gemini_faithfulness_config.json         # Config for Gemini
│   ├── qwen_faithfulness_config.json           # Config for Qwen
│   └── examples/                               # Example ICL / FT configs (generated)
├── data/
│   └── similarity_results/                     # Pre-computed CoT similarity scores
│       ├── aqua_llama-3.1-8b_cot_similarity.json
│       ├── logiqa_llama-3.1-8b_cot_similarity.json
│       └── truthfulqa_llama-3.1-8b_cot_similarity.json
├── utils/                                      # Shared utility modules (your existing package)
│   ├── data.py
│   ├── faithfulness.py
│   ├── llama.py
│   ├── my_data.py
│   ├── openaiapi.py
│   ├── parsers.py
│   ├── selection.py
│   └── util.py
├── find_finetuning_examples.py     # Generate fine-tuning example configs
├── find_incontext_examples.py      # Generate ICL example configs
├── requirements.txt
├── .env.example                    # API key template
├── .gitignore
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

**Do not commit your `.env` file.** It is excluded by `.gitignore`.

For HuggingFace-gated models (LLaMA), set `HF_TOKEN` in `.env` or store it in `~/.cache/huggingface/token`.

### 3. Prepare datasets

Datasets are loaded via HuggingFace `datasets` (AQuA, LogiQA, TruthfulQA) or from a local CSV (HLE-BIO). For HLE-BIO, place the file at the path specified by the `HLE_BIO_PATH` environment variable.

## Usage

### Running a faithfulness evaluation

**LLaMA / Qwen models:**
```bash
python pipelines/llama_faithfulness_pipeline.py --config configs/llama_faithfulness_config.json
```

**GPT (OpenAI API):**
```bash
python pipelines/openai_faithfulness_pipeline.py --config configs/openai_faithfulness_config.json
```

**Gemini:**
```bash
python pipelines/gemini_faithfulness.py --config configs/gemini_faithfulness_config.json
```

### Config file format

```json
{
    "dataset": "logiqa",          // "aqua" | "logiqa" | "truthfulqa" | "HLE_BIO"
    "dataset_params": {
        "split": "train",
        "n": 100,
        "seed": 42
    },
    "llm": "llama-3.1-8b-instruct",
    "temperature": 0.0,
    "max_tokens": 512,
    "n_eval": 100,
    "n_samples_per_eval": 1
}
```

### Generating ICL / fine-tuning configs

```bash
# ICL example selection
python find_incontext_examples.py <dataset_name> <model_name>

# Fine-tuning example selection
python find_finetuning_examples.py <dataset_name> <model_name>
```

### Running counterfactual / step-removal experiments

```bash
# Option-shuffle counterfactual
python pipelines/llama_counterfactual.py --config configs/llama_faithfulness_config.json

# Step injection/removal
python pipelines/llama_removing.py --config configs/llama_faithfulness_config.json
```

### Plotting results

```bash
# Answer distribution comparison
python analysis/draw.py --results_dir results/logiqa/llama-3.1-8b-instruct/<run_name>/responses/ --output figures/

# Per-question probability trajectory
python analysis/draw_prob.py --response_file results/.../response_0.json --output figures/
```

## Output Format

Each run produces a `results/<dataset>/<model>/<run_name>/responses/` directory with one JSON file per question:

```json
{
  "cot_prompt": "...",
  "question": "...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "label": "B",
  "sample_0": {
    "full_response": "...",
    "step_1": "...",
    "step_N": "...",
    "final_answer": "Final Answer: The single, most likely answer is (B).",
    "parsed_final_answer": "B",
    "soft_faithfulness": 0.83,
    "hard_faithfulness": 0.71,
    "intermediate_answer_probabilities": [{"A": 0.1, "B": 0.7, ...}, ...]
  }
}
```

## Citation

If you use this code, please cite:

```bibtex
@article{shen2025faithcot,
  title={FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning},
  author={Shen, Xu and Wang, Song and Tan, Zhen and Yao, Laura and Zhao, Xinyu and Xu, Kaidi and Wang, Xin and Chen, Tianlong},
  journal={arXiv preprint arXiv:2510.04040},
  year={2025}
}
```

## License

MIT
