"""
Step injection / removal faithfulness experiment for LLaMA / Qwen.

Prepends counterfactual reasoning steps from a pre-generated file and
re-queries the model to check whether the injected (wrong) steps cause
the model to change its answer. If the answer is unchanged, the CoT is
unfaithful (the model ignores its own stated reasoning).

Cohen's Kappa is computed against the `faithful_type` ground-truth labels.

Usage:
    python pipelines/llama_removing.py \\
        --config configs/llama_faithfulness_config.json \\
        --dataset truthfulqa \\
        --model llama-3.1-8b-instruct \\
        --counterfact_dir results_adding_mistakes/truthfulqa/llama-3.1-8b-instruct \\
        --results_dir results/truthfulqa/llama-3.1-8b-instruct/train_n_100_seed_42_temp_0.0_maxtokens_512
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from pipelines.prompts import COT_PROMPT, FINAL_ANSWER_STR, PREFIX
from utils.data import load_data_subset
from utils.faithfulness import construct_response_dict
from utils.llama import set_up_inference_pipeline
from utils.parsers import parse_mcq_answer
from utils.util import (
    bold,
    get_multiple_choice_question,
    load_config,
    print_gpu_memory,
)

SKIP_INDICES: dict[tuple[str, str], list[int]] = {
    ("logiqa",     "llama-3.1-8b-instruct"):  [1, 3, 4, 9, 11, 23, 26, 43, 45, 54, 64, 66, 96],
    ("truthfulqa", "llama-3.1-8b-instruct"):  [22, 37, 40, 58, 73, 75, 85, 89, 98, 99],
}


def cohen_kappa(confusion: np.ndarray, k: int) -> float:
    mat = np.asmatrix(confusion)
    total = np.sum(mat)
    p0 = sum(mat[i, i] for i in range(k)) / total
    x_sum = np.sum(mat, axis=1)
    y_sum = np.sum(mat, axis=0)
    pe = float(y_sum * x_sum) / total ** 2
    return float((p0 - pe) / (1 - pe))


def main(
    config_path: str,
    dataset: str,
    model_name: str,
    counterfact_dir: str,
    results_dir: str,
) -> None:
    torch.cuda.empty_cache()
    print_gpu_memory()

    config = load_config(config_path)
    config["dataset"] = dataset
    config["llm"] = model_name

    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    n_samples_per_eval = config["n_samples_per_eval"]
    n_eval = config["n_eval"]

    if dataset == "logiqa":
        from utils.my_data import load_data_subset as load_logiqa
        df = load_logiqa(config)
    else:
        df = load_data_subset(config)

    save_dir = os.path.join(results_dir, "results_adding_mistakes")
    os.makedirs(save_dir, exist_ok=True)

    model, tokenizer, inference_pipeline = set_up_inference_pipeline(model_name)

    skip = set(SKIP_INDICES.get((dataset, model_name), []))
    total_num = 0
    original_num = 0

    for i in range(n_eval):
        if i in skip:
            continue

        save_path = os.path.join(save_dir, f"response_{i}.json")
        if os.path.exists(save_path):
            print(f"Skipping question {i} (already exists).")
            continue

        # Load counterfactual data
        cf_path = os.path.join(counterfact_dir, f"response_{i}.json")
        if not os.path.exists(cf_path):
            print(f"No counterfactual file for question {i}, skipping.")
            continue

        with open(cf_path, "r", encoding="utf-8") as f:
            cf_data = json.load(f)

        if dataset == "logiqa":
            question = df["question"].iloc[i]
            options = [df[f"option_{k}"].iloc[i] for k in range(4)]
            label = df["label"].iloc[i]
        else:
            question = df["question"].iloc[i]
            options = df["options"].iloc[i]
            label = df["label"].iloc[i]

        # Build prompt with injected counterfactual steps
        base_prompt = COT_PROMPT + get_multiple_choice_question(question, options)
        combined_counterfact = " ".join(cf_data["counterfact"])
        prompt_text = base_prompt + combined_counterfact

        # Query the model
        total_num += 1
        do_sample = temperature > 0
        response_text = inference_pipeline(
            prompt_text, do_sample=do_sample, max_new_tokens=max_tokens,
            truncation=True, top_p=None, temperature=temperature,
        )[0]["generated_text"]

        parsed_answer = parse_mcq_answer(response_text, FINAL_ANSWER_STR)

        # Load original response to compare
        original_path = os.path.join(results_dir, "responses", f"response_{i}.json")
        with open(original_path, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        original_label = original_data["sample_0"]["parsed_final_answer"]

        # Keep only step/answer fields from the original sample to reduce file size
        trimmed_sample = {
            k: v for k, v in original_data["sample_0"].items()
            if k.startswith("step") or k in ("final_answer", "parsed_final_answer")
        }
        new_data = {**original_data, "sample_0": trimmed_sample}
        new_data["new_response"] = response_text
        new_data["counterfact"] = cf_data["counterfact"]
        new_data["c_parsed_answer"] = parsed_answer
        new_data["couter_unfaithful"] = int(original_label == parsed_answer)

        if original_label == parsed_answer:
            original_num += 1

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"\ntotal_num: {total_num}, unchanged_answer: {original_num}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    new_truth_label: list[int] = []
    faithfulness_pred: list[int] = []

    for i in range(n_eval):
        if i in skip:
            continue

        orig_path = os.path.join(results_dir, "responses", f"response_{i}.json")
        save_path = os.path.join(save_dir, f"response_{i}.json")

        if not os.path.exists(orig_path) or not os.path.exists(save_path):
            continue

        with open(orig_path, "r", encoding="utf-8") as f:
            orig = json.load(f)
        with open(save_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        ft = orig.get("faithful_type")
        if ft == 1:
            new_truth_label.append(0)
        elif ft == 2:
            new_truth_label.append(1)
        elif ft == 3:
            new_truth_label.append(0)
        elif ft == 4:
            new_truth_label.append(1)
        else:
            continue

        faithfulness_pred.append(result["couter_unfaithful"])

    cm = confusion_matrix(new_truth_label, faithfulness_pred)
    kappa = cohen_kappa(np.array(cm), 2)
    acc = accuracy_score(new_truth_label, faithfulness_pred)
    f1 = f1_score(new_truth_label, faithfulness_pred)
    print(f"\n[{dataset} / {model_name}]")
    print(cm)
    print(f"Cohen's Kappa: {kappa:.3f}  Accuracy: {acc:.3f}  F1: {f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step injection/removal faithfulness experiment.")
    parser.add_argument("--config", required=True, help="Path to base JSON config file.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. truthfulqa).")
    parser.add_argument("--model", required=True, dest="model_name", help="Model name.")
    parser.add_argument(
        "--counterfact_dir", required=True,
        help="Directory containing counterfactual response_{i}.json files."
    )
    parser.add_argument(
        "--results_dir", required=True,
        help="Directory containing original responses/ and where output is saved."
    )
    args = parser.parse_args()
    main(args.config, args.dataset, args.model_name, args.counterfact_dir, args.results_dir)
