"""
Counterfactual option-shuffling experiment for local HuggingFace models.

For each question, the answer options are randomly shuffled and the model
is re-queried. A faithful model should change its answer when the correct
option moves position; an unfaithful model will answer based on position
bias rather than reasoning.

Cohen's Kappa is reported against the ground-truth faithfulness labels
stored in `faithful_type` fields of the original response files.

Usage:
    python pipelines/llama_counterfactual.py \\
        --config configs/llama_faithfulness_config.json \\
        --data_names logiqa truthfulqa \\
        --model_names llama-3.1-8b-instruct Qwen2.5-7B-Instruct
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from pipelines.prompts import COT_PROMPT, FINAL_ANSWER_STR, PREFIX
from utils.data import load_data_subset
from utils.faithfulness import (
    calculate_faithfulness_explanation_mcq,
    construct_response_dict,
    construct_samples_dict,
)
from utils.llama import get_option_probabilities, set_up_inference_pipeline
from utils.parsers import parse_cot_explanation, parse_mcq_answer
from utils.util import (
    bold,
    construct_save_dir,
    get_multiple_choice_question,
    load_config,
    print_gpu_memory,
    save_json,
)

# Indices to skip (questions with known annotation issues per dataset/model).
# These are maintained as a lookup rather than scattered if-chains.
SKIP_INDICES: dict[tuple[str, str], list[int]] = {
    ("logiqa",     "Qwen2.5-7B-Instruct"):       [1, 11, 4, 54],
    ("logiqa",     "llama-3.1-8b-instruct"):      [1, 3, 4, 9, 11, 23, 26, 43, 45, 54, 64, 66, 96],
    ("truthfulqa", "Qwen2.5-7B-Instruct"):        [3, 10, 71],
    ("truthfulqa", "llama-3.1-8b-instruct"):      [22, 37, 40, 58, 73, 75, 85, 89, 98, 99],
}

RESULTS_BASE = "results"


def cohen_kappa(confusion: np.ndarray, k: int) -> float:
    """Compute Cohen's Kappa from a k×k confusion matrix."""
    mat = np.asmatrix(confusion)
    total = np.sum(mat)
    p0 = sum(mat[i, i] for i in range(k)) / total
    x_sum = np.sum(mat, axis=1)
    y_sum = np.sum(mat, axis=0)
    pe = float(y_sum * x_sum) / total ** 2
    return float((p0 - pe) / (1 - pe))


def original_results_dir(data_name: str, model_name: str) -> str:
    return (
        f"{RESULTS_BASE}/{data_name}/{model_name}/"
        "train_n_100_seed_42_temp_0.0_maxtokens_512/responses"
    )


def load_original_response(data_name: str, model_name: str, i: int) -> dict:
    path = os.path.join(original_results_dir(data_name, model_name), f"response_{i}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(
    config: dict,
    data_name: str,
    model_name: str,
    model,
    tokenizer,
    inference_pipeline,
) -> None:
    """Run the counterfactual experiment for one dataset/model pair."""
    config["dataset"] = data_name
    config["llm"] = model_name

    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    n_samples_per_eval = config["n_samples_per_eval"]
    n_eval = config["n_eval"]

    if data_name == "logiqa":
        from utils.my_data import load_data_subset as load_logiqa
        df = load_logiqa(config)
    else:
        df = load_data_subset(config)

    save_dir = (
        f"{RESULTS_BASE}/{data_name}/{model_name}/"
        "train_n_100_seed_42_temp_0.0_maxtokens_512/responses_shuffle"
    )
    os.makedirs(save_dir, exist_ok=True)

    skip = set(SKIP_INDICES.get((data_name, model_name), []))
    model_kwargs = {"model": model, "tokenizer": tokenizer}
    get_probs_func = get_option_probabilities
    faithfulness_pred: list[int] = []

    for i in tqdm(range(n_eval), desc=f"{data_name}/{model_name}"):
        if i in skip:
            continue

        save_path = os.path.join(save_dir, f"response_{i}.json")

        if os.path.exists(save_path):
            # Already computed — just load and record prediction.
            orig = load_original_response(data_name, model_name, i)
            original_label = orig["sample_0"]["parsed_final_answer"]
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            parsed = data["sample_0"]["parsed_final_answer"]
            faithfulness_pred.append(int(parsed != original_label))
            continue

        if data_name == "logiqa":
            question = df["question"].iloc[i]
            options = [df[f"option_{k}"].iloc[i] for k in range(4)]
            label = df["label"].iloc[i]
        else:
            question = df["question"].iloc[i]
            options = list(df["options"].iloc[i])
            label = df["label"].iloc[i]

        random.shuffle(options)
        response_dict = construct_response_dict(
            COT_PROMPT, FINAL_ANSWER_STR, PREFIX, question, options, label
        )

        for j in range(n_samples_per_eval):
            prompt_text = COT_PROMPT + get_multiple_choice_question(question, options) + PREFIX
            do_sample = temperature > 0
            response_text = inference_pipeline(
                prompt_text, do_sample=do_sample, max_new_tokens=max_tokens,
                truncation=True, top_p=None, temperature=temperature,
            )[0]["generated_text"]

            cot_steps = parse_cot_explanation(PREFIX + response_text)
            fa_idx = response_text.find("Final Answer:")
            prompt_text += response_text[:fa_idx] + FINAL_ANSWER_STR
            cot_answer = response_text[fa_idx:]

            cot_answer_probs = get_option_probabilities(model, tokenizer, prompt_text, options)
            parsed_cot_answer = parse_mcq_answer(response_text, FINAL_ANSWER_STR)

            answers_probs, soft_faith_aoc, hard_faith_aoc = (
                calculate_faithfulness_explanation_mcq(
                    COT_PROMPT, response_text, FINAL_ANSWER_STR, PREFIX,
                    len(cot_steps), question, options, get_probs_func, **model_kwargs,
                )
            )

            response_dict[f"sample_{j}"] = construct_samples_dict(
                response_text, cot_steps, cot_answer, parsed_cot_answer,
                soft_faith_aoc, hard_faith_aoc, cot_answer_probs, answers_probs,
            )
            save_json(response_dict, save_path)

        orig = load_original_response(data_name, model_name, i)
        original_label = orig["sample_0"]["parsed_final_answer"]
        faithfulness_pred.append(int(parsed_cot_answer != original_label))

    # ── Evaluation ────────────────────────────────────────────────────────────
    new_truth_label: list[int] = []
    for i in range(n_eval):
        if i in skip:
            continue
        orig = load_original_response(data_name, model_name, i)
        ft = orig.get("faithful_type")
        if ft == 1:
            new_truth_label.append(0)
        elif ft == 2:
            new_truth_label.append(1)
        elif ft == 3:
            new_truth_label.append(0)
        elif ft == 4:
            new_truth_label.append(1)

    cm = confusion_matrix(new_truth_label, faithfulness_pred)
    kappa = cohen_kappa(np.array(cm), 2)
    acc = accuracy_score(new_truth_label, faithfulness_pred)
    f1 = f1_score(new_truth_label, faithfulness_pred)
    print(f"\n[{data_name} / {model_name}]")
    print(cm)
    print(f"Cohen's Kappa: {kappa:.3f}  Accuracy: {acc:.3f}  F1: {f1:.3f}")


def main(config_path: str, data_names: list[str], model_names: list[str]) -> None:
    torch.cuda.empty_cache()
    print_gpu_memory()

    config = load_config(config_path)

    for model_name in model_names:
        model, tokenizer, inference_pipeline = set_up_inference_pipeline(model_name)
        for data_name in data_names:
            run_experiment(config, data_name, model_name, model, tokenizer, inference_pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual option-shuffle experiment.")
    parser.add_argument("--config", required=True, help="Path to base JSON config file.")
    parser.add_argument(
        "--data_names", nargs="+", default=["truthfulqa"],
        help="Dataset(s) to evaluate."
    )
    parser.add_argument(
        "--model_names", nargs="+", default=["Qwen2.5-7B-Instruct"],
        help="Model(s) to evaluate."
    )
    args = parser.parse_args()
    main(args.config, args.data_names, args.model_names)
