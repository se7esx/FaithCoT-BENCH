"""
Generate in-context learning (ICL) example config files.

Selects Q/A/Explanation examples from existing results according to
faithfulness scores, saving ready-to-use ICL configs for the ICL pipelines.

Usage:
    python find_incontext_examples.py <dataset_name> <model_name>

Example:
    python find_incontext_examples.py logiqa llama-3.1-8b-instruct
"""

import copy
import glob
import json
import os
import random
import sys
from string import ascii_uppercase

from utils.faithfulness import get_faith_aoc, parse_mcq_answer
from utils.util import construct_save_dir, load_config

FAITHFULNESS_COLUMN = "soft_faithfulness"
FILTER_CORRECT = True
N_ICL_EXAMPLES = 10


def _get_answer(sample: dict) -> str | None:
    answer = sample.get("parsed_final_answer")
    if answer:
        return answer
    for opt in ascii_uppercase:
        fa = sample.get("final_answer", "")
        if f"({opt})" in fa or f" {opt}." in fa:
            return opt
    return None


def _get_faithfulness(response_dict: dict, sample_key: str) -> float:
    if FAITHFULNESS_COLUMN == "soft_faithfulness":
        return response_dict[sample_key][FAITHFULNESS_COLUMN]
    elif FAITHFULNESS_COLUMN == "hard_faithfulness":
        response_text = response_dict[sample_key]["full_response"]
        final_answer_str = response_dict["final_answer_str"]
        parsed = parse_mcq_answer(response_text, final_answer_str)
        probs = response_dict["sample_0"]["intermediate_answer_probabilities"]
        _, hard = get_faith_aoc(probs, parsed)
        return hard
    raise ValueError(f"Unknown faithfulness column: {FAITHFULNESS_COLUMN}")


def _perturbed_path(path: str) -> str:
    assert "_temp_0.0_" in path
    return path.replace("_temp_0.0_", "_temp_0.3_")


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_correct(path: str) -> bool:
    d = _load(path)
    return d["label"] == _get_answer(d["sample_0"])


def _most_faithful_sample_id(path: str) -> str:
    perturbed = _perturbed_path(path)
    d = _load(perturbed)
    sample_keys = [k for k in d if k.startswith("sample_")]
    assert len(sample_keys) == 5, f"Expected 5 samples, got {len(sample_keys)} in {perturbed}"
    sample_keys.sort(key=lambda k: _get_faithfulness(d, k), reverse=True)
    return sample_keys[0]


def _write_config(config: dict, config_dir: str, run_name: str) -> None:
    path = os.path.join(config_dir, f"{run_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"  Saved: {run_name}.json")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python find_incontext_examples.py <dataset_name> <model_name>")
        sys.exit(1)

    dataset_name, model_name = sys.argv[1], sys.argv[2]

    # Build a config pointing at the training-split results
    base_config = load_config("configs/llama_faithfulness_config.json")
    base_config.update({
        "dataset": dataset_name,
        "max_tokens": 512,
        "temperature": 0.0,
        "llm": model_name,
    })
    base_config["dataset_params"].update({"n": 400})

    responses_dir = construct_save_dir(base_config, save_config=False)
    all_response_files = glob.glob(os.path.join(responses_dir, "responses", "response_*.json"))

    if FILTER_CORRECT:
        all_count = len(all_response_files)
        correct_files = [p for p in all_response_files if _is_correct(p)]
        print(f"Filtered {len(correct_files)} correct from {all_count} total samples.")
    else:
        correct_files = all_response_files

    # Sort by faithfulness (descending)
    top_from_all = sorted(
        all_response_files,
        key=lambda p: _get_faithfulness(_load(p), "sample_0"),
        reverse=True,
    )
    top_correct = sorted(
        correct_files,
        key=lambda p: _get_faithfulness(_load(p), "sample_0"),
        reverse=True,
    )

    random.seed(42)
    random_k_correct = random.sample(correct_files, k=N_ICL_EXAMPLES)
    random_k_all = random.sample(all_response_files, k=N_ICL_EXAMPLES)

    # ICL pipeline config template
    icl_base = {
        "dataset": dataset_name,
        "dataset_params": {"split": "test", "n": 100, "seed": 42},
        "llm": model_name,
        "temperature": 0.0,
        "max_tokens": 1024,
        "n_eval": 100,
        "n_samples_per_eval": 1,
        "n_probs": 20,
        "add_final_answer": True,
        "exclude_explanation": False,
    }

    config_dir = construct_save_dir(icl_base, prefix="icl_configs")

    def _examples(files, sample_id="sample_0"):
        return [{"response_file": p, "sample_id": sample_id} for p in files]

    def _most_faithful_examples(files):
        return [
            {"response_file": _perturbed_path(p), "sample_id": _most_faithful_sample_id(p)}
            for p in files
        ]

    conditions = {
        # Baseline 2: Random Q+A (no explanation)
        "baseline_2": {**copy.deepcopy(icl_base), "run_name": "baseline_2",
                       "exclude_explanation": True, "icl_examples": _examples(random_k_correct)},
        # Baseline 3: Random Q+A+E from all
        "baseline_3": {**copy.deepcopy(icl_base), "run_name": "baseline_3",
                       "icl_examples": _examples(random_k_all)},
        # Baseline 3c: Random Q+A+E from correct only
        "baseline_3_c": {**copy.deepcopy(icl_base), "run_name": "baseline_3_c",
                         "icl_examples": _examples(random_k_correct)},
        # Approach 1: Top-10 most faithful from all
        "approach_1": {**copy.deepcopy(icl_base), "run_name": "approach_1",
                       "icl_examples": _examples(top_from_all[:N_ICL_EXAMPLES])},
        # Approach 1c: Top-10 most faithful from correct
        "approach_1_c": {**copy.deepcopy(icl_base), "run_name": "approach_1_c",
                         "icl_examples": _examples(top_correct[:N_ICL_EXAMPLES])},
        # Approach 2: Random Q+A+E, most-faithful instance from perturbed
        "approach_2": {**copy.deepcopy(icl_base), "run_name": "approach_2",
                       "icl_examples": _most_faithful_examples(random_k_all)},
        # Approach 2c: Same but from correct only
        "approach_2_c": {**copy.deepcopy(icl_base), "run_name": "approach_2_c",
                         "icl_examples": _most_faithful_examples(random_k_correct)},
        # Approach 3: Top-10 most faithful instances from all (perturbed)
        "approach_3": {**copy.deepcopy(icl_base), "run_name": "approach_3",
                       "icl_examples": _most_faithful_examples(top_from_all[:N_ICL_EXAMPLES])},
        # Approach 3c: Same from correct only
        "approach_3_c": {**copy.deepcopy(icl_base), "run_name": "approach_3_c",
                         "icl_examples": _most_faithful_examples(top_correct[:N_ICL_EXAMPLES])},
    }

    for run_name, config in conditions.items():
        _write_config(config, config_dir, run_name)

    print(f"\nAll ICL configs saved to: {config_dir}")
