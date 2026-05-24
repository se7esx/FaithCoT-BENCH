"""
Generate fine-tuning example config files for all experimental conditions.

Reads response files from an existing results directory and selects examples
according to various faithfulness-based strategies, saving them as JSONL
configs ready for the OpenAI fine-tuning pipeline.

Usage:
    python find_finetuning_examples.py <dataset_name> <model_name>

Example:
    python find_finetuning_examples.py logiqa gpt-4-0613
"""

import sys

from utils.selection import (
    get_ea_response,
    get_ft_examples,
    get_label_response,
    get_qa_prompt,
    get_qe_prompt,
)
from utils.util import bold, construct_save_dir, save_ft_config

# Config for the pipeline that will run *after* fine-tuning.
FT_CONFIG_TEMPLATE = {
    "dataset": "",
    "dataset_params": {
        "split": "test",
        "n": 100,
        "seed": 42,
    },
    "llm": "",
    "temperature": 0.0,
    "max_tokens": 512,
    "n_eval": 100,
    "n_samples_per_eval": 1,
}

# Experimental conditions: name → [responses_dir, n_examples, prompt_fn, response_fn, correct_only]
def build_run_args(responses_dir_template: str) -> dict:
    return {
        "baseline_2":   [responses_dir_template.format(0.0), 100, get_qa_prompt, get_label_response, False],
        "baseline_3":   [responses_dir_template.format(0.0), 100, get_qe_prompt, get_ea_response,    False],
        "approach_1":   [responses_dir_template.format(0.0), 10,  get_qe_prompt, get_ea_response,    False],
        "approach_2":   [responses_dir_template.format(0.3), 100, get_qe_prompt, get_ea_response,    False],
        "approach_3":   [responses_dir_template.format(0.3), 10,  get_qe_prompt, get_ea_response,    False],
        "approach_4":   [responses_dir_template.format(0.0), 50,  get_qe_prompt, get_ea_response,    False],
        "approach_5":   [responses_dir_template.format(0.3), 50,  get_qe_prompt, get_ea_response,    False],
        "baseline_3_c": [responses_dir_template.format(0.0), 100, get_qe_prompt, get_ea_response,    True],
        "approach_1_c": [responses_dir_template.format(0.0), 10,  get_qe_prompt, get_ea_response,    True],
        "approach_2_c": [responses_dir_template.format(0.3), 100, get_qe_prompt, get_ea_response,    True],
        "approach_3_c": [responses_dir_template.format(0.3), 10,  get_qe_prompt, get_ea_response,    True],
        "approach_4_c": [responses_dir_template.format(0.0), 50,  get_qe_prompt, get_ea_response,    True],
        "approach_5_c": [responses_dir_template.format(0.3), 50,  get_qe_prompt, get_ea_response,    True],
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python find_finetuning_examples.py <dataset_name> <model_name>")
        sys.exit(1)

    dataset_name, model_name = sys.argv[1], sys.argv[2]
    print(bold(f"Generating fine-tuning examples for {dataset_name}/{model_name}"))

    ft_config = dict(FT_CONFIG_TEMPLATE)
    ft_config["dataset"] = dataset_name
    ft_config["llm"] = model_name

    save_dir = construct_save_dir(ft_config, save_config=False, prefix="ft_configs")
    responses_dir_template = (
        f"results/{dataset_name}/{model_name}/"
        "train_n_400_seed_42_temp_{}_maxtokens_512/responses/"
    )

    run_args = build_run_args(responses_dir_template)

    for run_name, args in run_args.items():
        examples = get_ft_examples(*args)
        save_ft_config(ft_config, save_dir, run_name, args[0], examples)
        print(f"  Saved config: {run_name}  ({len(examples)} examples)")

    print()
