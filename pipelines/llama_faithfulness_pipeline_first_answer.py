"""
LLaMA / Qwen faithfulness pipeline — answer-first variant.

The model gives its answer before the reasoning steps, allowing us to test
whether the CoT post-hoc rationalises a pre-committed answer.

Usage:
    python pipelines/llama_faithfulness_pipeline_first_answer.py \\
        --config configs/llama_faithfulness_config.json
"""

import argparse
import os
import time

import torch

from pipelines.prompts import COT_PROMPT_ANSWER_FIRST, FINAL_ANSWER_STR, PREFIX
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


def load_dataset(config: dict):
    dataset = config["dataset"]
    if dataset == "logiqa":
        from utils.my_data import load_data_subset as load_logiqa
        return load_logiqa(config)
    return load_data_subset(config)


def get_question_data(df, i: int, dataset: str):
    if dataset == "logiqa":
        question = df["question"].iloc[i]
        options = [df[f"option_{k}"].iloc[i] for k in range(4)]
        label = df["label"].iloc[i]
    else:
        question = df["question"].iloc[i]
        options = df["options"].iloc[i]
        label = df["label"].iloc[i]
    return question, options, label


def main(config_path: str) -> None:
    start_time = time.time()

    torch.cuda.empty_cache()
    print_gpu_memory()

    config = load_config(config_path)
    dataset = config["dataset"]
    llm = config["llm"]
    n_eval = config["n_eval"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    n_samples_per_eval = config["n_samples_per_eval"]

    df = load_dataset(config)

    save_dir = construct_save_dir(config, save_config=True)
    print(f"Save directory: {save_dir}")

    model, tokenizer, inference_pipeline = set_up_inference_pipeline(llm)
    model_kwargs = {"model": model, "tokenizer": tokenizer}
    get_probs_func = get_option_probabilities

    for i in range(n_eval):
        question, options, label = get_question_data(df, i, dataset)
        response_dict = construct_response_dict(
            COT_PROMPT_ANSWER_FIRST, FINAL_ANSWER_STR, PREFIX, question, options, label
        )
        print(bold(f"\nProcessing question {i}/{n_eval}:") + "\n" + question)
        question_start_time = time.time()

        for j in range(n_samples_per_eval):
            # Answer-first format: question comes before the prompt
            prompt_text = get_multiple_choice_question(question, options) + COT_PROMPT_ANSWER_FIRST

            do_sample = temperature > 0
            response_text = inference_pipeline(
                prompt_text,
                do_sample=do_sample,
                max_new_tokens=max_tokens,
                truncation=True,
                top_p=None,
                temperature=temperature,
            )[0]["generated_text"]

            try:
                cot_steps = parse_cot_explanation(response_text)
                fa_idx = response_text.find("Final Answer:")
                prompt_text += response_text[:fa_idx] + FINAL_ANSWER_STR
                cot_answer = response_text[fa_idx:]

                cot_answer_probs = get_option_probabilities(model, tokenizer, prompt_text, options)
                parsed_cot_answer = parse_mcq_answer(response_text, FINAL_ANSWER_STR)

                answers_probs, soft_faith_aoc, hard_faith_aoc = (
                    calculate_faithfulness_explanation_mcq(
                        COT_PROMPT_ANSWER_FIRST, response_text, FINAL_ANSWER_STR, PREFIX,
                        len(cot_steps), question, options, get_probs_func,
                        **model_kwargs,
                    )
                )

                response_dict[f"sample_{j}"] = construct_samples_dict(
                    response_text, cot_steps, cot_answer, parsed_cot_answer,
                    soft_faith_aoc, hard_faith_aoc, cot_answer_probs, answers_probs,
                )
            except Exception as e:
                print(bold(f"Failed to process question {i} sample {j}:") + f"\n{e}")
                response_dict[f"sample_{j}"] = {
                    "error": f"Failed to process response: {e}",
                    "full_response": response_text,
                }

            save_json(response_dict, f"{save_dir}/responses/response_{i}.json")

        print(f"\nTime for question {i}: {time.time() - question_start_time:.2f}s")

    print(f"\nTotal time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA answer-first faithfulness pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()
    main(args.config)
