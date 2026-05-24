"""
OpenAI / GPT faithfulness evaluation pipeline.

Uses the OpenAI API to generate CoT explanations and computes step-removal
faithfulness scores via repeated logprob queries.

Usage:
    python pipelines/openai_faithfulness_pipeline.py \\
        --config configs/openai_faithfulness_config.json
"""

import argparse
import time

from pipelines.prompts import COT_PROMPT, FINAL_ANSWER_STR, PREFIX
from utils.data import load_data_subset
from utils.faithfulness import (
    calculate_faithfulness_explanation_mcq,
    construct_response_dict,
    construct_samples_dict,
)
from utils.openaiapi import (
    get_option_probabilities,
    get_text_from_response,
    initialize_costs,
    robust_openai_query,
    update_and_save_costs,
)
from utils.parsers import parse_cot_explanation, parse_mcq_answer
from utils.util import (
    bold,
    construct_save_dir,
    get_multiple_choice_question,
    get_time_string,
    load_config,
    save_json,
)


def main(config_path: str) -> None:
    start_time = time.time()
    per_question_times: list[float] = []

    config = load_config(config_path)
    dataset = config["dataset"]
    llm = config["llm"]
    n_eval = config["n_eval"]
    temperature = config["temperature"]
    max_tokens = config["max_tokens"]
    n_samples_per_eval = config["n_samples_per_eval"]

    df = load_data_subset(config)

    save_dir = construct_save_dir(config, save_config=True)
    costs = initialize_costs(save_dir)

    model_kwargs = {"llm": llm}
    get_probs_func = get_option_probabilities

    for i in range(n_eval):
        question = df["question"].iloc[i]
        options = df["options"].iloc[i]
        label = df["label"].iloc[i]

        response_dict = construct_response_dict(
            COT_PROMPT, FINAL_ANSWER_STR, PREFIX, question, options, label
        )
        print(bold(f"\nProcessing question {i}/{n_eval} ({dataset}/{llm}):") + "\n" + question)
        question_start_time = time.time()

        for j in range(n_samples_per_eval):
            prompt_text = COT_PROMPT + get_multiple_choice_question(question, options) + PREFIX

            response = robust_openai_query(
                prompt_text, model_name=llm, temperature=temperature,
                n_probs=0, max_tokens=max_tokens,
            )
            costs = update_and_save_costs(
                costs, responses=[response], query_type="cot_explanation",
                model_name=llm, save_dir=save_dir,
            )
            response_text = get_text_from_response(response)

            try:
                cot_steps = parse_cot_explanation(PREFIX + response_text)
                fa_idx = response_text.find("Final Answer:")
                prompt_text += response_text[:fa_idx] + FINAL_ANSWER_STR
                cot_answer = response_text[fa_idx:]

                response_probs, cot_answer_probs = get_option_probabilities(
                    prompt_text, options, llm
                )
                parsed_cot_answer = parse_mcq_answer(response_text, FINAL_ANSWER_STR)

                responses, answers_probs, soft_faith_aoc, hard_faith_aoc = (
                    calculate_faithfulness_explanation_mcq(
                        COT_PROMPT, response_text, FINAL_ANSWER_STR, PREFIX,
                        len(cot_steps), question, options, get_probs_func,
                        **model_kwargs,
                    )
                )
                costs = update_and_save_costs(
                    costs, responses=responses, query_type="faithfulness",
                    model_name=llm, save_dir=save_dir,
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

        question_time = time.time() - question_start_time
        per_question_times.append(question_time)
        avg_time = sum(per_question_times) / len(per_question_times)
        remaining = (n_eval - i - 1) * avg_time
        print(f"\nTime for question {i}: {get_time_string(question_time)}")
        print(f"Average: {get_time_string(avg_time)} | Remaining: {get_time_string(remaining)}")

    print(f"\nTotal time: {get_time_string(time.time() - start_time)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI faithfulness evaluation pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()
    main(args.config)
