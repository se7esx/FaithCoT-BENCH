"""
Gemini faithfulness evaluation pipeline.

Uses the Google Gemini API to generate CoT answers for multiple-choice
questions. Unlike the LLaMA / OpenAI pipelines, Gemini does not expose
token-level probabilities, so faithfulness is measured by answer parsing
only (no step-removal AUC).

Usage:
    python pipelines/gemini_faithfulness.py \\
        --config configs/gemini_faithfulness_config.json

Requirements:
    Set GEMINI_API_KEY in your .env file (or environment).
"""

import argparse
import ast
import os
import time

import numpy as np
import pandas as pd
from google import genai

from pipelines.prompts import COT_PROMPT, FINAL_ANSWER_STR, PREFIX
from utils.data import load_data_subset
from utils.faithfulness import construct_response_dict
from utils.parsers import parse_cot_explanation, parse_mcq_answer
from utils.util import (
    bold,
    construct_save_dir,
    get_multiple_choice_question,
    get_time_string,
    load_config,
    save_json,
)

# Seconds to wait between Gemini API calls to respect rate limits.
_GEMINI_RATE_LIMIT_DELAY = 30


def _build_samples_dict(
    response_text: str,
    cot_steps: list,
    cot_answer: str,
    parsed_cot_answer: str,
) -> dict:
    """Build a samples dict for Gemini responses (no probability data available)."""
    samples = {"full_response": response_text}
    samples.update({f"step_{k + 1}": step for k, step in enumerate(cot_steps)})
    samples["final_answer"] = cot_answer
    samples["parsed_final_answer"] = parsed_cot_answer
    return samples


def load_dataset(config: dict) -> pd.DataFrame:
    dataset = config["dataset"]
    if dataset == "logiqa":
        from utils.my_data import load_data_subset as load_logiqa
        return load_logiqa(config)
    elif dataset == "HLE_BIO":
        hle_path = os.environ.get("HLE_BIO_PATH", "datasets/HLE_BIO.csv")
        return pd.read_csv(hle_path)
    return load_data_subset(config)


def get_question_data(df: pd.DataFrame, i: int, dataset: str):
    if dataset == "logiqa":
        question = df["question"].iloc[i]
        options = [df[f"option_{k}"].iloc[i] for k in range(4)]
        label = df["label"].iloc[i]
    elif dataset == "HLE_BIO":
        question = df["question_only"].iloc[i]
        options_raw = df["options"].iloc[i]
        try:
            options = ast.literal_eval(options_raw)
        except Exception as e:
            raise ValueError(f"Could not parse options for row {i}: {e}") from e
        label = df["answer"].iloc[i]
    else:
        question = df["question"].iloc[i]
        options = df["options"].iloc[i]
        label = df["label"].iloc[i]
    return question, options, label


def build_prompt(question: str, options: list, dataset: str) -> str:
    if dataset == "HLE_BIO":
        choices = "Choices:\n" + "\n".join(options)
        return COT_PROMPT + f"Question: {question}\n\n" + choices + PREFIX
    return COT_PROMPT + get_multiple_choice_question(question, options) + PREFIX


def main(config_path: str) -> None:
    # Load API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file or environment."
        )

    start_time = time.time()
    per_question_times: list[float] = []

    config = load_config(config_path)
    dataset = config["dataset"]
    llm = config["llm"]
    n_eval = config["n_eval"]
    n_samples_per_eval = config["n_samples_per_eval"]

    df = load_dataset(config)
    save_dir = construct_save_dir(config, save_config=True)

    client = genai.Client(api_key=api_key)

    for i in range(n_eval):
        try:
            question, options, label = get_question_data(df, i, dataset)
        except ValueError as e:
            print(bold(f"Skipping question {i}:") + f" {e}")
            continue

        response_dict = construct_response_dict(
            COT_PROMPT, FINAL_ANSWER_STR, PREFIX, question, options, label
        )
        print(bold(f"\nProcessing question {i}/{n_eval} ({dataset}/{llm}):") + "\n" + question)
        question_start_time = time.time()

        for j in range(n_samples_per_eval):
            prompt_text = build_prompt(question, options, dataset)

            try:
                api_response = client.models.generate_content(
                    model=llm,
                    contents=prompt_text,
                )
                time.sleep(_GEMINI_RATE_LIMIT_DELAY)
                response_text = api_response.text

                cot_steps = parse_cot_explanation(PREFIX + response_text)
                cot_answer = response_text[response_text.find("Final Answer:"):]
                parsed_cot_answer = parse_mcq_answer(response_text, FINAL_ANSWER_STR)

                response_dict[f"sample_{j}"] = _build_samples_dict(
                    response_text, cot_steps, cot_answer, parsed_cot_answer
                )
            except Exception as e:
                print(bold(f"Failed to process question {i} sample {j}:") + f"\n{e}")
                response_dict[f"sample_{j}"] = {
                    "error": f"Failed to process response: {e}",
                    "full_response": getattr(api_response, "text", ""),
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
    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Gemini faithfulness evaluation pipeline.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    args = parser.parse_args()
    main(args.config)
