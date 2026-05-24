"""
Plot the distribution of ground-truth labels vs. model-predicted answers
for a single results directory.

Usage:
    python analysis/draw.py \\
        --results_dir results/logiqa/llama-3.1-8b-instruct/train_n_100_seed_42_temp_0.0_maxtokens_512/responses \\
        --n 50 \\
        --output figures/logiqa_llama_distribution.pdf
"""

import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt


def load_responses(results_dir: str, n: int) -> tuple[Counter, Counter, list[int]]:
    """Load response files and return label/parsed-answer counters and mismatch IDs."""
    label_counter: Counter = Counter()
    parsed_counter: Counter = Counter()
    mismatch_ids: list[int] = []

    for i in range(n):
        path = os.path.join(results_dir, f"response_{i}.json")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        label = data.get("label", "").strip()
        if label in "ABCDEFGHI":
            label_counter[label] += 1

        for key, value in data.items():
            if not (key.startswith("sample_") and isinstance(value, dict)):
                continue
            try:
                parsed = value.get("parsed_final_answer", "").strip()
                if parsed in "ABCDEFGHI":
                    parsed_counter[parsed] += 1
                    if parsed != label:
                        mismatch_ids.append(i)
            except Exception as e:
                print(f"Error in {path} key={key}: {e}")

    return label_counter, parsed_counter, mismatch_ids


def plot_distribution(
    label_counter: Counter,
    parsed_counter: Counter,
    output_path: str,
    options: list[str] | None = None,
) -> None:
    if options is None:
        options = ["A", "B", "C", "D"]

    label_counts = [label_counter[o] for o in options]
    parsed_counts = [parsed_counter[o] for o in options]

    x = range(len(options))
    bar_width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(
        [i - bar_width / 2 for i in x], label_counts, width=bar_width,
        label="Label (Ground Truth)", color="#4C72B0",
    )
    plt.bar(
        [i + bar_width / 2 for i in x], parsed_counts, width=bar_width,
        label="Parsed Answer (Prediction)", color="#55A868",
    )
    plt.xticks(x, options)
    plt.xlabel("Answer Option")
    plt.ylabel("Count")
    plt.title("Distribution of Ground-Truth Labels vs. Model Predictions")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot answer distribution from results.")
    parser.add_argument("--results_dir", required=True, help="Path to responses/ directory.")
    parser.add_argument("--n", type=int, default=100, help="Number of response files to load.")
    parser.add_argument("--output", default="figures/distribution.pdf", help="Output file path.")
    parser.add_argument(
        "--options", nargs="+", default=["A", "B", "C", "D"],
        help="Answer option letters to include."
    )
    args = parser.parse_args()

    label_counter, parsed_counter, mismatch_ids = load_responses(args.results_dir, args.n)
    print(f"Mismatch count: {len(mismatch_ids)}")
    print(f"Label distribution:  {dict(label_counter)}")
    print(f"Parsed distribution: {dict(parsed_counter)}")
    plot_distribution(label_counter, parsed_counter, args.output, args.options)


if __name__ == "__main__":
    main()
