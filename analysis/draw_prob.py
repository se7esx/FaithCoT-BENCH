"""
Plot the intermediate answer-probability trajectory for a single question.

Usage:
    python analysis/draw_prob.py \\
        --response_file results/aqua/llama-3.1-8b-instruct/.../responses/response_0.json \\
        --sample sample_0 \\
        --output figures/response_0_probs.pdf
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def plot_probability_trajectory(
    intermediate_probs: list[dict],
    title: str,
    output_path: str,
    options: list[str] | None = None,
) -> None:
    if options is None:
        options = ["A", "B", "C", "D"]

    time_steps = list(range(len(intermediate_probs)))
    prob_by_option = {opt: [] for opt in options}

    for step in intermediate_probs:
        for opt in options:
            prob_by_option[opt].append(step.get(opt, 0.0))

    plt.figure(figsize=(10, 6))
    for opt in options:
        plt.plot(time_steps, prob_by_option[opt], label=f"Option {opt}", marker="o")

    plt.xlabel("CoT Step")
    plt.ylabel("Probability")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot intermediate answer-probability trajectory."
    )
    parser.add_argument("--response_file", required=True, help="Path to a response JSON file.")
    parser.add_argument("--sample", default="sample_0", help="Sample key to plot (default: sample_0).")
    parser.add_argument("--output", default="figures/probability_trajectory.pdf", help="Output file path.")
    parser.add_argument(
        "--options", nargs="+", default=["A", "B", "C", "D"],
        help="Answer option letters to plot."
    )
    args = parser.parse_args()

    with open(args.response_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data.get(args.sample)
    if sample is None:
        raise KeyError(f"Key '{args.sample}' not found in {args.response_file}.")

    intermediate_probs = sample.get("intermediate_answer_probabilities")
    if not intermediate_probs:
        raise ValueError(
            f"'intermediate_answer_probabilities' is missing or empty in {args.sample}."
        )

    faithfulness = sample.get("soft_faithfulness", "N/A")
    title = f"Intermediate Answer Probabilities over Steps (soft faithfulness = {faithfulness:.3f})"

    plot_probability_trajectory(intermediate_probs, title, args.output, args.options)


if __name__ == "__main__":
    main()
