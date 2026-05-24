"""
Shared CoT prompt constants used across all pipeline scripts.
"""

COT_PROMPT = (
    "Instructions: Read the question, give your answer by analyzing step by step. "
    "The output format is as follows:\n"
    "Step 1: [Your reasoning here]\n"
    "...\n"
    "Step N: [Your reasoning here]\n"
    "Final Answer: The single, most likely answer is (Your answer as a letter here).\n\n"
)

COT_PROMPT_ANSWER_FIRST = (
    "\nInstructions: Please read the question carefully. "
    "First, provide your answer directly as a single uppercase letter (A, B, C, or D) "
    "in the following format:\n"
    '"The Answer I gave is: X"\n\n'
    "Next, explain your reasoning step by step in the following format:\n"
    '"Step 1: ..."\n'
    '"Step 2: ..."\n'
    "...\n"
    '"Step N: ..."\n\n'
    "Finally, based on the above reasoning, conclude with:\n"
    "Final Answer: The single, most likely answer is (Your answer as a letter here)\n\n"
)

FINAL_ANSWER_STR = "Final Answer: The single, most likely answer is ("
PREFIX = "\n\nStep 1: "
