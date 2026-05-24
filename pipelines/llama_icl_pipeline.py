"""
LLaMA / Qwen in-context learning (ICL) faithfulness pipeline.

Usage:
    python pipelines/llama_icl_pipeline.py --config <path_to_icl_config.json>
"""
import argparse
import time
import torch
from pipelines.prompts import COT_PROMPT as cot_prompt, FINAL_ANSWER_STR as final_answer_str, PREFIX as prefix
from utils.parsers import parse_cot_explanation, parse_mcq_answer
from utils.faithfulness import calculate_faithfulness_explanation_mcq, construct_response_dict, construct_samples_dict, construct_icl_save_dir
from utils.util import load_config, print_gpu_memory, save_json, get_multiple_choice_question, bold, construct_icl_prompt_from_examples
from utils.data import load_data_subset
from utils.llama import set_up_inference_pipeline, get_option_probabilities

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to ICL config JSON.")
    args = parser.parse_args()
    config_file_path = args.config
    start_time = time.time()



    # Print device and available memory
    torch.cuda.empty_cache()
    print_gpu_memory()

    # Load config
    config = load_config(config_file_path)
    dataset, llm, n_eval = config['dataset'], config['llm'], config['n_eval']
    temperature, max_tokens, n_samples_per_eval = config['temperature'], config['max_tokens'], config['n_samples_per_eval']
    icl_examples, run_name, exclude_explanation = config['icl_examples'], config['run_name'], config['exclude_explanation']

    # Load dataset
    df = load_data_subset(config)

    # Create save directory and save config
    save_dir = construct_icl_save_dir(config, save_config=True)
    print(f'Save directory: {save_dir}')

    # Construct prompt from icl examples
    icl_prompt = construct_icl_prompt_from_examples(icl_examples, exclude_explanation)

    # Set up inference pipeline
    model, tokenizer, inference_pipeline = set_up_inference_pipeline(llm)
    model_kwargs = {'model': model, 'tokenizer': tokenizer}  # used to compute option probabilities
    get_probs_func = get_option_probabilities
    
    # Loop over questions
    for i in range(n_eval):
        # Get question and label
        question, options, label = df['question'].iloc[i], df['options'].iloc[i], df['label'].iloc[i]
        response_dict = construct_response_dict(cot_prompt, final_answer_str, prefix, question, options, label)
        print(bold(f'\nProcessing question {i}:') + '\n' + question)
        question_start_time = time.time()
        
        # Loop over samples
        for j in range(n_samples_per_eval):
            # Get full prompt text for CoT
            prompt_text = icl_prompt + cot_prompt + get_multiple_choice_question(question, options) + prefix

            # Get final answer
            do_sample = True if temperature else False
            response_text = inference_pipeline(prompt_text, do_sample=do_sample, max_new_tokens=max_tokens, truncation=True,
                                               top_p=None, temperature=temperature)[0]['generated_text']

            # Parse CoT explanation, steps, and answer
            try:
                # Parse CoT steps, prompt_text, and answer
                cot_steps = parse_cot_explanation(prefix + response_text)
                prompt_text += response_text[:response_text.find("Final Answer:")] + final_answer_str
                cot_answer = response_text[response_text.find("Final Answer:"):]

                # Get option probabilities and parsed answer
                cot_answer_probs = get_option_probabilities(model, tokenizer, prompt_text, options)
                parsed_cot_answer = parse_mcq_answer(response_text, final_answer_str)

                # Calculate faithfulness and update costs
                answers_probs, soft_faith_aoc, hard_faith_aoc =\
                    calculate_faithfulness_explanation_mcq(cot_prompt, response_text, final_answer_str, prefix,
                                                           len(cot_steps), question, options, get_probs_func, **model_kwargs)

                # Add samples_dict to response_dict
                response_dict[f'sample_{j}'] = construct_samples_dict(response_text, cot_steps, cot_answer, parsed_cot_answer,
                                                                      soft_faith_aoc, hard_faith_aoc, cot_answer_probs, answers_probs)

                # Save CoT explanation (for live debugging)
                save_json(response_dict, f'{save_dir}/response_{i}.json')
            except Exception as e:
                print(bold(f"Failed to process question {i} sample {j}:") + f'\n{e}')
                response_dict[f'sample_{j}'] = {'error': f'Failed to process response: {e}', 'full_response': response_text}
                save_json(response_dict, f'{save_dir}/response_{i}.json')
        
        # Print time for question
        print(f'\nTime for question {i}: {time.time() - question_start_time:.2f} seconds')

    # Print time
    print(f'\nTotal time: {time.time() - start_time:.2f} seconds')