import time
import torch
import sys
import os
import json
import pyvene as pv
from torch.nn.functional import softmax
from utils.parsers import parse_cot_explanation, parse_mcq_answer
from utils.faithfulness import calculate_faithfulness_explanation_mcq, construct_response_dict, construct_samples_dict
from utils.util import load_config, print_gpu_memory, save_json, get_multiple_choice_question, bold, construct_data_path
from utils.data import load_data_subset
from utils.llama import get_base_model, get_model_and_tokenizer
from transformers import pipeline
from string import ascii_uppercase

letters = list(ascii_uppercase)

cot_prompt = """Instructions: Read the question, give your answer by analyzing step by step. The output format is as follows:
Step 1: [Your reasoning here]
...
Step N: [Your reasoning here]
Final Answer: The single, most likely answer is (Your answer as a letter here).\n\n"""
final_answer_str = 'Final Answer: The single, most likely answer is ('
prefix = '\n\nStep 1: '

def construct_intervention_save_dir(config, save_config=True):
    """
    Construct save directory and save config
    config: dictionary, configuration dictionary
    save_config: boolean, if True, save the config to the save directory (will overwrite existing file)
    returns: string, save directory path
    """
    _, data_name, data_path = construct_data_path(config).rstrip('.parquet').split('/')
    llm, temperature, max_tokens, run_name = config['llm'], config['temperature'], config['max_tokens'], config['run_name']
    save_dir = f'intervention_results/{data_name}/{llm}/{data_path}_temp_{temperature}_maxtokens_{max_tokens}/{run_name}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_config:
        with open(f'{save_dir}/faithfulness_config.json', 'w') as f:
            json.dump(config, f, indent=4)
    return save_dir

def get_option_probabilities(model, tokenizer, prompt_text, options):
    # Tokenize the input and send to the same device as the model
    inputs = tokenizer(prompt_text, return_tensors="pt").to(torch.device('cuda:0'))
    with torch.no_grad():
        outputs = model(inputs)[1]
        logits = outputs.logits

    # Get logits for the last token position before completion
    last_token_logits = logits[0, -1, :]

    # Map options to their respective token IDs
    letter_options = letters[:len(options)]
    option_ids = [tokenizer.encode(letter_option + ")", add_special_tokens=False)[0] for letter_option in letter_options]

    # Calculate probabilities
    probs = softmax(last_token_logits, dim=-1)

    # Extract probabilities for the options
    option_probs = {letter_option: probs[option_id].item() for letter_option, option_id in zip(letter_options, option_ids)}

    return option_probs


def set_up_intervened_inference_pipeline(llm, activations):
    """
    Set up the inference pipeline
    llm: string, name of the LLM
    returns: pipeline, inference pipeline
    """
    # Configure environment
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Load the model and tokenizer
    torch.cuda.empty_cache()
    base_model = get_base_model(llm)
    print("Setting up inference pipeline for model:", base_model)

    # Quantization
    model, tokenizer = get_model_and_tokenizer(base_model)

    pv_model = pv.IntervenableModel([{
            "component": f"model.layers[{i}].self_attn.o_proj.output",
            "intervention": pv.AdditionIntervention(
                source_representation=activations[i].to("cuda")
            )
        } for i in range(32) if torch.count_nonzero(activations[i])], 
        model=model
    )
    
    # Set up the inference pipeline
    # inference_pipeline = pipeline("text-generation", model=pv_model, tokenizer=tokenizer, return_full_text=False)
    inference_pipeline = None
    return pv_model, tokenizer, inference_pipeline


if __name__ == '__main__':
    # Start timer
    start_time = time.time()

    assert len(sys.argv) == 2, "pass config_file_path as a cli args!"

    config_file_path = sys.argv[1]

    # Print device and available memory
    torch.cuda.empty_cache()
    print_gpu_memory()

    # Load config
    config = load_config(config_file_path)
    dataset, llm, n_eval = config['dataset'], config['llm'], config['n_eval']
    temperature, max_tokens, n_samples_per_eval = config['temperature'], config['max_tokens'], config['n_samples_per_eval']
    activations_path, run_name, exclude_explanation = config['activations_path'], config['run_name'], config['exclude_explanation']
    flip_intervention = config.get('flip_intervention', False)

    # Load dataset
    df = load_data_subset(config)

    activations = torch.load(activations_path)
    if flip_intervention:
        print('Flipping the translations')
        for layer_id in activations:
            activations[layer_id] = -1 * activations[layer_id]

    # Create save directory and save config
    save_dir = construct_intervention_save_dir(config, save_config=True)
    print(f'Save directory: {save_dir}')

    # Set up inference pipeline
    model, tokenizer, inference_pipeline = set_up_intervened_inference_pipeline(llm, activations)
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
            prompt_text = cot_prompt + get_multiple_choice_question(question, options) + prefix

            # Get final answer
            do_sample = True if temperature else False
            # response_text = inference_pipeline(prompt_text, do_sample=do_sample, max_new_tokens=max_tokens, truncation=True,
                                            #    top_p=None, temperature=temperature)[0]['generated_text']
            input_ids = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            _, response_shared = model.generate(input_ids, max_new_tokens=max_tokens, top_p=None, do_sample=False)
            src_len = input_ids['input_ids'].shape[1]
            response_text = tokenizer.decode(response_shared[0][src_len:], skip_special_tokens=True)

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