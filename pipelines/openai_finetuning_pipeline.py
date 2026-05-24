import os
import sys
import time
from utils.util import load_config, save_json, save_json_lines, bold, print_config
from utils.util import construct_ft_dir, get_multiple_choice_question, get_time_string
from utils.openaiapi import (
    robust_openai_query, initialize_costs, update_and_save_costs,
    get_option_probabilities, get_text_from_response,
    finetune_model, finetuning_costs, retrieve_and_save_job,
    get_finetuning_messages, compute_finetuning_cost, retrieve_finetuning
)
from utils.parsers import parse_cot_explanation, parse_mcq_answer
from utils.faithfulness import calculate_faithfulness_explanation_mcq, construct_response_dict, construct_samples_dict
from utils.data import load_data_subset
from openai_faithfulness_pipeline import cot_prompt, final_answer_str, prefix

def finetune(llm, all_messages, save_dir, val_ratio=0.1):
    # Start timer
    ft_start_time = time.time()

    # Split messages into training and validation sets
    n_train = int((1-val_ratio) * len(all_messages))
    train_messages, val_messages = all_messages[:n_train], all_messages[n_train:]

    # Save the messages to JSONL files
    paths = {}
    for split_name, messages in zip(['train', 'val'], [train_messages, val_messages]):
        paths[split_name] = f'{save_dir}{split_name}_messages.jsonl'
        save_json_lines(messages, paths[split_name])
        print(bold(f"Saved finetuning {split_name} messages to:"), paths[split_name])

    # Fine-tune the model
    ft_job = finetune_model(trainfile_path=paths['train'],
                            model_name=llm,
                            valfile_path=paths['val'],
                            return_json=False)
    print(bold("Fine-tuning job:"), ft_job)

    # Check and refresh the job status periodically until it's done
    while ft_job.status not in ["succeeded", "failed", "cancelled"]:
        time.sleep(30)  # Sleep for 60 seconds before refreshing to avoid excessive requests
        ft_job = retrieve_finetuning(ft_job.id)
        print(bold("Job status:"), ft_job.status)
        print(bold("Time taken:"), f"{get_time_string(time.time() - ft_start_time)}")

    # Compute the cost of fine-tuning
    ft_cost, trained_tokens = compute_finetuning_cost(ft_job, model_name=llm)
    cost_save_path = f'{save_dir}ft_costs.json'
    save_json({"trained_tokens": trained_tokens, "cost_per_token": finetuning_costs[llm], "cost": ft_cost}, cost_save_path)
    print(bold(f"Fine-tuning cost: ${ft_cost:.2f}"))
    print(bold("Final job status:"), ft_job.status)
    print(bold("Total fine-tuning time:"), f"{get_time_string(time.time() - ft_start_time)}")

    # Save the final job details to a JSON file
    retrieve_and_save_job(ft_job.id, save_dir, save_checkpoints=True)

    if ft_job.status in ["failed", "cancelled"]:
        raise Exception(f"Fine-tuning job {ft_job.status}!")
    else:
        return ft_job

if __name__ == "__main__":
    # Get config file path
    assert len(sys.argv) == 2, "pass config_file_path as a cli arg!"
    config_file_path = sys.argv[1]

    # Load config
    # assert len(sys.argv) == 2, "pass config_file_path as a cli arg!"
    config_file_path = sys.argv[1]
    config = load_config(config_file_path)
    dataset, llm, n_eval = config['dataset'], config['llm'], config['n_eval']
    temperature, max_tokens, n_samples_per_eval = config['temperature'], config['max_tokens'], config['n_samples_per_eval']
    ft_examples, run_name = config['ft_examples'], config['run_name']
    print_config(config, ignore_keys=['ft_examples'])

    # Create save directory and save config
    save_dir = construct_ft_dir(config, save_config=True)
    print(bold(f"Save directory: {save_dir}"))

    # Check if finetuning job already exists
    job_status_path = f'{save_dir}job_status_succeeded.json'
    if not os.path.exists(job_status_path):
        # retrieve_and_save_job('ftjob-2BPaZf2W2v8fgU2fIUNuC9Mq', save_dir)
        print(bold(f"Fine-tuning {llm} with {len(ft_examples)} examples..." + '\n'))
        all_messages = get_finetuning_messages(ft_examples)
        ft_job = finetune(llm, all_messages, save_dir, val_ratio=0.1)
    else:
        job_dict = load_config(job_status_path)
        print(bold(f"Fine-tuning job {llm} already exists!")); print_config(job_dict)

    # Select the fine-tuned model checkpoint
    for checkpoint in range(1, 2):
        ft_checkpoint = load_config(f'{save_dir}ft_checkpoint_{checkpoint}.json')
        llm = ft_checkpoint['fine_tuned_model_checkpoint']
        print(bold(f"Fine-tuned Model Selected: {llm}"))
        print_config(ft_checkpoint)

        # Configure get_option_probabilities parameters
        model_kwargs = {'llm': llm}
        get_probs_func = get_option_probabilities

        # Load dataset
        df = load_data_subset(config)

        # Load costs.json if it exists, otherwise initialize a new one
        costs = initialize_costs(save_dir)
        
        # Start timer
        start_time = time.time()
        per_question_times = []
        
        # Loop over questions
        for i in range(n_eval):
            # Get question and label
            question, options, label = df['question'].iloc[i], df['options'].iloc[i], df['label'].iloc[i]
            response_dict = construct_response_dict(cot_prompt, final_answer_str, prefix, question, options, label)
            print(bold(f'\nProcessing question {i}/{n_eval} ({dataset}/{run_name}/{llm}):') + '\n' + question)
            question_start_time = time.time()
            
            # Loop over samples
            for j in range(n_samples_per_eval):
                # Get full prompt text for CoT
                prompt_text = cot_prompt + get_multiple_choice_question(question, options) + prefix

                # Get CoT explanation and update costs
                response = robust_openai_query(prompt_text, model_name=llm, temperature=temperature, n_probs=0, max_tokens=max_tokens)
                costs = update_and_save_costs(costs, responses=[response], query_type='cot_explanation',
                                            model_name=llm, save_dir=save_dir)
                response_text = get_text_from_response(response)

                try:
                    # Parse CoT steps, prompt_text, and answer
                    cot_steps = parse_cot_explanation(prefix + response_text)
                    prompt_text += response_text[:response_text.find("Final Answer:")] + final_answer_str
                    cot_answer = response_text[response_text.find("Final Answer:"):]

                    # Get option probabilities and parsed answer
                    cot_answer_probs = get_option_probabilities(prompt_text, options, llm)
                    parsed_cot_answer = parse_mcq_answer(response_text, final_answer_str)

                    # Calculate faithfulness and update costs
                    responses, answers_probs, soft_faith_aoc, hard_faith_aoc =\
                        calculate_faithfulness_explanation_mcq(cot_prompt, response_text, final_answer_str, prefix,
                                                            len(cot_steps), question, options, get_probs_func, **model_kwargs)
                    costs = update_and_save_costs(costs, responses=responses, query_type='faithfulness', model_name=llm, save_dir=save_dir)

                    # Add samples_dict to response_dict
                    response_dict[f'sample_{j}'] = construct_samples_dict(response_text, cot_steps, cot_answer, parsed_cot_answer,
                                                                            soft_faith_aoc, hard_faith_aoc, cot_answer_probs, answers_probs)

                    # Save CoT explanation (for live debugging)
                    save_json(response_dict, f'{save_dir}/responses_{checkpoint}/response_{i}.json')
                except Exception as e:
                    print(bold(f"Failed to process question {i} sample {j}:") + f'\n{e}')
                    response_dict[f'sample_{j}'] = {'error': f'Failed to process response: {e}', 'full_response': response_text}
                    save_json(response_dict, f'{save_dir}/responses_{checkpoint}/response_{i}.json')

            # Print time for question
            question_time = time.time() - question_start_time
            print(f'\nTime for question {i}: {get_time_string(question_time)}')
            per_question_times.append(question_time)
            avg_time = sum(per_question_times) / len(per_question_times)
            print(f'Average time per question: {get_time_string(avg_time)}')
            remaining_time = (n_eval - i - 1) * avg_time
            print(f'Estimated remaining time: {get_time_string(remaining_time)}')

        # Print time
        total_time = time.time() - start_time
        print(f'\nTotal time: {get_time_string(total_time)}')