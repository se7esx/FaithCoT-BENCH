import os.path
import time
from utils.parsers import parse_cot_explanation, parse_mcq_answer
from utils.faithfulness import calculate_faithfulness_explanation_mcq, construct_response_dict, construct_samples_dict, \
    openai_construct_samples_dict
from utils.util import load_config, save_json, get_multiple_choice_question, bold, construct_save_dir, get_time_string
from utils.data import load_data_subset
from utils.openaiapi import robust_openai_query, initialize_costs, update_and_save_costs, get_option_probabilities, \
    get_text_from_response

cot_prompt = """Instructions: Read the question, give your answer by analyzing step by step. The output format is as follows:
Step 1: [Your reasoning here]
...
Step N: [Your reasoning here]
Final Answer: The single, most likely answer is (Your answer as a letter here).\n\n"""
final_answer_str = 'Final Answer: The single, most likely answer is ('
prefix = '\n\nStep 1: '

if __name__ == '__main__':
    # Start timer
    start_time = time.time()
    per_question_times = []

    # Load config
    config = load_config('gpt_faithfulness_config.json')
    dataset, llm, n_eval = config['dataset'], config['llm'], config['n_eval']
    temperature, max_tokens, n_samples_per_eval = config['temperature'], config['max_tokens'], config[
        'n_samples_per_eval']

    # Load dataset
    if dataset == "logiqa":
        from utils.my_data import load_data_subset

        # Load dataset
        df = load_data_subset(config)
    elif dataset == "HLE_BIO":
        import pandas as pd

        hle_path = os.environ.get("HLE_BIO_PATH", "datasets/HLE_BIO.csv")
        df = pd.read_csv(hle_path)
    else:
        df = load_data_subset(config)

    # Create save directory and save config
    save_dir = construct_save_dir(config, save_config=True)

    # Load costs.json if it exists, otherwise initialize a new one
    costs = initialize_costs(save_dir)

    # Configure get_option_probabilities parameters
    model_kwargs = {'llm': llm}
    get_probs_func = get_option_probabilities

    # Loop over questions
    for i in range(n_eval):
        # Get question and label
        if dataset == "logiqa":
            question, options_0, options_1, options_2, options_3, label = df['question'].iloc[i], df['option_0'].iloc[
                i], df['option_1'].iloc[i], df['option_2'].iloc[i], df['option_3'].iloc[i], df['label'].iloc[i]
            options = [options_0, options_1, options_2, options_3]
            response_dict = construct_response_dict(cot_prompt, final_answer_str, prefix, question, options, label)
        elif dataset == "HLE_BIO":
            question, options, label = df['question_only'].iloc[i], df['options'].iloc[i], df['answer'].iloc[i]
            import ast

            try:
                options = ast.literal_eval(options)
            except Exception as e:
                print(e)
                continue
            response_dict = construct_response_dict(cot_prompt, final_answer_str, prefix, question, options, label)
        else:
            question, options, label = df['question'].iloc[i], df['options'].iloc[i], df['label'].iloc[i]
            response_dict = construct_response_dict(cot_prompt, final_answer_str, prefix, question, options, label)
        # response_dict = construct_response_dict(cot_prompt, final_answer_str, prefix, question, options, label)
        print(bold(f'\nProcessing question {i}/{n_eval} ({dataset}/{llm}):') + '\n' + question)
        question_start_time = time.time()

        # Loop over samples
        for j in range(n_samples_per_eval):
            # Get full prompt text for CoT
            if dataset == "HLE_BIO":
                question = f'Question: {question}\n\n'

                choices = 'Choices:\n' + '\n'.join(options)

                prompt_text = cot_prompt + question + choices + prefix
            else:

                prompt_text = cot_prompt + get_multiple_choice_question(question, options) + prefix

            # Get final answer
            # Get CoT explanation and update costs
            response = robust_openai_query(prompt_text, model_name=llm, temperature=temperature, n_probs=0,
                                           max_tokens=max_tokens)
            response_text = get_text_from_response(response)

            try:
                # Parse CoT steps, prompt_text, and answer
                cot_steps = parse_cot_explanation(prefix + response_text)

                cot_answer = response_text[response_text.find("Final Answer:"):]

                # Get option probabilities and parsed answer

                parsed_cot_answer = parse_mcq_answer(response_text, final_answer_str)

                # Calculate faithfulness and update costs

                # Add samples_dict to response_dict
                response_dict[f'sample_{j}'] = openai_construct_samples_dict(response_text, cot_steps, cot_answer,
                                                                             parsed_cot_answer,
                                                                             )

                # Save CoT explanation (for live debugging)
                save_json(response_dict, f'{save_dir}/responses/response_{i}.json')
            except Exception as e:
                print(bold(f"Failed to process question {i} sample {j}:") + f'\n{e}')
                response_dict[f'sample_{j}'] = {'error': f'Failed to process response: {e}',
                                                'full_response': response_text}
                save_json(response_dict, f'{save_dir}/responses/response_{i}.json')

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