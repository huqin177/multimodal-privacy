import argparse
import OpenLLMInfoExtraction as IE
from OpenLLMInfoExtraction.utils import open_config
from OpenLLMInfoExtraction.utils import open_txt, load_image
from OpenLLMInfoExtraction.utils import load_instruction, parsed_data_to_string
import time
import os
import numpy as np

def conditional_sleep(cnt, provider):
    if cnt == 1 and provider in ('palm2', 'gpt'):
        print('sleeping...')
        time.sleep(1)
    if cnt == 1 and provider in ('gemini'):
        print('sleeping... for 60 seconds...')
        time.sleep(60)
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIE Experiments')
    parser.add_argument('--model_config_path', default='./configs/model_configs/gpt_config.json', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--task_config_path', default='./configs/task_configs/person100.json', type=str)
    parser.add_argument('--api_key_pos', default=0, type=int)
    parser.add_argument('--defense', default='no', type=str)
    parser.add_argument('--prompt_type', default='direct', type=str)
    parser.add_argument('--gpus', default='', type=str)
    parser.add_argument('--icl_num', default=0, type=int)
    parser.add_argument('--adaptive_attack', default='no', type=str)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--redundant_info_filtering', default='True', type=str)
    args = parser.parse_args()

    # Set up the task configuration and manager (dataset, such as person100, professor100, etc.)
    task_config = open_config(config_path=args.task_config_path)
    task_manager, icl_manager = IE.create_task(task_config)

    # Set up the defense
    defense = IE.create_defense(args.defense)

    # Set up the model configuration
    model_config = open_config(config_path=args.model_config_path)
    model_config['model_info']['name'] = args.model_name
    # Adjust API key for closed-source LLMs
    if 'palm' in args.model_config_path or 'gemini' in args.model_config_path or 'gpt' in args.model_config_path:
        assert (0 <= args.api_key_pos < len(model_config["api_key_info"]["api_keys"]))
        model_config["api_key_info"]["api_key_use"] = args.api_key_pos
        print(f'API KEY POS = {model_config["api_key_info"]["api_key_use"]}')
    # Adjust GPU information for open-source LLMs
    else:
        args.gpus = args.gpus.split(',')
        assert (args.gpus is not [])
        model_config["params"]["gpus"] = args.gpus
        print(f'GPUS = {model_config["params"]["gpus"]}')

    # Set up the model. The model is used to create the attacker
    model = IE.create_model(config=model_config)
    model.print_model_info()
    attacker = IE.create_attacker(model, adaptive_attack=args.adaptive_attack, icl_manager=icl_manager, prompt_type=args.prompt_type)

    # Load the PII category and the corresponding system instruction
    info_cats = open_txt('./data/system_prompts/info_category.txt')
    evaluator = IE.create_evaluator(model.provider, info_cats)
    instructions = load_instruction(args.prompt_type, info_cats)

    # Adjust the adaptive attacks trategy
    need_adaptive_attack = (args.defense in ('pi_ci', 'pi_id', 'pi_ci_id', 'no'))
    if not need_adaptive_attack:  args.adaptive_attack = 'no'
    email_only = defense.defense not in ['no', 'pi_ci', 'pi_id', 'pi_ci_id', 'image']
    
    # Manually adjust the saving path. Could be improved
    if args.redundant_info_filtering == 'True':
        res_save_path = f'./responses/{model.provider}_{model.name.split("/")[-1]}/{task_manager.dataset}_{args.defense}_{args.prompt_type}_{args.icl_num}_adaptive_attack_{args.adaptive_attack}'
    else:
        res_save_path = f'./responses/{model.provider}_{model.name.split("/")[-1]}/{task_manager.dataset}_{args.defense}_{args.prompt_type}_{args.icl_num}_adaptive_attack_{args.adaptive_attack}_{args.redundant_info_filtering}'
    os.makedirs(res_save_path, exist_ok=True)

    # Declare the data structures to save the final results
    all_raw_responses = dict(zip(info_cats, [[] for _ in range(len(info_cats))]))
    all_labels = dict(zip(info_cats, [[] for _ in range(len(info_cats))]))

    # If not None, will stop early. Used initially for checking for reproducibility
    test_num = None

    # Iterate over each person's profile
    for i in range(len(task_manager)):

        # Exist early for image query
        if i >= 10 and defense.defense == 'image' and model.provider == 'gpt':
            print('\nFor budget reason, exit.\n')
            break

        # Get the raw profile and the labels
        raw_list, curr_label = task_manager[i]
        try:
            raw_list = defense.apply(raw_list, curr_label)
        except ValueError:
            print('Not applicable. Skip')
            continue
        
        if args.verbose > 0:  print(f'{i+1} / {len(task_manager)}: {curr_label["name"]}')

        # Load the image if the t2i defense is in use
        if args.defense == 'image':
            if model.provider == 'gemini':
                img = load_image(f'./data/person100_images/{curr_label["name"]}.jpg')
                if img is None:
                    print(f'Skip bad image: ./data/person100_images/{curr_label["name"]}.jpg\n')
                    continue
            elif model.provider == 'gpt':
                img = f'./data/person100_images/{curr_label["name"]}.jpg'
            else:
                raise ValueError
        else:
            img = None

        raw = '\n'.join(raw_list)

        # If the redundant information filter is in use, process the raw profile before querying
        if args.redundant_info_filtering == 'True':
            redundant_info_filter = IE.get_parser(task_manager.dataset, (args.defense == 'hyperlink'))
            redundant_info_filter.feed(raw)
            processed_data = redundant_info_filter.data

            # Apply the defense. 
            # Ideally, defense should be used regardless of the redundant information filter. 
            # In our paper, we use the filter by default, so we write it here. 
            # This could be improved by moving the application of the defense outside of the application of the redundant information filter. 
            try:
                processed_data = defense.apply(parsed_data_to_string(task_manager.dataset, processed_data, model.name), curr_label)
            except ValueError:
                print('Not applicable. Skip')
                continue
        else:
            processed_data = raw

        cnt = 0
        verbose = 1

        # Iterate over each PII category
        for info_cat, instruction in instructions.items():

            # For defenses, we only run in emails
            if email_only and info_cat != 'email':
                continue
            
            # Sleep to avoid over-querying
            conditional_sleep(cnt, model.provider)

            # Query the LLM to get the extracted PII
            try:
                raw_response = attacker.query(
                    instruction, 
                    processed_data,
                    icl_num=args.icl_num,
                    info_cat=info_cat,
                    need_adaptive_attack=need_adaptive_attack,
                    verbose=verbose,
                    idx=i,
                    total=len(task_manager),
                    image=img
                )
                verbose = 0
            except RuntimeError:
                raw_response = ""

            # Append the response and label to the data structure to be saved
            all_raw_responses[info_cat].append(raw_response)
            all_labels[info_cat].append(curr_label[info_cat])

            # Update the evaluator
            _ = evaluator.update(raw_response, curr_label, info_cat, defense, verbose=args.verbose)

            # Update the counter for sleeping purpose
            cnt = (cnt + 1) % 2
        
        # Only professor100 has this column
        if 'defense' in curr_label:
            print(f'Defense: {curr_label["defense"]}')
            if 'defense' in all_labels:
                all_labels['defense'].append(curr_label['defense'])
            else:
                all_labels['defense'] = [curr_label['defense']]
        
        # Verbose the separator
        if args.verbose > 0:  print('\n----------------\n')

        # Save the intermediate result. Only run in testing
        if test_num != None and i >= test_num:
            np.savez(f'{res_save_path}/all_raw_responses_test_2.npz', res=all_raw_responses, label=all_labels)
            print(f'\nResults are saved at: {res_save_path}\n')
            print('TEST ONLY. EXIT')
            exit()
    
    # Save the final result
    np.savez(f'{res_save_path}/all_raw_responses.npz', res=all_raw_responses, label=all_labels)
    print(f'\nResults are saved at: {res_save_path}\n')
    print('[END]')