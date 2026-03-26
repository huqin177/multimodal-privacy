import argparse
import OpenLLMInfoExtraction as IE
from OpenLLMInfoExtraction.utils import open_config
from OpenLLMInfoExtraction.utils import open_txt, open_json, load_image
from OpenLLMInfoExtraction.utils import load_instruction, parsed_data_to_string, remove_symbols
from OpenLLMInfoExtraction.evaluator import Evaluator
import time
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def main(args):
    task_config = open_config(config_path=args.task_config_path)
    task_manager, icl_manager = IE.create_task(task_config)

    defense = IE.create_defense(args.defense)

    model_config = open_config(config_path=f'./configs/model_configs/{args.provider}_config.json')
    if args.model_name != '':  model_config['model_info']['name'] = args.model_name

    res_save_path = f'./responses/{model_config["model_info"]["provider"]}_{model_config["model_info"]["name"].split("/")[-1]}/{task_manager.dataset}_{args.defense}_{args.prompt_type}_{args.icl_num}_adaptive_attack_{args.adaptive_attack}'
    if args.redundant_info_filtering == 'False':
        res_save_path = f'{res_save_path}_False'

    raw_responses_npz = np.load(f'{res_save_path}/all_raw_responses.npz', allow_pickle=True)
    # ablation_length_perfs = np.load(f'{res_save_path}/ablation_length_perfs.npz', allow_pickle=True)

    all_raw_responses = raw_responses_npz['res'].item()
    all_labels = raw_responses_npz['label'].item()
    # print(all_raw_responses)
    # exit()
    # print(all_labels)
    # exit()
    total_num = len(all_raw_responses['email'])
    if args.defense in ('no', 'pi_ci_id', 'pi_ci', 'pi_id'):
        info_cats = open_txt('./data/system_prompts/info_category.txt')
    else:
        info_cats = ['email']
    evaluator = IE.create_evaluator(model_config["model_info"]["provider"], info_cats, metric_2=args.m2)

    for i in range(total_num):
        if len(info_cats) > 1 and 'flan' not in model_config['model_info']['name']:
            _, curr_label = task_manager[i]
        else:
            try:
                curr_label = dict(zip(info_cats, [all_labels[info_cat][i] for info_cat in info_cats]))
            except IndexError:
                continue
        for info_cat in info_cats:
            try:
                raw_response = all_raw_responses[info_cat][i]
            except IndexError:
                continue
            if raw_response is None or raw_response == '' or 'too long' in raw_response:
                continue
            
            _ = evaluator.update(raw_response, curr_label, info_cat, defense, verbose=0)
    
    evaluator.print_result()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Open Prompt Injection Evaluation')
    parser.add_argument('--s', default='0', type=str)
    parser.add_argument('--provider', default='gpt', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--dataset', default='person100', type=str)
    parser.add_argument('--defense', default='no', type=str)
    parser.add_argument('--prompt_type', default='direct_question_answering', type=str)
    parser.add_argument('--icl_num', default=0, type=int)
    parser.add_argument('--adaptive_attack', default='no', type=str)
    parser.add_argument('--m2', default='bert-score', type=str)
    parser.add_argument('--redundant_info_filtering', default='True', type=str)
    args = parser.parse_args()

    args.task_config_path = f'./configs/task_configs/{args.dataset}.json'

    main(args)