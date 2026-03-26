import os
from OpenLLMInfoExtraction.utils import open_txt


def read_result(provider='palm2', model='models/text-bison-001', dataset='synthetic', icl_num=0, defense='no', prompt_type='direct_question_answering', adaptive_attack_on_pi='no', m2='rouge1', redundant_info_filtering='True'):
    if dataset in ('celebrity', 'physician', 'person100', 'famous100', 'physician100'):
        dataset = {
            'celebrity' : 'famous100',
            'physician' : 'physician100',
            'person100' : 'person100',
            'famous100' : 'famous100',
            'physician100' : 'physician100'
        }[dataset]
        log_dir = f'./logs/evaluate/{provider}_{model.split("/")[-1]}'
        log_file = f'{log_dir}/{dataset}_{defense}_{prompt_type}_{icl_num}_{adaptive_attack_on_pi}_{m2}'
        if redundant_info_filtering == 'False':
            log_file += '_False'
        log_file += '.txt'
        info_cats = ['email', 'phone', 'mail', 'name', 'work', 'education', 'affiliation', 'occupation']
        result = dict(zip(info_cats, [None]*len(info_cats)))
        
        try:
            raw_file = open_txt(log_file)
            
            for line in raw_file:
                for info_cat in info_cats:
                    if info_cat in line:
                        result[info_cat] = float(line.split(' = ')[-1])
            return result
        except FileNotFoundError:
            return result
    
    else:
        raise ValueError

def run(provider, model_name, dataset, defense, prompt_type, icl_num, adaptive_attack_on_pi, m2, redundant_info_filtering):
    try:
        result = read_result(provider, model_name, dataset, icl_num, defense, prompt_type, adaptive_attack_on_pi, m2)
        quit = True
        for k in result.keys():
            if result[k] is None:
                quit = False
        if quit and not rerun:
            print('No need to run unless being forced')
            return None
    except:
        pass


    log_dir = f'./logs/evaluate/{provider}_{model_name.split("/")[-1]}'
    os.makedirs(log_dir, exist_ok=True)

    log_file = f'{log_dir}/{dataset}_{defense}_{prompt_type}_{icl_num}_{adaptive_attack_on_pi}_{m2}'
    if redundant_info_filtering == 'False':
        log_file += '_False'
    log_file += '.txt'

    cmd = f"CUDA_VISIBLE_DEVICES=5, nohup python3 -u evaluate.py \
            --provider {provider} \
            --model_name {model_name} \
            --dataset {dataset} \
            --icl_num {icl_num} \
            --prompt_type {prompt_type} \
            --defense {defense} \
            --adaptive_attack {adaptive_attack_on_pi} \
            --redundant_info_filtering {redundant_info_filtering} \
            --m2 {m2} \
            > {log_file} &"

    os.system(cmd)
    return log_file

""" 1 """
rerun = True
# model_info = [
#     'palm2',
#     'models/text-bison-001'
# ]
# model_info = [
#     'palm2',
#     'models/chat-bison-001'
# ]
# model_info = [
#     'gpt',
#     'gpt-3.5-turbo'
# ]
# model_info = [
#     'gpt',
#     'gpt-4'
# ]
# model_info = [
#     'gemini',
#     'gemini-pro'
# ]
# model_info = [
#     'vicuna',
#     'lmsys/vicuna-13b-v1.3'
# ]
model_info = [
    'vicuna',
    'lmsys/vicuna-7b-v1.3'
]
# model_info = [
#     'llama',
#     'meta-llama/Llama-2-7b-chat-hf'
# ]
# model_info = [
#     'internlm',
#     'internlm/internlm-chat-7b'
# ]
# model_info = [
#     'flan',
#     'google/flan-ul2'
# ]

""" 2 """
datasets = [
    'person100',
]



""" 3 """
defenses = ['no']
# defenses = ['image']
# defenses = ['pi_ci', 'pi_id']
# defenses = ['pi_ci_id', 'no']
# defenses = ['replace_at', 'replace_dot', 'replace_at_dot', 'hyperlink', 'mask']

""" 4 """
prompt_types = ['direct_question_answering']
# prompt_types = ['pseudocode', 'contextual', 'persona', 'direct_information_extraction', 'direct_conversation']
# prompt_types = ['pseudocode', 'contextual']
# prompt_types = ['persona']

""" 5 """
icl_nums = [
    0,
    # 1,
    # 2,
    # 3
]

""" 6 """
adaptive_attacks_on_pi = ['no']
# adaptive_attacks_on_pi = ['random_seq']
# adaptive_attacks_on_pi = ['sandwich', 'instructional', 'paraphrasing', 'retokenization', 'xml', 'delimiters', 'random_seq']
# adaptive_attacks_on_pi = ['paraphrasing', 'delimiters', 'xml']

""" 7 """
m2s = ['rouge1', 'bert-score']

redundant_info_filtering = 'True'
# redundant_info_filtering = 'False'

""" Sanity check """
for dataset in datasets:
    assert (dataset in ['person100', 'person100_enhanced', 'famous100', 'physician100', 'professor100'])
for defense in defenses:
    assert (defense in ['no', 'replace_at', 'replace_at_dot', 'replace_dot', 'hyperlink', 'mask', 'pi_ci', 'pi_id', 'pi_ci_id', 'image'])   # 'picture' is in a separate file
for adaptive_attack_on_pi in adaptive_attacks_on_pi:
    assert (adaptive_attack_on_pi in ['no', 'sandwich', 'xml', 'delimiters', 'random_seq', 'instructional', 'paraphrasing', 'retokenization'])
    if adaptive_attack_on_pi != 'no':
        for defense in defenses:
            assert ( 'pi' in defense or 'no' in defense )


user_decision = input(f"Total process: {len(adaptive_attacks_on_pi)*len(datasets)*len(defenses)*len(prompt_types)*len(icl_nums)*len(m2s)}\nRun? (y/n): ")

if user_decision.lower() != 'y':
    exit()



provider = model_info[0]
model_name = model_info[1]
for m2 in m2s:
    for adaptive_attack_on_pi in adaptive_attacks_on_pi:
        for data in datasets:

            for defense in defenses:

                for prompt_type in prompt_types:
                
                    for icl_num in icl_nums:

                        # execute
                        run(provider, model_name, data, defense, prompt_type, icl_num, adaptive_attack_on_pi, m2, redundant_info_filtering)
