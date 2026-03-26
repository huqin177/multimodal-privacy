import os
import time
from tqdm import tqdm


def run(provider, model_name, dataset, api_key_pos, defense, prompt_type, icl_num, gpus, adaptive_attack_on_pi, redundant_info_filtering):
    model_config_path = f'./configs/model_configs/{provider}_config.json'
    task_config_path = f'./configs/task_configs/{dataset}.json'

    log_dir = f'./logs/{provider}_{model_name.split("/")[-1]}'
    os.makedirs(log_dir, exist_ok=True)

    log_file = f'{log_dir}/{dataset}_{defense}_{prompt_type}_{icl_num}_adaptive_{adaptive_attack_on_pi}_filter_{redundant_info_filtering}.txt'
    
    if dataset == 'court':
        script_name = 'main_court.py'
    else:
        script_name = 'main.py'

    cmd = f"python main.py --model_config_path {model_config_path} --model_name {model_name} --task_config_path {task_config_path} --icl_num {icl_num} --prompt_type {prompt_type} --api_key_pos {api_key_pos} --defense {defense} --adaptive_attack {adaptive_attack_on_pi} --redundant_info_filtering {redundant_info_filtering}"
    os.system(cmd)   

    #cmd = f"CUDA_VISIBLE_DEVICES={gpus}, nohup python3 -u {script_name} \
     #       --model_config_path {model_config_path} \
      #      --model_name {model_name} \
      #      --task_config_path {task_config_path} \
       #     --icl_num {icl_num} \
         #   --prompt_type {prompt_type} \
        #    --api_key_pos {api_key_pos} \
          #  --defense {defense} \
           # --gpus {gpus} \
            #--adaptive_attack {adaptive_attack_on_pi} \
     #       --redundant_info_filtering {redundant_info_filtering} \
      #      > {log_file} &"

    # if provider == 'internlm':
    #     cmd = f"CUDA_VISIBLE_DEVICES=2, nohup python3 -u {script_name} \
    #             --model_config_path {model_config_path} \
    #             --model_name {model_name} \
    #             --task_config_path {task_config_path} \
    #             --icl_num {icl_num} \
    #             --prompt_type {prompt_type} \
    #             --api_key_pos {api_key_pos} \
    #             --defense {defense} \
    #             --adaptive_attack {adaptive_attack_on_pi} \
    #             --redundant_info_filtering {redundant_info_filtering} \
    #             > {log_file} &"
    # else:
    #     cmd = f"CUDA_VISIBLE_DEVICES=5, nohup python3 -u {script_name} \
    #             --model_config_path {model_config_path} \
    #             --model_name {model_name} \
    #             --task_config_path {task_config_path} \
    #             --icl_num {icl_num} \
    #             --prompt_type {prompt_type} \
    #             --api_key_pos {api_key_pos} \
    #             --defense {defense} \
    #             --gpus {gpus} \
    #             --adaptive_attack {adaptive_attack_on_pi} \
    #             --redundant_info_filtering {redundant_info_filtering} \
    #             > {log_file} &"

    #os.system(cmd)
    #return log_file


def check_complete(log_paths):
    iter = 0
    while len(log_paths) > 0:
        # Prevent inf loop
        if iter > 10000:
            print('MAX ITER REACHED! SOMETHING BAD MAY HAVE HAPPENED! ')
            return False

        new_log_paths = []

        for log_path in log_paths:
            with open(log_path) as file:
                lines = file.read().splitlines()
                if '[END]' not in lines[-1]:
                    new_log_paths.append(log_path)

        log_paths = new_log_paths.copy()

        # Sleep for a while to avoid waste of CPU
        interactive_sleep(60)
        iter += 1

    print('COMPLETE')
    return True


def interactive_sleep(sleep_time):
    assert (0 < sleep_time and sleep_time < 181 and type(sleep_time) == int)
    for i in tqdm(range(sleep_time)):
        time.sleep(1)

""" 1 """

# model_info = [
#     'palm2',
#     'models/text-bison-001'
# ]

# model_info = [
#     'palm2',
#     'models/chat-bison-001'
# ]
# model_info = [
#     'gemini',
#     'gemini-pro'
# ]
# model_info = [
#     'gpt',
#     'gpt-3.5-turbo'
# ]
# model_info = [
#     'gpt',
#     'gpt-4'
# ]
model_info = [
    'vicuna',
    'lmsys/vicuna-13b-v1.3'
]
# model_info = [
#     'vicuna',
#     'lmsys/vicuna-7b-v1.3'
# ]

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

datasets = [
    'person100',
    # 'court',
]


""" 2 """
defenses = ['no']



# defenses = ['pi_ci_id']

# defenses = ['mask']  # mask is the keyword replacement in our paper
# defenses = ['hyperlink']
# defenses = ['replace_at']
# defenses = ['replace_dot']
# defenses = ['replace_at_dot']



# defenses = ['pi_ci', 'pi_id', 'pi_ci_id']

""" 3 """
prompt_types = ['direct_question_answering']
# prompt_types = ['pseudocode', 'contextual', 'persona', 'direct_information_extraction', 'direct_conversation']
# prompt_types = ['pseudocode', 'contextual']
# prompt_types = ['persona']


""" 4 """
adaptive_attacks_on_pi = ['no']
# adaptive_attacks_on_pi = ['sandwich', 'instructional', 'paraphrasing', 'retokenization', 'xml']#, 'delimiters', 'random_seq']
# adaptive_attacks_on_pi = ['random_seq', 'delimiters']

redundant_filtering = "True"
# redundant_filtering = "False"

""" Sanity check """
for dataset in datasets:
    assert (dataset in ['court', 'professor100', 'person100', 'famous100', 'physician100'])
for defense in defenses:
    assert (defense in ['no', 'replace_at', 'replace_at_dot', 'replace_dot', 'hyperlink', 'mask', 'pi_ci', 'pi_id', 'pi_ci_id', 'image'])
for adaptive_attack_on_pi in adaptive_attacks_on_pi:
    assert (adaptive_attack_on_pi in ['no', 'sandwich', 'xml', 'delimiters', 'random_seq', 'instructional', 'paraphrasing', 'retokenization'])
    if adaptive_attack_on_pi != 'no':
        for defense in defenses:
            assert ( 'pi' in defense or 'no' in defense )

log_paths = []
api_key_pos = 0
gpus = '0,3,4,5,6,7,8'

user_decision = input(f"Total process: {len(adaptive_attacks_on_pi)*len(datasets)*len(defenses)*len(prompt_types)}\nRun? (y/n): ")
if user_decision.lower() != 'y':
    exit()

provider = model_info[0]
model_name = model_info[1]
for adaptive_attack_on_pi in adaptive_attacks_on_pi:
    for data in datasets:

        for defense in defenses:

            for prompt_type in prompt_types:
            
                # execute
                tmp = run(provider, model_name, data, api_key_pos, defense, prompt_type, 0, str(gpus), adaptive_attack_on_pi, redundant_filtering)
                log_paths.append(tmp)

                if provider in ('palm2', 'gemini'):
                    api_key_pos = (api_key_pos + 1) % 7
                
                else:
                    gpus = ','.join(gpus.split(',')[1:])

# Sleep for a while to let the programs print something into the log
# interactive_sleep(30)
# check_complete(log_paths)
# print()
