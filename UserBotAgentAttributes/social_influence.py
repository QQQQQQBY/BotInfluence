# quantify the potential impact of users in disseminating information
# the number of direct connections of the user node

import json
import glob
from utils import load_config, load_prompt, llm, convert, validate_json_response, format_numbered_items, process_user
from set_logger import *
import time_utils
from tqdm import tqdm

def SI_main(config, logger):
    
    logger.info(f'SI Program start time:{time_utils.get_now_time()}')
    logger.info(" ")
    jsons_files = glob.glob(config['paths']['RawDataset'] + '/*.json')
    for input_file in jsons_files:
        if "concathumantext" in input_file:
            continue
        logger.info(f'Start time of {input_file} file:{time_utils.get_now_time()}')
        logger.info(" ")
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            user_data_batch = json.load(f)
        out_file = input_file.split("_time_truncation")[0]
        out_file = config['newpaths']['SI_Json'] + out_file.split(config['paths']['RawDataset'])[-1] + "_social_influence.json"
        if os.path.exists(out_file):
            logger.info(f"{out_file} file already exists, skipping...")
            continue
        comm = input_file.split(config['paths']['RawDataset'])[-1]
        logger.info(f"Begin Processing: {comm}")
        count = 0
        for user_datas in tqdm(user_data_batch):
            result = {}
            result['id'] = count + 1
            count = count + 1
            result['user_id'] = user_datas['user_id']
            result['followers_count'] = user_datas['profile']['followers_count']
            results.append(result)
        with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(convert(results), f, indent=4, ensure_ascii=False)
        logger.info(f"SI Program complete time:{time_utils.get_now_time()}")
        logger.info(" ")
    logger.info(f"SI Program complete time:{time_utils.get_now_time()}")
    logger.info(" ")
    return "complete calculate social influence"

def concat_SI_score(config, logger):
    results = []
    count = 0
    output_file = config['newpaths']['SI_Json'] + "concat_social_influence.json"
    if os.path.exists(output_file):
        logger.info(f"{output_file} file already exists, skipping...")
        return "complete concatenate SI score"
    jsons_files = glob.glob(config['newpaths']['SI_Json'] + '*.json')
    for input_file in jsons_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            SI_score_list = json.load(f)
        for SI_score in SI_score_list:
            user = {}
            user['id'] = count + 1
            count = count + 1
            user['user_id'] = SI_score['user_id']
            user['followers_count'] = SI_score['followers_count']
            results.append(user)
    with open(output_file, 'w', encoding='utf-8') as f: 
        json.dump(convert(results), f, indent=4, ensure_ascii=False)
    logger.info(f"concat_SI_score Program complete time:{time_utils.get_now_time()}")
    logger.info(" ")
    return "complete concatenate SI score"

def social_influence_main():
    # 1. calculate social influence score
    config = load_config("UserBotAgentAttributes/config.yaml")
    logger = set_logger(config['log']['SI_log_file'])
    outcome = SI_main(config, logger)
    print(outcome)
    # 2. concatenate social influence score
    outcome = concat_SI_score(config, logger)
    print(outcome)
    
if __name__ == '__main__':
    social_influence_main()
      