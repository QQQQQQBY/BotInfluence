# assess the trust threshold score

import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import glob
import os
from langchain_core.output_parsers import JsonOutputParser
from utils import load_config, load_prompt, llm, convert, validate_json_response, format_numbered_items, process_user
from set_logger import *
import time_utils
from tqdm import tqdm
def TT_main(config, logger):
    
    logger.info(f'TT Program start time:{time_utils.get_now_time()}')
    logger.info(" ")

    # 1. load prompt
    prompt, input_variables, attribute = load_prompt(config["paths"]["TrustThreshold"])
    # 2. load user info, Process all users
    jsons_files = glob.glob(config['paths']['RawDataset'] + '*.json')

    # Cycle through user information within each community (json file)
    for input_file in tqdm(jsons_files):
        if "concathumantext" in input_file:
            continue

        logger.info(f'Start time of {input_file} file:{time_utils.get_now_time()}')
        logger.info(" ")
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            user_data_batch = json.load(f)
        
        # set output file
        out_file = input_file.split("_time_truncation")[0]
        out_file = config['newpaths']['TT_Json'] + out_file.split(config['paths']['RawDataset'])[-1] + "_trust_threshold.json"
        
        # check if the output file exists
        if os.path.exists(out_file):
            logger.info(f"{out_file} file already exists, skipping...")
            continue
        
        count = 0
        comm = input_file.split("TimeTruncationData")[-1]
        logger.info(f"Begin Processing: {comm}")
        for user_datas in tqdm(user_data_batch):
            user_data = {
                "follower_count": user_datas.get('profile', {}).get('followers_count', ""),
                "following_count": user_datas.get('profile', {}).get('following_count', ""),
                "personal_description": user_datas.get('profile', {}).get('introduction', ""),
                "historical_posts": format_numbered_items([
                    p['content'] for p in user_datas.get('Posts', {}).values() 
                    if isinstance(p, dict) and 'content' in p
                ]),
                "historical_retweets": format_numbered_items([
                    p['content'] for p in user_datas.get('Retweet', {}).values()
                    if isinstance(p, dict) and 'content' in p
                ]),
                "historical_quotes": format_numbered_items([
                    p['content'] for p in user_datas.get('Quote', {}).values()
                    if isinstance(p, dict) and 'content' in p
                ])
            }
            if not user_data["personal_description"]:
                user_data["personal_description"] = "[No personal description available]"
            
            # 3. invoke llm, get json info
            result = process_user(prompt, llm, set(input_variables), attribute, user_data)
            if result:
                result['user_id'] = user_datas['user_id']
                result['id'] = count + 1
                count = count + 1
                results.append(result)
                print(f"Successfully processed user: {result['user_id']}")
                # print(json.dumps(result, indent=2))
            

        logger.info(f"End of Processing: {comm}")
        # 4. save interest community evaluation outcome
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=4, ensure_ascii=False)

        logger.info(f'Save to {out_file} file:{time_utils.get_now_time()}')

        logger.info(" ")
        logger.info(f'End time of {input_file} file:{time_utils.get_now_time()}')

    logger.info(f'End time of TT Program:{time_utils.get_now_time()}')
    return "Successfully processed all users"

def dispose_input(interest_domain_scores):
    """
    Disposal of input data
    """
    result = {}
    for items in interest_domain_scores:
        try:
            result[items['domain']] = float(items['score'])
        except:
            result[items['domain']] = 0.1
    return result

def concatusers_TT(output_file, json_files, logger):
    # 0. detect whether the output file exists
    if os.path.exists(output_file):
        logger.info(f"Output file already exists, skipping...")
        return "Output file already exists, skipping..."
    
    # concat users
    All_Users_batch = []
    count = 0
    for input_file in json_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            user_data_batch = json.load(f)
        for user_data in user_data_batch:
            user = {}
            user['id'] = count + 1
            count = count + 1
            user['user_id'] = str(user_data['user_id'])
            user['trust threshold'] = dispose_input(user_data['Trust_threshold_scores'])
            All_Users_batch.append(user)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert(All_Users_batch), f, indent=4, ensure_ascii=False)
    return "Successfully processed all users"

def trust_threshold_main():
    # 1. load config
    config = load_config("UserBotAgentAttributes/config.yaml")
    logger = set_logger(config['log']['TT_log_file'])
    # 2. process all users
    outcome = TT_main(config, logger)
    print(outcome)
    # 3. concat users
    concat_output_file = config['newpaths']['TT_Json'] + 'concat_trust_threshold.json'
    outcome = concatusers_TT(concat_output_file, glob.glob(config['newpaths']['TT_Json'] + '*.json'), logger)
    print(outcome)

if __name__ == '__main__':
    trust_threshold_main()
