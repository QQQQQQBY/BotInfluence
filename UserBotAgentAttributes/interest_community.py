# assess the interest community score

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

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)

def IC_main(config, logger):
    
    logger.info(f'IC Program start time:{time_utils.get_now_time()}')
    logger.info(" ")

    # 1. load prompt
    prompt, input_variables, attribute = load_prompt(config["paths"]["InterestCommunity"])
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
        out_file = config['newpaths']['IC_Json'] + out_file.split(config['paths']['RawDataset'])[-1] + "_interest_community.json"        
        # check if the output file exists
        if os.path.exists(out_file):
            logger.info(f"{out_file} file already exists, skipping...")
            continue
        
        count = 0
        comm = input_file.split("TimeTruncationData")[-1]
        logger.info(f"Begin Processing: {comm}")
        for user_datas in tqdm(user_data_batch):
            user_data = {
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

        logger.info(f"End of Processing: {comm}")
        # 4. save interest community evaluation outcome
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=4, ensure_ascii=False)

        logger.info(f'Save to {out_file} file:{time_utils.get_now_time()}')

        logger.info(" ")
        logger.info(f'End time of {input_file} file:{time_utils.get_now_time()}')
        
    logger.info(f'End time of IC Program:{time_utils.get_now_time()}')
    return "Successfully processed all users"

def dispose_interest_community(community_dict, comm):
    flag = 0
    interest_community = []
    for key, value in community_dict.items():
        if value >= 8:
            interest_community.append(key)
            flag = 1
    if flag == 0:
        max_value = max(community_dict.values())
        max_index = list(community_dict.values()).index(max_value)
        if max_index != 0:
            interest_community.append(list(community_dict.keys())[max_index])
        elif max_index == 0:
            interest_community.append(comm)    
    return interest_community

def dispose_input(interest_domain_scores):
    """
    Disposal of input data
    """
    result = {}
    for items in interest_domain_scores:
        try:
            result[items['domain']] = float(items['score'])
        except:
            # prevent the error of divide by zero
            result[items['domain']] = 1
    return result

def concatusers_IC(output_file, json_files, logger, config):
    # 0. detect whether the output file exists
    if os.path.exists(output_file):
        logger.info(f"Output file already exists, skipping...")
        return "Output file already exists, skipping..."
    
    # concat users
    All_Users_batch = []
    count = 0
    for input_file in json_files: 
        comm = (input_file.split("_interest_community.json")[0]).split(config['newpaths']['IC_Json'])[-1]
        with open(input_file, 'r', encoding='utf-8') as f:
            user_data_batch = json.load(f)
        for user_data in user_data_batch:
            user = {}
            user['id'] = count + 1
            count = count + 1
            user['user_id'] = str(user_data['user_id'])
            user['community'] = dispose_input(user_data['interest_domain_scores'])
            user['interest community'] = dispose_interest_community(user['community'], comm)
            All_Users_batch.append(user)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert(All_Users_batch), f, indent=4, ensure_ascii=False)
    return "Successfully processed all users"

def interest_community_main():
    # 1. load config
    config = load_config("UserBotAgentAttributes/config.yaml")
    logger = set_logger(config['log']['IC_log_file'])
    # 2. process all users
    outcome =  IC_main(config, logger)
    print(outcome)
    # 3. concat users
    concat_output_file = config['newpaths']['IC_Json'] + 'concat_interest_community.json'
    outcome = concatusers_IC(concat_output_file, glob.glob(config['newpaths']['IC_Json'] + '*.json'), logger, config)
    print(outcome)

if __name__ == '__main__':
    interest_community_main()
