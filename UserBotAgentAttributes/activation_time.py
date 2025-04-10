# statistics activation time

import json
import glob
from utils import load_config, load_prompt, llm, convert, validate_json_response, format_numbered_items, process_user
from set_logger import *
import time_utils
from tqdm import tqdm
from datetime import datetime

# calculate probability
def probability(times):
    # times = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in times]
    hour_counts = [0] * 24
    for t in times:
        hour = t.hour  # Acquisition hours (0-23)
        hour_counts[hour] += 1
    total = len(times)
    if total != 0:
        probabilities = [count/total for count in hour_counts]
        return probabilities
    if total == 0:
        return hour_counts


def AT_main(config, logger):
    
    logger.info(f'AT Program start time:{time_utils.get_now_time()}')
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
        out_file = config['newpaths']['AT_Json'] + out_file.split(config['paths']['RawDataset'])[-1] + "_activation_time.json"
        if os.path.exists(out_file):
            logger.info(f"{out_file} file already exists, skipping...")
            continue
        comm = input_file.split(config['paths']['RawDataset'])[-1]
        logger.info(f"Begin Processing: {comm}")
        count = 0
        for user_datas in tqdm(user_data_batch):
            # time series
            result = {}
            result['id'] = count + 1
            count = count + 1
            result['user_id'] = str(user_datas['user_id'])
            
            # time series
            result['timeseries'] = []
            for index in range(len(user_datas["Posts"])):
                content = user_datas["Posts"][str(index + 1)]
                result['timeseries'].append(content['create_time'])
            for index in range(len(user_datas["Quote"])):
                content = user_datas["Quote"][str(index + 1)]
                result['timeseries'].append(content['create_time'])
            for index in range(len(user_datas["Retweet"])):
                content = user_datas["Retweet"][str(index + 1)]
                result['timeseries'].append(content['create_time'])
            if len(result['timeseries']) == 0:
                continue
            # sort time type list
            sorted_times = sorted([datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in result['timeseries']])
            result['activation_possibility'] = probability(sorted_times)
            result['sharetimes'] = round(len(result['timeseries'])/3)
            results.append(result)

        logger.info(f"End of Processing: {comm}")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=4, ensure_ascii=False)
        logger.info(f'Save to {out_file} file:{time_utils.get_now_time()}')

        logger.info(" ")
        logger.info(f'End time of {input_file} file:{time_utils.get_now_time()}')
    logger.info(f'AT Program complete time:{time_utils.get_now_time()}')
    logger.info(" ")
    return "complete calculate activation time"

def concat_AT_score(config, logger):
    results = []
    count = 0
    output_file = config['newpaths']['AT_Json'] + "concat_activation_time.json"
    if os.path.exists(output_file):
        logger.info(f"{output_file} file already exists, skipping...")
        return "complete concatenate activation time"
    jsons_files = glob.glob(config['newpaths']['AT_Json'] + '*.json')
    for input_file in jsons_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            AT_score_list = json.load(f)
        for AT_score in AT_score_list:
            user = {}
            user['id'] = count + 1
            count = count + 1
            user['user_id'] = AT_score['user_id']
            user['sharetimes'] = AT_score['sharetimes']
            user['activation_possibility'] = AT_score['activation_possibility']
            results.append(user)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=4, ensure_ascii=False)
    return "complete concatenate activation time"

def activation_time_main():
    config = load_config("UserBotAgentAttributes/config.yaml")
    logger = set_logger(config['log']['AT_log_file'])
    outcome = AT_main(config, logger)
    print(outcome)
    # 2. concatenate activation time
    outcome = concat_AT_score(config, logger)
    print(outcome)
    
if __name__ == '__main__':
    activation_time_main()
