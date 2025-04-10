# Add edges for MBot

import json
import numpy as np
from utils import load_config, convert
from set_logger import *

def Concat_Human_Bots(config, logger):
    # load human data
    logger.info("Load human data")
    human_data_file = config['paths']['SampleDataset']
    with open(human_data_file, 'r', encoding='utf-8') as f:
        human_data = json.load(f)
    # load bot data
    logger.info("Load bot data")
    bot_data_file = config['newpaths']['MBotDataset']
    with open(bot_data_file, 'r', encoding='utf-8') as f:
        bot_data = json.load(f)
    Results = {}
    for human_data_key in human_data.keys():
        Results[human_data_key] = human_data[human_data_key]
    for bot_data_key in bot_data.keys():
        Results[bot_data_key] = bot_data[bot_data_key]
    output_file = config['newpaths']['ConcatHumanBots']
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert(Results), f, indent=4, ensure_ascii=False)
        

if __name__ == '__main__':
    # load config
    config = load_config("DisseminationNetwork/config.yaml")
    logger = set_logger(config['log']['construct_network'])
    # run
    Concat_Human_Bots(config, logger)