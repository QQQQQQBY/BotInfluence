# Add edges for MBot

import json
import numpy as np
from utils import load_config, convert, set_logger

def Concat_MBot_Human_LBot(config, logger):
    # load human data
    logger.info("Load human mbots data")
    human_mbots_data_file = config['paths']['MBotHumanDataset']
    with open(human_mbots_data_file, 'r', encoding='utf-8') as f:
        human_mbots_data = json.load(f)
    # load bot data
    logger.info("Load bot data")
    lbot_data_file = config['newpaths']['LBotDataset']
    with open(lbot_data_file, 'r', encoding='utf-8') as f:
        lbot_data = json.load(f)
    Results = {}
    for human_mbots_data_key in human_mbots_data.keys():
        Results[human_mbots_data_key] = human_mbots_data[human_mbots_data_key]
    for lbot_data_key in lbot_data.keys():
        Results[lbot_data_key] = lbot_data[lbot_data_key]
    output_file = config['newpaths']['ConcatHumanMBotLBot']
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert(Results), f, indent=4, ensure_ascii=False)
        

def Concat_MBot_Human_LBot_main():
    config = load_config("DisseminationNetwork/config.yaml")
    logger = set_logger(config['log']['construct_network'])
    Concat_MBot_Human_LBot(config, logger)

if __name__ == '__main__':
    Concat_MBot_Human_LBot_main()