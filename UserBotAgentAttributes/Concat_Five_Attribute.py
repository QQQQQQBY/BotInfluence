import json
from utils import load_config, convert
from set_logger import *
from interest_community import interest_community_main
from trust_threshold import trust_threshold_main
from disseminate_tendency import disseminate_tendency_main
from social_influence import social_influence_main
from activation_time import activation_time_main
from tqdm import tqdm
import os


def Concat_main(config, logger):
    concat_file = config['ConcatFiles']
    # if os.path.exists(concat_file):
    #     logger.info(f"{concat_file} file already exists, skipping...")
    #     return "complete concatenate five attribute"
    results = {}

    logger.info("Start concatenate five attribute")
    logger.info("1. load interest community")
    # 1. load interest community
    IC_file = config['newpaths']['IC_Json'] + 'concat_interest_community.json'
    with open(IC_file, 'r', encoding='utf-8') as f:
        IC_data = json.load(f)

    # 2. load trust threshold
    logger.info("2. load trust threshold")
    TT_file = config['newpaths']['TT_Json'] + 'concat_trust_threshold.json'
    with open(TT_file, 'r', encoding='utf-8') as f:
        TT_data = json.load(f)

    # 3. load dissemination tendency
    logger.info("3. load dissemination tendency")
    DT_file = config['newpaths']['DT_Json'] + 'concat_disseminate_tendency.json' 
    with open(DT_file, 'r', encoding='utf-8') as f:
        DT_data = json.load(f)

    # 4. load social influence
    logger.info("4. load social influence")
    SI_file = config['newpaths']['SI_Json'] + 'concat_social_influence.json'
    with open(SI_file, 'r', encoding='utf-8') as f:
        SI_data = json.load(f)
   
    # 5. load activation time
    logger.info("5. load activation time")
    AT_file = config['newpaths']['AT_Json'] + 'concat_activation_time.json'
    with open(AT_file, 'r', encoding='utf-8') as f:
        AT_data = json.load(f)

    # Concatenate all users
    logger.info("6. Concatenate all users")
    count = 0
    for index in tqdm(range(len(IC_data))):
            if IC_data[index]['user_id'] not in results:
                try:
                    result = {}
                    result['user_id'] = IC_data[index]['user_id']
                    result['community'] = IC_data[index]['community']
                    result['interest_community'] = IC_data[index]['interest community']
                    result['trust_threshold'] = TT_data[index]['trust threshold']
                    result['dissemination_tendency'] = DT_data[index]['DTScore']
                    result['social_influence'] = SI_data[index]['followers_count']
                    result['activation_time'] = AT_data[index]['activation_possibility']
                    result['sharetimes'] = AT_data[index]['sharetimes']
                    result['label'] = 'Human'
                    result['id'] = count 
                    count += 1
                    results[IC_data[index]['user_id']] = result
                except:
                    continue
                
    logger.info("7. Save to file")
    with open(concat_file, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=4, ensure_ascii=False)
    logger.info("8. Concatenate five attribute complete")
    return "complete concatenate five attribute"

if __name__=='__main__':
    # 1. process each attribute
    interest_community_main()
    trust_threshold_main()
    disseminate_tendency_main()
    social_influence_main()
    activation_time_main()
    # 2. concatenate all attribute
    config = load_config("UserBotAgentAttributes/config.yaml")
    logger = set_logger(config['log']['Concat_log_file'])
    outcome = Concat_main(config, logger)
    print(outcome)