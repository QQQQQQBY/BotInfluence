# specify the users activation time step
# set specific time step for Humans and add step 1 for MBots

import json
from utils import load_config, convert
import random
import numpy as np

def random_select_time_step(activationtime, activationtimes):
    timesteps = [i for i in range(1,25)]
    if len(activationtime) != len(timesteps):
        raise ValueError("List length must be the same")
    np.random.seed(42)
    return sorted(list(set(random.choices(timesteps, weights=activationtime, k=activationtimes))))

def three_days(activation_time):
    new_list = [x + 24 for x in activation_time]
    activation_time = set(activation_time)| set(new_list)
    new_list = [x + 24 for x in new_list]
    activation_time = set(activation_time)| set(new_list)
    return list(activation_time)

def activation_time_step(config):
    CommunityList = ['Education', 'Politics', 'Technology', 'Sports', 'Entertainment', 'Business']
    Count = {}
    concatfile = config['paths']['human_mbot_concatfile']
    with open(concatfile, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    results = {}
    for userdata_key in datas.keys():
        if datas[userdata_key]['label'] == 'Human':
            datas[userdata_key]['activation_time'] = random_select_time_step(datas[userdata_key]['activation_time'], datas[userdata_key]['sharetimes'])
            datas[userdata_key]['activation_time'] = three_days(datas[userdata_key]['activation_time'])
            results[userdata_key] = datas[userdata_key]
        
        if datas[userdata_key]['label'] == 'MBot':
            for community in CommunityList:
                if community == datas[userdata_key]['interest_community'][0]:
                    if community not in Count.keys():
                        Count[community] = []
                    Count[community].append(userdata_key)

    for community in CommunityList:
        count = 0
        for userdata_key in Count[community]:
            if 1 in datas[userdata_key]['activation_time']:
                count = count + 1
            results[userdata_key] = datas[userdata_key]
        if count < config['Parameters']['BeginNodeNumber']:
            # if count == 0:
            #     for k in range(config['Parameters']['BeginNodeNumber']):
            #         results[Count[community][k]]['activation_time'].append(1)
            #         results[Count[community][k]]['sharetimes'] = results[Count[community][0]]['sharetimes'] + 1                    
            # if count < config['Parameters']['BeginNodeNumber']:
            for k in range(count, config['Parameters']['BeginNodeNumber']):
                for userdata_key in Count[community]:
                    if 1 not in datas[userdata_key]['activation_time']:
                        results[userdata_key]['activation_time'].append(1)
                        results[userdata_key]['sharetimes'] = results[userdata_key]['sharetimes'] + 1
                        break
            
    # save the datas
    filename = config['newpaths']['HumanMBotData']
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    config = load_config("DisseminationNetwork/config.yaml")
    activation_time_step(config)

