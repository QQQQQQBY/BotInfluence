# build bot attribute
# bot 
import json
from utils import load_config, convert
import random
import numpy as np
from tqdm import tqdm
from set_logger import set_logger

def random_user_id(MBot_id_list):
  random_number = ''.join([str(random.randint(0, 9)) for _ in range(18)])
  if random_number not in MBot_id_list:
    return random_number
  else:
    random_user_id(MBot_id_list)

def random_socialinfluence(RegularUserSocialInfluenceFile):
  with open(RegularUserSocialInfluenceFile, 'r', encoding='utf-8') as f:
    RegularUserSocialInfluenceDatas = json.load(f)
  SocialInfluenceList = []
  for si_key in RegularUserSocialInfluenceDatas.keys():
    SocialInfluenceList.append(RegularUserSocialInfluenceDatas[si_key]["social_influence"])
  # random select one social influence from the list 1/3-2/3
  tertiles = np.percentile(SocialInfluenceList, [100/10, 200/5])
  return random.randint(int(tertiles[0]), int(tertiles[1]))
  
def generate_activation_time(config, ActivationTimes, count):
  np.random.seed(count)
  numbers_1 = list(range(config['parameters']['numbers_1'], config['parameters']['numbers_2']))   
  numbers_2 = list(range(config['parameters']['numbers_2'], 73))  
  weights_1 = [config['parameters']['weights_1'] / len(numbers_1)] * len(numbers_1)
  weights_2 = [config['parameters']['weights_2'] / len(numbers_2)] * len(numbers_2)
  all_numbers = numbers_1 + numbers_2
  all_weights = weights_1 + weights_2
  activation_time = np.random.choice(all_numbers, size=ActivationTimes, p=all_weights, replace=False)
  return activation_time

def generate_bots(config, logger):
  # load the number of users in diff community
  Education_community = []
  Politics_community = []
  Technology_community = []
  Sports_community = []
  Entertainment_community = []
  Business_community = []
  with open(config['paths']['SampleDataset'], 'r', encoding='utf-8') as f:
    user_data_batch = json.load(f)
  for user_data_key in user_data_batch.keys():
    user_data = user_data_batch[user_data_key]
    community = user_data['interest_community']
    if 'Education' in community:
      Education_community.append(user_data['id'])
    if 'Politics' in community:
      Politics_community.append(user_data['id'])
    if 'Technology' in community:
      Technology_community.append(user_data['id'])
    if 'Sports' in community:
      Sports_community.append(user_data['id'])
    if 'Entertainment' in community:
      Entertainment_community.append(user_data['id'])
    if 'Business' in community:
      Business_community.append(user_data['id'])
  

  # calculate the number of deploying Mbots 
  EntertainmentBotNum = round(config['MBotRatio']['EntertainmentBotRatio'] * len(Entertainment_community))
  EducationBotNum = round(config['MBotRatio']['EducationBotRatio'] * len(Education_community))
  BusinessBotNum = round(config['MBotRatio']['BusinessBotRatio'] * len(Business_community))
  SportsBotNum = round(config['MBotRatio']['SportsBotRatio'] * len(Sports_community))
  PoliticsBotNum = round(config['MBotRatio']['PoliticsBotRatio'] * len(Politics_community))
  TechnologyBotNum = round(config['MBotRatio']['TechnologyBotRatio'] * len(Technology_community))

  MBotNum = {'EntertainmentBotNum': EntertainmentBotNum, 'EducationBotNum': EducationBotNum, 'BusinessBotNum': BusinessBotNum, 'SportsBotNum': SportsBotNum, 'PoliticsBotNum': PoliticsBotNum, 'TechnologyBotNum':TechnologyBotNum} 

  # generate five attributes
  # 1. activation times
  MaxActivationTimes = config['parameters']['MaxActivationTimes']
  MBots = {}
  MBot_id_list = []
  count = len(user_data_batch)
  for num_key in tqdm(list(MBotNum.keys())):
    for i in range(MBotNum[num_key]):
      user_id = random_user_id(MBot_id_list)
      MBot_id_list.append(user_id)
      if user_id not in MBots:
        MBots[user_id] = {}      
        MBots[user_id]['id'] = count
        count = count + 1
        # 2. generate user_id
        MBots[user_id]['user_id'] = user_id
        MBots[user_id]['community'] = {}
        # 3. generate community
        MBots[user_id]['interest_community'] = [num_key.split('BotNum')[0]]
        # 4. generate activation time
        np.random.seed(count) 
        ActivationTimes = random.randint(1, MaxActivationTimes)
        ActivationTime = generate_activation_time(config, ActivationTimes, count)     
        MBots[user_id]['activation_time'] = ActivationTime.tolist()
        MBots[user_id]['sharetimes'] = ActivationTimes
        # 5. generate trust threshold
        TrustThreshold = {
                  "Entertainment": 0,
                  "Technology": 0,
                  "Sports": 0,
                  "Business": 0,
                  "Politics": 0,
                  "Education": 0
              }            
        MBots[user_id]['trust_threshold'] = TrustThreshold
        # 6. generate dissemination tendency
        DisseminationTendency = {
              "Entertainment": 1,
              "Technology": 1,
              "Sports": 1,
              "Business": 1,
              "Education": 1,
              "Politics": 1
          }
        MBots[user_id]['dissemination_tendency'] = DisseminationTendency
        # 7. generate social influence
        RegularUserSocialInfluenceFile = config['paths']['SampleDataset']
        SocialInfluence = random_socialinfluence(RegularUserSocialInfluenceFile)
        MBots[user_id]['social_influence'] = SocialInfluence
        MBots[user_id]['label'] = 'MBot'

  output_file = config['newpaths']['MBotDataset']
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(MBots, f, indent=4, ensure_ascii=False)

def generate_mbot_data():
  config = load_config("DisseminationNetwork/config.yaml")
  logger = set_logger(config['log']['construct_network'])
  generate_bots(config, logger)


if __name__ == "__main__":
  generate_mbot_data()
