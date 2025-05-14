# build bot attribute
# bot 
import json
from utils import load_config, convert, set_logger
import random
import numpy as np
from tqdm import tqdm

def random_user_id(LBot_id_list):
  random_number = ''.join([str(random.randint(0, 9)) for _ in range(18)])
  if random_number not in LBot_id_list:
    return random_number
  else:
    random_user_id(LBot_id_list)

def random_socialinfluence(RegularUserSocialInfluenceFile):
  with open(RegularUserSocialInfluenceFile, 'r', encoding='utf-8') as f:
    RegularUserSocialInfluenceDatas = json.load(f)
  SocialInfluenceList = []
  for si_key in RegularUserSocialInfluenceDatas.keys():
    SocialInfluenceList.append(RegularUserSocialInfluenceDatas[si_key]["social_influence"])
  # random select one social influence from the list 1/3-2/3
  tertiles = np.percentile(SocialInfluenceList, [100/10, 200/5])
  return random.randint(int(tertiles[0]), int(tertiles[1]))


def generate_early_correction_activation_time(config, ActivationTimes, count, early_correction_time, middle_correction_time):
  np.random.seed(count)
  numbers_1 = list(range(early_correction_time, middle_correction_time))    # 12 to 36
  numbers_2 = list(range(middle_correction_time, 73))  # 37 to 73
  weights_1 = [config['parameters']['weights_1'] / len(numbers_1)] * len(numbers_1)
  weights_2 = [config['parameters']['weights_2'] / len(numbers_2)] * len(numbers_2)
  all_numbers = numbers_1 + numbers_2
  all_weights = weights_1 + weights_2
  activation_time = np.random.choice(all_numbers, size=ActivationTimes, p=all_weights, replace=False)
  return activation_time

def generate_middle_correction_activation_time(config, ActivationTimes, count,  middle_correction_time, late_correction_time):
  np.random.seed(count)
  numbers_1 = list(range(middle_correction_time, late_correction_time))    # 36 to 48
  numbers_2 = list(range(late_correction_time, 73))  # 48 to 73
  weights_1 = [config['parameters']['weights_1'] / len(numbers_1)] * len(numbers_1)
  weights_2 = [config['parameters']['weights_2'] / len(numbers_2)] * len(numbers_2)     
  all_numbers = numbers_1 + numbers_2
  all_weights = weights_1 + weights_2
  activation_time = np.random.choice(all_numbers, size=ActivationTimes, p=all_weights, replace=False)
  return activation_time

def generate_late_correction_activation_time(config, ActivationTimes, count, late_correction_time):
  np.random.seed(count)
  all_numbers = list(range(late_correction_time, 73))    # 48 to 73
  activation_time = np.random.choice(all_numbers, size=ActivationTimes, replace=False)
  return activation_time


def generate_lbots(config, logger):
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
  EntertainmentLBotNum = round(config['LBotRatio']['EntertainmentLBotRatio'] * len(Entertainment_community))
  EducationLBotNum = round(config['LBotRatio']['EducationLBotRatio'] * len(Education_community))
  BusinessLBotNum = round(config['LBotRatio']['BusinessLBotRatio'] * len(Business_community))
  SportsLBotNum = round(config['LBotRatio']['SportsLBotRatio'] * len(Sports_community))
  PoliticsLBotNum = round(config['LBotRatio']['PoliticsLBotRatio'] * len(Politics_community))
  TechnologyLBotNum = round(config['LBotRatio']['TechnologyLBotRatio'] * len(Technology_community))

  LBotNum = {'EntertainmentLBotNum': EntertainmentLBotNum, 'EducationLBotNum': EducationLBotNum, 'BusinessLBotNum': BusinessLBotNum, 'SportsLBotNum': SportsLBotNum, 'PoliticsLBotNum': PoliticsLBotNum, 'TechnologyLBotNum':TechnologyLBotNum} 

  # generate five attributes
  # 1. activation times
  MaxActivationTimes = config['parameters']['MaxActivationTimes']
  LBot = {}
  LBot_id_list = []
  with open(config['paths']['MBotHumanDataset'], 'r', encoding='utf-8') as f:
    human_mbots = json.load(f)
  count = len(human_mbots)
  for num_key in tqdm(list(LBotNum.keys())):
    for i in range(LBotNum[num_key]):
      user_id = random_user_id(LBot_id_list)
      LBot_id_list.append(user_id)
      if user_id not in LBot:
        LBot[user_id] = {}      
        LBot[user_id]['id'] = count
        count = count + 1
        # 2. generate user_id
        LBot[user_id]['user_id'] = user_id
        LBot[user_id]['community'] = {}
        # 3. generate community
        LBot[user_id]['interest_community'] = [num_key.split('LBotNum')[0]]
        # 4. generate activation time
        np.random.seed(count) 
        ActivationTimes = random.randint(5, MaxActivationTimes)
        # EarlyCorrectionTime
        early_correction_activation_time = generate_early_correction_activation_time(config, ActivationTimes, count, config['LBotCorrectionTime']['EarlyCorrectionTime'], config['LBotCorrectionTime']['MiddleCorrectionTime'])   
        # MiddleCorrectionTime
        middle_correction_activation_time = generate_middle_correction_activation_time(config, ActivationTimes, count, config['LBotCorrectionTime']['MiddleCorrectionTime'], config['LBotCorrectionTime']['LateCorrectionTime'])   
        # LateCorrectionTime
        late_correction_activation_time = generate_late_correction_activation_time(config, ActivationTimes, count, config['LBotCorrectionTime']['LateCorrectionTime'])   

        LBot[user_id]['early_correction_activation_time'] = early_correction_activation_time.tolist()
        LBot[user_id]['middle_correction_activation_time'] = middle_correction_activation_time.tolist()
        LBot[user_id]['late_correction_activation_time'] = late_correction_activation_time.tolist()
        LBot[user_id]['sharetimes'] = ActivationTimes
        # 5. generate trust threshold
        TrustThreshold = {
                  "Entertainment": 1,
                  "Technology": 1,
                  "Sports": 1,
                  "Business": 1,
                  "Politics": 1,
                  "Education": 1
              }            
        LBot[user_id]['trust_threshold'] = TrustThreshold
        # 6. generate dissemination tendency
        DisseminationTendency = {
              "Entertainment": 1,
              "Technology": 1,
              "Sports": 1,
              "Business": 1,
              "Education": 1,
              "Politics": 1
          }
        LBot[user_id]['dissemination_tendency'] = DisseminationTendency
        # 7. generate social influence
        RegularUserSocialInfluenceFile = config['paths']['SampleDataset']
        SocialInfluence = random_socialinfluence(RegularUserSocialInfluenceFile)
        LBot[user_id]['social_influence'] = SocialInfluence
        LBot[user_id]['label'] = 'LBot'

  output_file = config['newpaths']['LBotDataset']
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(LBot, f, indent=4, ensure_ascii=False)

def generate_lbots_main():
  config = load_config("DisseminationNetwork/config.yaml")
  logger = set_logger(config['log']['construct_network'])
  generate_lbots(config, logger)

if __name__ == "__main__":
  generate_lbots_main()
