from RandomSelectEqualNumber import sample_human_data
from Generate_Mbot import generate_mbot_data
from generate_lbot import generate_lbots_main
from Concat_MBot_Human_LBot import Concat_MBot_Human_LBot_main
from GenerateMLNetwork import construct_network
from generate_community_id import generate_community_id
from Concat_Human_MBots import Concat_Human_MBots_main
from preprocess_activation_time_step import activation_time_step

if __name__ == '__main__':
    # 1. sample human data
    sample_human_data(seed=42) # Dataset/NetworkDataset/SampleDataset/sampled_human_attribute_data.json
    # 2. generate mbot data
    generate_mbot_data() # Dataset/NetworkDataset/MBotDataset/MBotDataset.json
    # 3. generate mbot human data
    Concat_Human_MBots_main() # "Dataset/NetworkDataset/MBotHumanDataset/concat_human_mbots.json
    # 4. proprecess activation time
    activation_time_step() # Dataset/NetworkDataset/MBotHumanDataset/HumanMBotData.json
    # 5. generate lbot data
    generate_lbots_main() # Dataset/NetworkDataset/LBotDataset/LBotDataset.json
    # 6. concat human and mbot data
    Concat_MBot_Human_LBot_main() # Dataset/NetworkDataset/ConcatHumanMBotLBot/concat_human_mbot_lbot.json
    # 7. generate network
    construct_network() # Dataset/CorrectDataset/NetworkDataset/EdgeIndex/
    # 6. generate community id
    generate_community_id() # Dataset/NetworkDataset/CommunityID/