import numpy as np
import json
def generate_community_id():
    with open("Dataset/CorrectDataset/NetworkDataset/ConcatHumanMBotLBot/concat_human_mbot_lbot.json", 'r', encoding='utf-8') as f:
        user_data_batch = json.load(f)
    Education_community_human = []
    Education_community_mbot = []
    Education_community_lbot = []
    Politics_community_human = []
    Politics_community_mbot = []
    Politics_community_lbot = []
    Technology_community_human = []
    Technology_community_mbot = []
    Technology_community_lbot = []
    Sports_community_human = []
    Sports_community_mbot = []
    Sports_community_lbot = []
    Entertainment_community_human = []
    Entertainment_community_mbot = []
    Entertainment_community_lbot = []
    Business_community_human = []
    Business_community_mbot = []
    Business_community_lbot = []
    for user_data_key in user_data_batch.keys():
        user_data = user_data_batch[user_data_key]
        community = user_data['interest_community']
        if 'Education' in community:
            if user_data['label'] == 'Human':
                Education_community_human.append(user_data['id'])
            elif user_data['label'] == 'MBot':
                Education_community_mbot.append(user_data['id'])
            elif user_data['label'] == 'LBot':
                Education_community_lbot.append(user_data['id'])
        if 'Politics' in community:
            if user_data['label'] == 'Human':
                Politics_community_human.append(user_data['id'])
            elif user_data['label'] == 'MBot':
                Politics_community_mbot.append(user_data['id'])
            elif user_data['label'] == 'LBot':
                Politics_community_lbot.append(user_data['id'])
        if 'Technology' in community:
            if user_data['label'] == 'Human':
                Technology_community_human.append(user_data['id'])
            elif user_data['label'] == 'MBot':
                Technology_community_mbot.append(user_data['id'])
            elif user_data['label'] == 'LBot':
                Technology_community_lbot.append(user_data['id'])
        if 'Sports' in community:
            if user_data['label'] == 'Human':
                Sports_community_human.append(user_data['id'])
            elif user_data['label'] == 'MBot':
                Sports_community_mbot.append(user_data['id'])
            elif user_data['label'] == 'LBot':
                Sports_community_lbot.append(user_data['id'])
        if 'Entertainment' in community:
            if user_data['label'] == 'Human':
                Entertainment_community_human.append(user_data['id'])
            elif user_data['label'] == 'MBot':
                Entertainment_community_mbot.append(user_data['id'])
            elif user_data['label'] == 'LBot':
                Entertainment_community_lbot.append(user_data['id'])
        if 'Business' in community:
            if user_data['label'] == 'Human':
                Business_community_human.append(user_data['id'])
            elif user_data['label'] == 'MBot':
                Business_community_mbot.append(user_data['id'])
            elif user_data['label'] == 'LBot':
                Business_community_lbot.append(user_data['id'])
    # 保存为npy文件
    np.save("Dataset/NetworkDataset/CommunityID/Education_community_human.npy", np.array(Education_community_human))
    np.save("Dataset/NetworkDataset/CommunityID/Education_community_mbot.npy", np.array(Education_community_mbot))
    np.save("Dataset/NetworkDataset/CommunityID/Education_community_lbot.npy", np.array(Education_community_lbot))

    np.save("Dataset/NetworkDataset/CommunityID/Politics_community_human.npy", np.array(Politics_community_human))
    np.save("Dataset/NetworkDataset/CommunityID/Politics_community_mbot.npy", np.array(Politics_community_mbot))
    np.save("Dataset/NetworkDataset/CommunityID/Politics_community_lbot.npy", np.array(Politics_community_lbot))

    np.save("Dataset/NetworkDataset/CommunityID/Technology_community_human.npy", np.array(Technology_community_human))
    np.save("Dataset/NetworkDataset/CommunityID/Technology_community_mbot.npy", np.array(Technology_community_mbot))
    np.save("Dataset/NetworkDataset/CommunityID/Technology_community_lbot.npy", np.array(Technology_community_lbot))

    np.save("Dataset/NetworkDataset/CommunityID/Sports_community_human.npy", np.array(Sports_community_human))
    np.save("Dataset/NetworkDataset/CommunityID/Sports_community_mbot.npy", np.array(Sports_community_mbot))
    np.save("Dataset/NetworkDataset/CommunityID/Sports_community_lbot.npy", np.array(Sports_community_lbot))

    np.save("Dataset/NetworkDataset/CommunityID/Entertainment_community_human.npy", np.array(Entertainment_community_human))
    np.save("Dataset/NetworkDataset/CommunityID/Entertainment_community_mbot.npy", np.array(Entertainment_community_mbot))
    np.save("Dataset/NetworkDataset/CommunityID/Entertainment_community_lbot.npy", np.array(Entertainment_community_lbot))

    np.save("Dataset/NetworkDataset/CommunityID/Business_community_human.npy", Business_community_human)
    np.save("Dataset/NetworkDataset/CommunityID/Business_community_mbot.npy", np.array(Business_community_mbot))
    np.save("Dataset/NetworkDataset/CommunityID/Business_community_lbot.npy", np.array(Business_community_lbot))

    # 输出每个社区每个用户类型的数量
    print(f"Education_community_human: {len(Education_community_human)}")
    print(f"Education_community_mbot: {len(Education_community_mbot)}")
    print(f"Education_community_lbot: {len(Education_community_lbot)}")
    print(f"Politics_community_human: {len(Politics_community_human)}")
    print(f"Politics_community_mbot: {len(Politics_community_mbot)}")
    print(f"Politics_community_lbot: {len(Politics_community_lbot)}")
    print(f"Technology_community_human: {len(Technology_community_human)}")
    print(f"Technology_community_mbot: {len(Technology_community_mbot)}")
    print(f"Technology_community_lbot: {len(Technology_community_lbot)}")
    print(f"Sports_community_human: {len(Sports_community_human)}")
    print(f"Sports_community_mbot: {len(Sports_community_mbot)}")
    print(f"Sports_community_lbot: {len(Sports_community_lbot)}")
    
    
if __name__ == "__main__":
    generate_community_id()