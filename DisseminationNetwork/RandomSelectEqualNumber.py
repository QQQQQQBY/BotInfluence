# To ensure that the number of each community is the same, we randomly select the same number of data from the dataset.
import json
import random   
import os

def dispose_data():
    Entertainment_community = []
    Education_community = []
    Sports_community = []
    Business_community = []
    Politics_community = []
    Technology_community = []
    with open('Dataset/AttributeDataset/concat_attribute_data.json', 'r', encoding='utf-8') as f:
        user_data_batch = json.load(f)
    multi_community_user_id = []
    for user_data_key in user_data_batch.keys():
        user_data = user_data_batch[user_data_key]
        community = user_data['interest_community']
        if len(community) > 1:
            multi_community_user_id.append(user_data['user_id'])
        else:
            if 'Education' in community:
                Education_community.append(user_data['user_id'])
            if 'Politics' in community:
                Politics_community.append(user_data['user_id'])
            if 'Technology' in community:
                Technology_community.append(user_data['user_id'])
            if 'Sports' in community:
                Sports_community.append(user_data['user_id'])
            if 'Entertainment' in community:
                Entertainment_community.append(user_data['user_id'])
            if 'Business' in community:
                Business_community.append(user_data['user_id'])
    min_length = min(len(Education_community), len(Politics_community), len(Technology_community), len(Sports_community), len(Entertainment_community), len(Business_community))
    Education_community = random.sample(Education_community, min_length)
    Politics_community = random.sample(Politics_community, min_length)
    Technology_community = random.sample(Technology_community, min_length)
    Sports_community = random.sample(Sports_community, min_length)
    Entertainment_community = random.sample(Entertainment_community, min_length)
    Business_community = random.sample(Business_community, min_length)
    return Education_community, Politics_community, Technology_community, Sports_community, Entertainment_community, Business_community, multi_community_user_id

def filter_data(CommunityList):
    results = {}
    count = 0
    with open('Dataset/AttributeDataset/concat_attribute_data.json', 'r', encoding='utf-8') as f:
        user_data_batch = json.load(f)
    for community in CommunityList:
        for user_id in community:
            if user_id not in results.keys():
                results[user_id] = user_data_batch[user_id]
                results[user_id]['id'] = count
                count += 1
    with open('Dataset/NetworkDataset/SampleDataset/sampled_human_attribute_data.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    Education_community, Politics_community, Technology_community, Sports_community, Entertainment_community, Business_community, multi_community_user_id = dispose_data()
    CommunityList = [Education_community, Politics_community, Technology_community, Sports_community, Entertainment_community, Business_community, multi_community_user_id]
    filter_data(CommunityList)