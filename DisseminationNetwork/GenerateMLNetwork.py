# Disinformation Dissemination Network, Algorithm 1
import json
from utils import load_config, set_logger, convert
from tqdm import tqdm
# read SI_Json
import numpy as np
import random

def SBModel(config, logger):
    """
        SBModel: Stochastic Block Model, Node Community Assignment
    """
    Entertainment_community = []
    Education_community = []
    Sports_community = []
    Business_community = []
    Politics_community = []
    Technology_community = []
    users_id_dict = {}
    logger.info("Start Stochastic Block Model")
    logger.info("Load User Data from ConcatFile")
    with open(config['newpaths']['ConcatHumanMBotLBot'], 'r', encoding='utf-8') as f:
        user_data_batch = json.load(f)
    # shuffle
    items = list(user_data_batch.items())
    random.shuffle(items)

    # 构造一个新的字典
    user_data_batch = dict(items)
    for user_data_key in user_data_batch.keys():
        user_data = user_data_batch[user_data_key]
        community = user_data['interest_community']
        if 'Education' in community:
            Education_community.append(user_data['id'])
            users_id_dict[str(user_data['id'])] = user_data['user_id']
        if 'Politics' in community:
            Politics_community.append(user_data['id'])
            users_id_dict[str(user_data['id'])] = user_data['user_id']
        if 'Technology' in community:
            Technology_community.append(user_data['id'])
            users_id_dict[str(user_data['id'])] = user_data['user_id']
        if 'Sports' in community:
            Sports_community.append(user_data['id'])
            users_id_dict[str(user_data['id'])] = user_data['user_id']
        if 'Entertainment' in community:
            Entertainment_community.append(user_data['id'])
            users_id_dict[str(user_data['id'])] = user_data['user_id']
        if 'Business' in community:
            Business_community.append(user_data['id'])
            users_id_dict[str(user_data['id'])] = user_data['user_id']
 

    logger.info("Finish Stochastic Block Model")
    logger.info("Return Community Data")
    return list(set(Education_community)), list(set(Politics_community)), list(set(Technology_community)), list(set(Sports_community)), list(set(Entertainment_community)), list(set(Business_community)), users_id_dict


def update_si_ratio(socialinfluence):
    socialinfluence_ratio = []
    for fc in socialinfluence:
            socialinfluence_ratio.append(fc/sum(socialinfluence))
    return np.array(socialinfluence_ratio)


def to_undirected_no_self_loops(edge_index):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    reversed_edges = np.stack([edge_index[1], edge_index[0]], axis=0)
    edge_index = np.concatenate([edge_index, reversed_edges], axis=1)
    edge_index = np.unique(edge_index, axis=1)
    return edge_index

def BAModel(config, users_id_dict, Education_community, Politics_community, Technology_community, Sport_community, Entertainment_community, Business_community, logger):
    """
        BAModel: Add edges between nodes in the same community
    """
    logger.info("Start BAModel")
    # 打乱List的顺序
    random.shuffle(Education_community)
    random.shuffle(Politics_community)
    random.shuffle(Technology_community)
    random.shuffle(Sport_community)
    random.shuffle(Entertainment_community)
    random.shuffle(Business_community)
    # 1. concat users
    community_lists = [Education_community, Politics_community, Technology_community, Sport_community, Entertainment_community, Business_community]

    commnities = ['Education', 'Politics', 'Technology', 'Sports', 'Entertainment', 'Business']
    All_edge_index = []
    # m nodes: completed edges between m nodes
    m = config['parameters']['m']
    # add m1 edges
    m1 = config['parameters']['m1']

    edge_index_file = config['MBotLBotPaths']['MBotLBotEdgeIndex'] + "mbot_human_lbot_edge_index.npy"
    with open(config['newpaths']['ConcatHumanMBotLBot'], 'r', encoding='utf-8') as f:
        user_data_batch = json.load(f)
    
    comm = 0
    for community_list in tqdm(community_lists):
        edge_index = []
        socialinfluence = []
        socialinfluence_ratio = []    
        comm = comm + 1
        beginnodes = []
        logger.info("Start BAModel: Add m nodes")
        # 1. add m nodes
        for i in range(m):
            id = community_list[i]
            followers_count = user_data_batch[users_id_dict[str(id)]]['social_influence']
            beginnodes.append(id)
            socialinfluence.append(followers_count)
        logger.info("m nodes completed")
        # 2. add m! edges
        for n in beginnodes:
            for k in beginnodes:
                if n != k:
                    edge_index.append([n,k])
                    All_edge_index.append([n,k])

        # 3. process remaining nodes
        logger.info("Start BAModel: Process remaining nodes")
        for i in range(m, len(community_list)):
            id = community_list[i]
            # 3.1select nodes
            socialinfluence_ratio = update_si_ratio(socialinfluence)
            socialinfluence_ratio = socialinfluence_ratio / socialinfluence_ratio.sum()
            # generate random number(int, 1-4)
            
            random_number = np.random.randint(1, m1)
            # choose nodes
            chosen_nodes = np.random.choice(beginnodes, size=random_number, replace=False, p=socialinfluence_ratio)
            
            # 3.2add node
            beginnodes.append(id)

            # 3.3 add edges
            for n in chosen_nodes:
                edge_index.append([n,id])
                All_edge_index.append([n,id])

            # update socialinfluence and socialinfluence_ratio
            followers_count = user_data_batch[users_id_dict[str(id)]]['social_influence']
            socialinfluence.append(followers_count)
        edge_index = to_undirected_no_self_loops(np.array(edge_index).T)
        # save edge_index
        edge_index_community_file = config['MBotLBotPaths']['MBotLBotEdgeIndex'] + "mbot_human_lbot_edge_index_community_" +commnities[comm-1] + ".npy"
        np.save(edge_index_community_file, edge_index)

    All_edge_index = to_undirected_no_self_loops(np.array(All_edge_index).T)
    np.save(edge_index_file, All_edge_index)
    logger.info("Finish BAModel")
    logger.info("Save edge_index to file")
    return All_edge_index

def construct_network():
    # 1. load config
    config = load_config("CorrectExperiment/GenerateMLNetwork/config.yaml")
    logger = set_logger(config['log']['construct_network'])
    # 2. Stochastic Block: Node Community Assignment
    Education_community, Politics_community, Technology_community, Sport_community, Entertainment_community, Business_community, users_id_dict = SBModel(config, logger)
    # 3. BAModel
    All_edge_index = BAModel(config, users_id_dict, Education_community, Politics_community, Technology_community, Sport_community, Entertainment_community, Business_community, logger)

if __name__ == '__main__':
    construct_network()