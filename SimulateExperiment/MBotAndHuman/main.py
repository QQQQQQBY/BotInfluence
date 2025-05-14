import numpy as np
import yaml
from pathlib import Path
from dissemination import DisseminationOptimizer, Config
from history_memory import HistoryMemoryManager, TimeStepManager
import time
from tqdm import tqdm
import json

def load_config(filename: str) -> dict:
    """加载配置文件"""
    with open(filename, "r", encoding='utf-8') as file:
        return yaml.safe_load(file)

def main():
    # 1. 加载配置
    config_dict = load_config("SimulateExperiment/MBotAndHuman/config.yaml")
    config = Config(
        paths=config_dict['paths'],
        newpaths=config_dict['newpaths'],
        History = config_dict['History'],
        parameters=config_dict['Parameters'],
        disinfopaths=config_dict['disinfopaths'],
        promptpaths=config_dict['promptpaths'], 
        log=config_dict['log'],
        disinfo_claim=config_dict['DisinfoClaim']
    )

    # 2. 初始化各个组件
    print("初始化系统组件...")
    history_manager = HistoryMemoryManager(config_dict)
    time_step_manager = TimeStepManager(config_dict)

    # 3. 初始化历史数据
    print("初始化历史数据...")
    history_manager.initialize_history()
    

    # 4. 加载必要数据
    print("加载用户数据...")
    mbot_ids = set( np.load(config.paths['mbot_ids_Politics']).tolist())
    human_ids = set(np.load(config.paths['human_ids_Politics']).tolist())
    lbot_ids = set(np.load(config.paths['lbot_ids_Politics']).tolist())
    human_num = len(human_ids)
    mbot_num = len(mbot_ids)
    lbot_num = len(lbot_ids)
    edge_index = np.load(config.paths['edge_index_Politics'])
    lbot_ids = set( np.load(config.paths['lbot_ids_Politics']).tolist())

    # 5. 初始化追踪数据结构
    # topics = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
    topics = ["Politics"]
    time_steps = range(1, 73)
    
    if Path(config.newpaths['PoliticsSimulationResult']).exists():
        tracking = yaml.safe_load(Path(config.newpaths['PoliticsSimulationResult']).read_text())
    else:
        tracking = {
            'SusceptibleUsers': {i: {topic: [] for topic in topics} for i in time_steps},
            'ExposedUsers': {i: {topic: [] for topic in topics} for i in time_steps},
            'InfectedSpreaders': {i: {topic: [] for topic in topics} for i in time_steps},
            'UninfectedSpreaders': {i: {topic: [] for topic in topics} for i in time_steps},
            # 'RestrictedUsers': {i: {topic: [] for topic in topics} for i in time_steps},
            'TotalTokens': {i: {topic: 0 for topic in topics} for i in time_steps}
    }

    # 6. 开始模拟
    print("开始信息传播模拟...")
    start_time = time.time()
    Infected_Politics_ids = []
    Uninfected_Politics_ids = []
    for user in tracking['InfectedSpreaders'].values():
        Infected_Politics_ids.extend(user['Politics'])
    for user in tracking['UninfectedSpreaders'].values():
        Uninfected_Politics_ids.extend(user['Politics'])
    share_nodes_record = set(Infected_Politics_ids) | set(Uninfected_Politics_ids)
    share_nodes_record = set(map(int, share_nodes_record))
     # 1.1 加载所有人类易感节点
    tracking['SusceptibleUsers'][0] = {}
    tracking['SusceptibleUsers'][0]["Politics"] = list(human_ids)


    for topic in tqdm(topics, desc="处理话题"):
        print(f"\n开始处理 {topic} 话题...")
        disinfo_claim = config.disinfo_claim[topic]
        for i in tqdm(time_steps, desc="时间步骤"):
            dissemination = DisseminationOptimizer(config)
            
            # 1. 第一个时间步特殊处理
            if i == 1:
               
                # 1.2 激活指定时间步的机器人节点，第一个时间步的机器人节点需要属于当前话题所在的社区，第一个时间步只能激活机器人节点
                mbot_nodes = set(time_step_manager.mbot_time_step[topic][i])
                lbot_nodes = set()
                bot_nodes = mbot_nodes | lbot_nodes
                # 1.3 获取邻居节点
                exposed_nodes = dissemination.load_neighbour_nodes(bot_nodes, edge_index)
                
                # 1.4 运行主要传播过程
                exposed_human_nodes, trust_nodes, unbelieve_nodes, total_tokens = dissemination.DWC_main(
                    topic=topic,
                    exposed_nodes=exposed_nodes,
                    center_nodes=bot_nodes,
                    index=i,
                    mbot_ids=mbot_ids,
                    lbot_ids=lbot_ids,
                    human_ids=human_ids,
                    disinfo=disinfo_claim,
                    human_num=human_num,
                    mbot_num=mbot_num,
                    lbot_num=lbot_num,
                    correctstrategy=config.parameters['correctstrategy']
                )
                
            else:
                # 2. 加载激活的节点
                # 2.1 加载机器人节点，机器人节点需要属于当前社区
                mbot_nodes = set(time_step_manager.mbot_time_step[topic][i])
                lbot_nodes = set()
                bot_nodes = mbot_nodes | lbot_nodes

                share_nodes = set(time_step_manager.human_time_step[i])
                center_nodes = bot_nodes | share_nodes_record.intersection(share_nodes)
                # 2.4 获取邻居节点，即将暴露于虚假信息的节点
                exposed_nodes = dissemination.load_neighbour_nodes(center_nodes, edge_index)
                
                # 2.5 运行主要传播过程
                exposed_human_nodes, trust_nodes, unbelieve_nodes, total_tokens = dissemination.DWC_main(
                    topic=topic,  
                    exposed_nodes=exposed_nodes,
                    center_nodes=center_nodes,
                    index=i,
                    mbot_ids=mbot_ids,
                    lbot_ids=lbot_ids,
                    human_ids=human_ids,
                    disinfo=config.disinfo_claim[topic],
                    human_num=human_num,
                    mbot_num=mbot_num,
                    lbot_num=lbot_num,
                    correctstrategy=config.parameters['correctstrategy']
                )
            share_nodes_record = set(share_nodes_record | set(trust_nodes)|set(unbelieve_nodes))
            # 3. 更新追踪数据
            tracking['ExposedUsers'][i][topic] = [str(i) for i in exposed_human_nodes]
            tracking['InfectedSpreaders'][i][topic] = [str(i) for i in trust_nodes]
            tracking['UninfectedSpreaders'][i][topic] = [str(i) for i in unbelieve_nodes]
            # tracking['RestrictedUsers'][i][topic] = [str(i) for i in uninterested_nodes]
            tracking['TotalTokens'][i][topic] = total_tokens
            # if i > 1:
            prev_susceptible = set(tracking['SusceptibleUsers'][i-1][topic])
            tracking['SusceptibleUsers'][i][topic] = [str(i) for i in list(prev_susceptible - set(exposed_human_nodes))]     

            # 4. 保存追踪数据
            results_file = config.newpaths['PoliticsSimulationResult']
            with open(results_file, 'w', encoding='utf-8') as f:
                yaml.dump(tracking, f)
    # 7. 输出统计信息
    end_time = time.time()
    print(f"\n模拟完成！")
    print(f"总运行时间: {end_time - start_time:.2f} 秒")
 
    # 8. 保存结果
    results_file = config.newpaths['PoliticsSimulationResult']
    with open(results_file, 'w', encoding='utf-8') as f:
        yaml.dump(tracking, f)

if __name__ == "__main__":
    main() 

# 后台运行
# nohup python CorrectExperiment/WOLBot/main.py > CorrectExperiment/WOLBot/Nohuplog/Politics_main.log 2>&1 &

