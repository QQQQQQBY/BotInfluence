import numpy as np
import yaml
from pathlib import Path
from optimized_disseminate import DisseminationOptimizer, Config
from optimized_history_memory import HistoryMemoryManager
import time
from tqdm import tqdm

def load_config(filename: str) -> dict:
    """加载配置文件"""
    with open(filename, "r", encoding='utf-8') as file:
        return yaml.safe_load(file)

def main():
    # 1. 加载配置
    config_dict = load_config("Experiment/config.yaml")
    config = Config(
        paths=config_dict['paths'],
        newpaths=config_dict['newpaths'],
        parameters=config_dict['Parameters'],
        disinfopaths=config_dict['disinfopaths'],
        log=config_dict['log'],
        disinfo_claim=config_dict['DisinfoClaim']
    )

    # 2. 初始化各个组件
    print("初始化系统组件...")
    dissemination = DisseminationOptimizer(config)
    history_manager = HistoryMemoryManager(config_dict)

    # 3. 初始化历史数据
    print("初始化历史数据...")
    history_manager.initialize_history()

    # 4. 加载必要数据
    print("加载用户数据...")
    bot_ids = set(np.load(config.paths['botidsfile']).tolist())
    human_ids = set(np.load(config.paths['humanidsfile']).tolist())
    human_num = len(human_ids)
    edge_index = np.load(config.paths['edge_index'])
    
    # 5. 初始化追踪数据结构
    communities = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
    time_steps = range(1, 25)
    
    tracking = {
        'SusceptibleUsers': {i: {comm: [] for comm in communities} for i in time_steps},
        'ExposedUsers': {i: {comm: [] for comm in communities} for i in time_steps},
        'InfectedSpreaders': {i: {comm: [] for comm in communities} for i in time_steps},
        'UninfectedSpreaders': {i: {comm: [] for comm in communities} for i in time_steps},
        'RestrictedUsers': {i: {comm: [] for comm in communities} for i in time_steps}
    }

    # 6. 开始模拟
    print("开始信息传播模拟...")
    start_time = time.time()

    for comm in tqdm(communities, desc="处理社区"):
        print(f"\n开始处理 {comm} 社区...")
        disinfo_claim = config.disinfo_claim[comm]
        for i in tqdm(time_steps, desc="时间步骤"):
            # 1. 第一个时间步特殊处理
            if i == 1:
                # 1.1 加载社区节点
                community_file = config.paths[comm]
                tracking['SusceptibleUsers'][i][comm] = history_manager._load_community_nodes(community_file, human_num)
                
                # 1.2 激活指定时间步的机器人节点，第一个时间步的机器人节点需要属于当前社区，第一个时间步只能激活机器人节点
                bot_nodes = set()
                with open(config.paths['botfile'], 'r') as f:
                    bots_data = history_manager._load_json(config.paths['botfile'])
                    for bot in bots_data:
                        if i in bot.get('activation time', []) and comm in bot.get('community', []):
                            bot_nodes.add(bot['id'])

                # 1.3 获取邻居节点
                exposed_nodes = dissemination.load_neighbour_nodes(bot_nodes, edge_index)
                
                # 1.4 运行主要传播过程
                exposed_human_nodes, trust_nodes, unbelieve_nodes, uninterested_nodes = dissemination.DWC_main(
                    comm=comm,
                    exposed_nodes=exposed_nodes,
                    center_nodes=bot_nodes,
                    index=i,
                    bot_ids=bot_ids,
                    human_ids=human_ids,
                    disinfo=disinfo_claim,
                )

            else:
                # 2. 加载激活的节点
                bot_nodes = set()
                human_nodes = set()
                
                # 2.1 加载机器人和人类节点
                bots_data = history_manager._load_json(config.paths['botfile'])
                humans_data = history_manager._load_json(config.paths['humanfile'])
                
                # 2.2 加载激活的机器人节点，机器人节点需要属于当前社区
                for bot in bots_data:
                    if i in bot.get('activation time', []) and comm in bot.get('community', []):
                        bot_nodes.add(bot['id'])
                
                # 2.3 加载激活的人类节点，人类节点可以不属于当前社区
                for human in humans_data:
                    if i in human.get('activation time', []):
                        human_nodes.add(human['id'])

                # 2.4 合并中心节点
                center_nodes = bot_nodes | human_nodes
                
                # 2.5 获取邻居节点，即将暴露于虚假信息的节点
                exposed_nodes = dissemination.load_neighbour_nodes(center_nodes, edge_index)
                
                # 2.6 运行主要传播过程
                exposed_human_nodes, trust_nodes, unbelieve_nodes, uninterested_nodes = dissemination.DWC_main(
                    comm=comm,
                    exposed_nodes=exposed_nodes,
                    center_nodes=center_nodes,
                    index=i,
                    bot_ids=bot_ids,
                    human_ids=human_ids,
                    disinfo=config.disinfo_claim[comm]   # 这里需要根据实际情况设置
                )

            # 3. 更新追踪数据
            tracking['ExposedUsers'][i][comm] = exposed_human_nodes
            tracking['InfectedSpreaders'][i][comm] = trust_nodes
            tracking['UninfectedSpreaders'][i][comm] = unbelieve_nodes
            tracking['RestrictedUsers'][i][comm] = uninterested_nodes
            
            # 更新易感用户
            if i > 1:
                prev_susceptible = set(tracking['SusceptibleUsers'][i-1][comm])
                tracking['SusceptibleUsers'][i][comm] = list(prev_susceptible - set(exposed_human_nodes))

    # 7. 输出统计信息
    end_time = time.time()
    print(f"\n模拟完成！")
    print(f"总运行时间: {end_time - start_time:.2f} 秒")

    # 8. 保存结果（可选）
    results_file = Path("simulation_results.yaml")
    with open(results_file, 'w', encoding='utf-8') as f:
        yaml.dump(tracking, f)

if __name__ == "__main__":
    main() 