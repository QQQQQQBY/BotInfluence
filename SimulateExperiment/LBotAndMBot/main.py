import numpy as np
import yaml
from pathlib import Path
from dissemination import DisseminationOptimizer, Config
from history_memory import HistoryMemoryManager, TimeStepManager
import time
from tqdm import tqdm
import json

def load_config(filename: str) -> dict:
    """Load configuration file"""
    with open(filename, "r", encoding='utf-8') as file:
        return yaml.safe_load(file)

def main():
    # 1. 加载配置
    config_dict = load_config("SimulateExperiment/LBotAndMBot/config.yaml")
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

    # 2. Initialize the components
    print("Initializing system components...")
    history_manager = HistoryMemoryManager(config_dict)
    time_step_manager = TimeStepManager(config_dict)

    # 3. Initialize the historical data
    print("Initializing historical data...")
    history_manager.initialize_history()
    

    # 4. Load necessary data
    print("Loading user data...")
    mbot_ids = set( np.load(config.paths['mbot_ids_Politics']).tolist())
    human_ids = set(np.load(config.paths['human_ids_Politics']).tolist())
    lbot_ids = set(np.load(config.paths['lbot_ids_Politics']).tolist())
    human_num = len(human_ids)
    mbot_num = len(mbot_ids)
    lbot_num = len(lbot_ids)
    edge_index = np.load(config.paths['edge_index_Politics'])
    lbot_ids = set( np.load(config.paths['lbot_ids_Politics']).tolist())

    # 5. Initialize the tracking data structure
    # topics = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
    topics = ["Politics"]
    time_steps = range(12, 73)
    
    if Path(config.newpaths['PoliticsSimulationResult']).exists():
        tracking = yaml.safe_load(Path(config.newpaths['PoliticsSimulationResult']).read_text())
    else:
        tracking = {
            'SusceptibleUsers': {i: {topic: [] for topic in topics} for i in time_steps},
            'ExposedUsers': {i: {topic: [] for topic in topics} for i in time_steps},
            'InfectedSpreaders': {i: {topic: [] for topic in topics} for i in time_steps},
            'UninfectedSpreaders': {i: {topic: [] for topic in topics} for i in time_steps},
            'TotalTokens': {i: {topic: 0 for topic in topics} for i in time_steps}
    }

    # 6. 开始模拟
    print("Starting information dissemination simulation...")
    start_time = time.time()
    Infected_Politics_ids = []
    Uninfected_Politics_ids = []
    for user in tracking['InfectedSpreaders'].values():
        Infected_Politics_ids.extend(user['Politics'])
    for user in tracking['UninfectedSpreaders'].values():
        Uninfected_Politics_ids.extend(user['Politics'])
    share_nodes_record = set(Infected_Politics_ids) | set(Uninfected_Politics_ids)
    share_nodes_record = set(map(int, share_nodes_record))
     # 1.1 Load all human susceptible nodes
    tracking['SusceptibleUsers'][0] = {}
    tracking['SusceptibleUsers'][0]["Politics"] = list(human_ids)


    for topic in tqdm(topics, desc="Processing topics"):
        print(f"\nStarting to process {topic} topic...")
        disinfo_claim = config.disinfo_claim[topic]
        for i in tqdm(time_steps, desc="Time steps"):
            dissemination = DisseminationOptimizer(config)
            
            # 1. The first time step is special
            if i == 1:
               
                # 1.2 Activate the robot nodes at the specified time step, the robot nodes at the first time step need to belong to the community of the current topic, and the robot nodes at the first time step can only be activated
                mbot_nodes = set(time_step_manager.mbot_time_step[topic][i])
                lbot_nodes = set(time_step_manager.lbot_time_step[topic][i])
                bot_nodes = mbot_nodes | lbot_nodes
                # 1.3 Get neighbor nodes
                exposed_nodes = dissemination.load_neighbour_nodes(bot_nodes, edge_index)
                
                # 1.4 Run the main dissemination process
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
                # 2. Load the activated nodes
                # 2.1 Load the robot nodes, the robot nodes need to belong to the current community
                mbot_nodes = set(time_step_manager.mbot_time_step[topic][i])
                lbot_nodes = set(time_step_manager.lbot_time_step[topic][i])
                bot_nodes = mbot_nodes | lbot_nodes

                share_nodes = set(time_step_manager.human_time_step[i])
                center_nodes = bot_nodes | share_nodes_record.intersection(share_nodes)
                # 2.4 Get neighbor nodes,That is, the nodes exposed to false information
                exposed_nodes = dissemination.load_neighbour_nodes(center_nodes, edge_index)
                
                # 2.5 Run the main dissemination process
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
            # 3. Update tracking data
            tracking['ExposedUsers'][i][topic] = [str(i) for i in exposed_human_nodes]
            tracking['InfectedSpreaders'][i][topic] = [str(i) for i in trust_nodes]
            tracking['UninfectedSpreaders'][i][topic] = [str(i) for i in unbelieve_nodes]
            tracking['TotalTokens'][i][topic] = total_tokens
            # if i > 1:
            prev_susceptible = set(tracking['SusceptibleUsers'][i-1][topic])
            tracking['SusceptibleUsers'][i][topic] = [str(i) for i in list(prev_susceptible - set(exposed_human_nodes))]     

            # 4. Save tracking data
            results_file = config.newpaths['PoliticsSimulationResult']
            with open(results_file, 'w', encoding='utf-8') as f:
                yaml.dump(tracking, f)
    # 7. Output statistical information
    end_time = time.time()
    print(f"\nSimulation completed!")
    print(f"Total running time: {end_time - start_time:.2f} seconds")
 
    # 8. Save results
    results_file = config.newpaths['PoliticsSimulationResult']
    with open(results_file, 'w', encoding='utf-8') as f:
        yaml.dump(tracking, f)

if __name__ == "__main__":
    main() 

# Run in the background
# nohup python CorrectExperiment/Simulate/main.py > CorrectExperiment/Simulate/Backgroundlog/Politics_main_narrative_early.log 2>&1 &

