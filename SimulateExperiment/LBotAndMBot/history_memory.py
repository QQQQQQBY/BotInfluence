from typing import Dict, List, Any
import json
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from collections import defaultdict
from typing import Set

class HistoryMemoryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.communities = ["Business", "Education", "Entertainment", 
                          "Politics", "Sports", "Technology"]
        self.edge_index_cache = self._load_neighbour_nodes(self.config["paths"]["edge_index_Politics"])
        self.edge_index_cache_social_influence = self._load_neighbour_nodes_social_influence(self.config["paths"]["HumanMBotLBotDataPolitics"])

    def initialize_history(self):
        """Initialize all historical data"""
        self._create_mbot_history()
        self._create_human_history()
        self._create_lbot_history()

    def _load_neighbour_nodes(self, edge_index: np.ndarray) -> Dict[int, List[int]]:
        """Get neighbor nodes"""
        edge_index = np.load(edge_index)
        neighbor_dict = defaultdict(list)
        # Traverse edges
        for src, dst in edge_index.T.tolist():  # After transpose, each row is an edge (src, dst)
            neighbor_dict[src].append(dst)
        neighbor_dict = dict(neighbor_dict)
        return neighbor_dict
    
    def _load_neighbour_nodes_social_influence(self, human_mbot_lbot_data: Dict) -> Dict[int, List[int]]:
        """Get neighbor nodes social influence"""
        with open(human_mbot_lbot_data, 'r', encoding='utf-8') as f:
            human_mbot_lbot_data = json.load(f)

        neighbor_dict = {}

        for key in human_mbot_lbot_data.keys():
            data = human_mbot_lbot_data[key]
            if data['id'] not in neighbor_dict:
                neighbor_dict[data['id']] = {}
                sum_social_influence = 0
            for node in self.edge_index_cache[data['id']]:
                neighbor_dict[data['id']][node] = len(self.edge_index_cache[node])
                sum_social_influence += len(self.edge_index_cache[node])
            for node in neighbor_dict[data['id']].keys():
                neighbor_dict[data['id']][node] = neighbor_dict[data['id']][node] / sum_social_influence
        return neighbor_dict
    
    def _create_mbot_history(self):
        """Optimized bot history creation"""
        if Path(self.config["History"]["MBotHistricalInfoPolitics"]).exists():
            return
            
        try:
            with open(self.config["paths"]["HumanMBotLBotDataPolitics"], 'r', encoding='utf-8') as f:
                bots_data = json.load(f)
                
            history_results = {}
            for bot_key in bots_data.keys():
                bot = bots_data[bot_key]
                if bot['label'] == 'MBot':
                    history_result = {
                        'id': bot['id'],
                        'user_id': bot['user_id'],
                        'social influence': bot['social_influence'],
                        'receive info': {comm: [] for comm in self.communities},
                        'share info': {comm: [] for comm in self.communities},
                        'neighbor_nodes': self.edge_index_cache[bot['id']],
                        'neighbor_nodes_social_influence': self.edge_index_cache_social_influence[bot['id']],
                    }
                    history_results[bot['id']] = history_result
                
            with open(self.config["History"]["MBotHistricalInfoPolitics"], 'w', encoding='utf-8') as f:
                json.dump(history_results, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Bot history creation failed: {str(e)}")

    def _create_lbot_history(self):
        """Optimized bot history creation"""
        if Path(self.config["History"]["LBotHistricalInfoPolitics"]).exists():
            return
        
        try:
            with open(self.config["paths"]["HumanMBotLBotDataPolitics"], 'r', encoding='utf-8') as f:
                bots_data = json.load(f)
                
            history_results = {}    
            for bot_key in bots_data.keys():
                bot = bots_data[bot_key]
                if bot['label'] == 'LBot':
                    history_result = {
                        'id': bot['id'],
                        'user_id': bot['user_id'],
                        'social influence': bot['social_influence'],    
                        'receive info': {comm: [] for comm in self.communities},
                        'share info': {comm: [] for comm in self.communities},
                        'neighbor_nodes': self.edge_index_cache[bot['id']],
                        'neighbor_nodes_social_influence': self.edge_index_cache_social_influence[bot['id']],
                    }
                    history_results[bot['id']] = history_result
                    
            with open(self.config["History"]["LBotHistricalInfoPolitics"], 'w', encoding='utf-8') as f:
                json.dump(history_results, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"LBot history creation failed: {str(e)}")

    def _create_human_history(self):
        """Optimized human user history creation"""
        if Path(self.config["History"]["HumanHistricalInfoPolitics"]).exists():
            return  # May return too early, should check if file content is complete
            
        try:
            # Add logging
            print("Starting to create human historical data...")
            
            # Serial data loading
            humans_data = self._load_json(self.config["paths"]["HumanMBotLBotDataPolitics"])
            users_text = self._load_json(self.config["paths"]["HumanSummerizeText"])
            share_nums = self._load_json(self.config["paths"]["sharenumfile"])
            
            print(f"Number of users loaded: {len(humans_data)}")
            
            history_results = {}
            for user_key in humans_data.keys():
                user = humans_data[user_key]
                if user['label'] == 'Human':
                    history_result = self._create_user_history(
                        user, users_text, share_nums)
                    if history_result:
                        history_results[user['id']] = history_result
                    else:
                        print(f"Failed to create history record for user {user.get('id')}")
            
            # Add result validation
            print(f"Number of successfully created history records: {len(history_results)}")
            
            if not history_results:
                raise ValueError("No history records were successfully created")
                
            # Modify write method to ensure data integrity
            temp_file = Path(self.config["History"]["HumanHistricalInfoPolitics"] + ".tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(history_results, f, indent=4, ensure_ascii=False)
            
            # Validate written data
            with open(temp_file, 'r', encoding='utf-8') as f:
                written_data = json.load(f)
                if len(written_data) != len(history_results):
                    raise ValueError("Written data is incomplete")
            
            # Replace original file after confirming data integrity
            temp_file.replace(Path(self.config["History"]["HumanHistricalInfoPolitics"]))
            
            print("Human historical data creation completed")
                
        except Exception as e:
            print(f"Failed to create human historical data: {str(e)}")
            # Clean up temporary file
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()

    def _create_user_history(self, user: Dict, users_text: Dict, share_nums: Dict) -> Dict:
        """Optimized single user history creation"""
        try:
            # Add input validation
            required_fields = ['id', 'user_id', 'trust_threshold']
            if not all(field in user for field in required_fields):
                print(f"User data missing required fields: {user.get('id')}")
                return None
                
            history = {
                'id': user['id'],
                'user_id': user['user_id'],
                'social influence': user['social_influence'],
                'neighbor_nodes': self.edge_index_cache[user['id']],
                'neighbor_nodes_social_influence': self.edge_index_cache_social_influence[user['id']],
                'dissemination tendency': [{0:user['dissemination_tendency']}],
                'trust threshold': {
                    0: {comm: self._get_community_value(
                        user['trust_threshold'], comm) 
                        for comm in self.communities}
                },
                'receive info': {comm: [] for comm in self.communities},
                'share info': {comm: [] for comm in self.communities},
                'user_text': users_text.get(user['user_id'], {}),
                'receive_info_num': 0,
            }
        
            # Add sharing number information, use default values to ensure fields exist
            share_num = share_nums.get(user['user_id'], {})
            history.update({
                'retweet_num': share_num.get('retweet_num', 0),
                'quote_num': share_num.get('quote_num', 0)
            })
            
            # Validate if created history record is complete
            required_history_fields = [
                'id', 'user_id', 'trust threshold', 'receive info', 
                'share info', 'user_text', 'retweet_num', 'quote_num',
                'neighbor_nodes', 'neighbor_nodes_social_influence'
            ]
            
            if not all(field in history for field in required_history_fields):
                missing_fields = [f for f in required_history_fields if f not in history]
                print(f"User {user['id']} history record missing fields: {missing_fields}")
                return None
                
            return history
            
        except Exception as e:
            print(f"Error creating history record for user {user.get('id')}: {str(e)}")
            return None

    @staticmethod
    def _load_json(file_path: str) -> Dict:
        """Optimized JSON loading"""
        try:
            if not Path(file_path).exists():
                print(f"File does not exist: {file_path}")
                return {}
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Successfully loaded file {file_path}, data size: {len(str(data))} bytes")
                return data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error {file_path}: {str(e)}")
            return {}
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return {}

    @staticmethod
    def _get_community_value(trust_threshold: List[Dict], community: str) -> float:
        """Optimized community value retrieval"""
        return trust_threshold[community]

class TimeStepManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.human_time_step = self._load_human_time_step()
        self.mbot_time_step = self._load_mbot_time_step()
        self.lbot_time_step = self._load_lbot_time_step(self.config["Parameters"]["inferenece_time"])

    def _load_human_time_step(self):
        with open(self.config["paths"]["HumanMBotLBotDataPolitics"], 'r', encoding='utf-8') as f:
            human_mbot_data = json.load(f)
        human_time_step = {}

        for time_step in range(1, 73):
            human_time_step[time_step] = []
            for key in human_mbot_data.keys():
                data = human_mbot_data[key]
                if data['label'] == 'Human':
                    if time_step in data['activation_time']:
                        human_time_step[time_step].append(data['id'])
        return human_time_step  

    def _load_mbot_time_step(self):
        with open(self.config["paths"]["HumanMBotLBotDataPolitics"], 'r', encoding='utf-8') as f:
            human_mbot_data = json.load(f)
        mbot_time_step = {}
        communitylist = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
        for comm in communitylist:
            mbot_time_step[comm] = {}
            for time_step in range(1, 73):
                mbot_time_step[comm][time_step] = []
                for key in human_mbot_data.keys():
                    data = human_mbot_data[key]
                    if data['label'] == 'MBot' and comm in data['interest_community']:
                        if time_step in data['activation_time']:
                            mbot_time_step[comm][time_step].append(data['id'])
        return mbot_time_step

    def _load_lbot_time_step(self, inferenece_time: str):
        with open(self.config["paths"]["HumanMBotLBotDataPolitics"], 'r', encoding='utf-8') as f:
            human_mbot_data = json.load(f)
        lbot_time_step = {}
        communitylist = ["Business", "Education", "Entertainment", "Politics", "Sports", "Technology"]
        for comm in communitylist:
            lbot_time_step[comm] = {}
            for time_step in range(1, 73):
                lbot_time_step[comm][time_step] = []
                for key in human_mbot_data.keys():
                    data = human_mbot_data[key]
                    if data['label'] == 'LBot' and comm in data['interest_community']:
                        if time_step in data[inferenece_time]:
                            lbot_time_step[comm][time_step].append(data['id'])
        return lbot_time_step
