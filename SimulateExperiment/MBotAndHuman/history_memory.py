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
        """初始化所有历史数据"""
        self._create_mbot_history()
        self._create_human_history()
        self._create_lbot_history()

    def _load_neighbour_nodes(self, edge_index: np.ndarray) -> Dict[int, List[int]]:
        """获取邻居节点"""
        edge_index = np.load(edge_index)
        neighbor_dict = defaultdict(list)
        # 遍历边
        for src, dst in edge_index.T.tolist():  # 转置后每一行是一个边 (src, dst)
            neighbor_dict[src].append(dst)
        neighbor_dict = dict(neighbor_dict)
        return neighbor_dict
    
    def _load_neighbour_nodes_social_influence(self, human_mbot_lbot_data: Dict) -> Dict[int, List[int]]:
        """获取邻居节点社交影响力"""
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
        """优化的机器人历史创建"""
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
        """优化的机器人历史创建"""
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
        """优化的人类用户历史创建"""
        if Path(self.config["History"]["HumanHistricalInfoPolitics"]).exists():
            return  # 这里可能过早返回，应该检查文件内容是否完整
            
        try:
            # 添加日志记录
            print("开始创建人类历史数据...")
            
            # 串行加载数据
            humans_data = self._load_json(self.config["paths"]["HumanMBotLBotDataPolitics"])
            users_text = self._load_json(self.config["paths"]["HumanSummerizeText"])
            share_nums = self._load_json(self.config["paths"]["sharenumfile"])
            
            print(f"加载的用户数量: {len(humans_data)}")
            
            history_results = {}
            for user_key in humans_data.keys():
                user = humans_data[user_key]
                if user['label'] == 'Human':
                    history_result = self._create_user_history(
                        user, users_text, share_nums)
                    if history_result:
                        history_results[user['id']] = history_result
                    else:
                        print(f"用户 {user.get('id')} 的历史记录创建失败")
            
            # 添加结果验证
            print(f"成功创建的历史记录数量: {len(history_results)}")
            
            if not history_results:
                raise ValueError("没有成功创建任何历史记录")
                
            # 修改写入方式，确保数据完整性
            temp_file = Path(self.config["History"]["HumanHistricalInfoPolitics"] + ".tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(history_results, f, indent=4, ensure_ascii=False)
            
            # 验证写入的数据
            with open(temp_file, 'r', encoding='utf-8') as f:
                written_data = json.load(f)
                if len(written_data) != len(history_results):
                    raise ValueError("写入的数据不完整")
            
            # 确认数据完整后，替换原文件
            temp_file.replace(Path(self.config["History"]["HumanHistricalInfoPolitics"]))
            
            print("人类历史数据创建完成")
                
        except Exception as e:
            print(f"创建人类历史数据失败: {str(e)}")
            # 清理临时文件
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()

    def _create_user_history(self, user: Dict, users_text: Dict, share_nums: Dict) -> Dict:
        """优化的单个用户历史创建"""
        try:
            # 添加输入验证
            required_fields = ['id', 'user_id', 'trust_threshold']
            if not all(field in user for field in required_fields):
                print(f"用户数据缺少必要字段: {user.get('id')}")
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
        
            
            # 添加分享数量信息，使用默认值确保字段存在
            share_num = share_nums.get(user['user_id'], {})
            history.update({
                'retweet_num': share_num.get('retweet_num', 0),
                'quote_num': share_num.get('quote_num', 0)
            })
            
            # 验证创建的历史记录是否完整
            required_history_fields = [
                'id', 'user_id', 'trust threshold', 'receive info', 
                'share info', 'user_text', 'retweet_num', 'quote_num',
                'neighbor_nodes', 'neighbor_nodes_social_influence'
            ]
            
            if not all(field in history for field in required_history_fields):
                missing_fields = [f for f in required_history_fields if f not in history]
                print(f"用户 {user['id']} 的历史记录缺少字段: {missing_fields}")
                return None
                
            return history
            
        except Exception as e:
            print(f"创建用户 {user.get('id')} 的历史记录时出错: {str(e)}")
            return None

    @staticmethod
    def _load_json(file_path: str) -> Dict:
        """优化的JSON加载"""
        try:
            if not Path(file_path).exists():
                print(f"文件不存在: {file_path}")
                return {}
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"成功加载文件 {file_path}, 数据大小: {len(str(data))} bytes")
                return data
        except json.JSONDecodeError as e:
            print(f"JSON解析错误 {file_path}: {str(e)}")
            return {}
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            return {}

    @staticmethod
    def _get_community_value(trust_threshold: List[Dict], community: str) -> float:
        """优化的社区值获取"""
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
