from typing import Dict, List, Any
import json
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class HistoryMemoryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.communities = ["Business", "Education", "Entertainment", 
                          "Politics", "Sports", "Technology"]
        
    def initialize_history(self):
        """初始化所有历史数据"""
        self._create_bot_history()
        self._create_human_history()
        
    def _create_bot_history(self):
        """优化的机器人历史创建"""
        if Path(self.config["newpaths"]["MBotHistricalInfo"]).exists():
            return
            
        try:
            with open(self.config["paths"]["botfile"], 'r', encoding='utf-8') as f:
                bots_data = json.load(f)
                
            history_results = []
            for bot in bots_data:
                history_result = {
                    'id': bot['id'],
                    'user_id': bot['user_id'],
                    'receive info': {comm: [] for comm in self.communities},
                    'share info': {comm: [] for comm in self.communities}
                }
                history_results.append(history_result)
                
            with open(self.config["newpaths"]["MBotHistricalInfo"], 'w', encoding='utf-8') as f:
                json.dump(history_results, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"Bot history creation failed: {str(e)}")

    def _create_human_history(self):
        """优化的人类用户历史创建"""
        if Path(self.config["newpaths"]["HumanHistricalInfo"]).exists():
            return  # 这里可能过早返回，应该检查文件内容是否完整
            
        try:
            # 添加日志记录
            print("开始创建人类历史数据...")
            
            # 串行加载数据
            humans_data = self._load_json(self.config["paths"]["humanfile"])
            users_text = self._load_json(self.config["paths"]["humantextfile"])
            share_nums = self._load_json(self.config["paths"]["sharenumfile"])
            
            # 添加数据验证
            if not humans_data:
                raise ValueError(f"人类用户数据加载失败: {self.config['paths']['humanfile']}")
            
            print(f"加载的用户数量: {len(humans_data)}")
            
            history_results = []
            for user in humans_data:
                history_result = self._create_user_history(
                    user, users_text, share_nums)
                if history_result:
                    history_results.append(history_result)
                else:
                    print(f"用户 {user.get('id')} 的历史记录创建失败")
            
            # 添加结果验证
            print(f"成功创建的历史记录数量: {len(history_results)}")
            
            if not history_results:
                raise ValueError("没有成功创建任何历史记录")
                
            # 修改写入方式，确保数据完整性
            temp_file = Path(self.config["newpaths"]["HumanHistricalInfo"] + ".tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(history_results, f, indent=4, ensure_ascii=False)
            
            # 验证写入的数据
            with open(temp_file, 'r', encoding='utf-8') as f:
                written_data = json.load(f)
                if len(written_data) != len(history_results):
                    raise ValueError("写入的数据不完整")
            
            # 确认数据完整后，替换原文件
            temp_file.replace(Path(self.config["newpaths"]["HumanHistricalInfo"]))
            
            print("人类历史数据创建完成")
                
        except Exception as e:
            print(f"创建人类历史数据失败: {str(e)}")
            # 清理临时文件
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink()

    def _create_user_history(self, user: Dict, users_text: Dict, 
                           share_nums: Dict) -> Dict:
        """优化的单个用户历史创建"""
        try:
            # 添加输入验证
            required_fields = ['id', 'user_id', 'trust threshold']
            if not all(field in user for field in required_fields):
                print(f"用户数据缺少必要字段: {user.get('id')}")
                return None
                
            history = {
                'id': user['id'],
                'user_id': user['user_id'],
                'trust threshold': {
                    0: {comm: self._get_community_value(
                        user['trust threshold'], comm) 
                        for comm in self.communities}
                },
                'receive info': {comm: [] for comm in self.communities},
                'share info': {comm: [] for comm in self.communities},
                'user_text': users_text.get(user['user_id'], {}),
            }
            
            # # 添加用户文本信息，使用默认值确保字段存在
            # user_text = users_text.get(user['user_id'], {})

            # history.update({
            #     'personal_description': user_text.get('personal_description', ''),
            #     'historical_posts': user_text.get('historical_posts', {}),
            #     'historical_retweets': user_text.get('historical_retweets', {}),
            #     'historical_quotes': user_text.get('historical_quotes', {})
            # })
            
            # 添加分享数量信息，使用默认值确保字段存在
            share_num = share_nums.get(user['user_id'], {})
            history.update({
                'retweet_num': share_num.get('retweet_num', 0),
                'quote_num': share_num.get('quote_num', 0)
            })
            
            # 验证创建的历史记录是否完整
            required_history_fields = [
                'id', 'user_id', 'trust threshold', 'receive info', 
                'share info', 'user_text', 'retweet_num', 'quote_num'
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

    def _load_community_nodes(self, community_file: str, human_num: int) -> List[str]:
        """加载社区节点"""
        with open(community_file, 'r', encoding='utf-8') as f:
            community_data = json.load(f)
        community_nodes = []
        for commdata in community_data:
            key_id = list(commdata.keys())[0]
            if commdata[key_id] <= human_num:
                community_nodes.append(commdata[key_id])
        return community_nodes

    @staticmethod
    def _get_community_value(trust_threshold: List[Dict], community: str) -> float:
        """优化的社区值获取"""
        for item in trust_threshold:
            if community in item:
                return item[community]
        return 0.0 