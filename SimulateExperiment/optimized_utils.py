import numpy as np
from typing import Dict, List, Tuple, Any
import json
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random

@dataclass
class UserAction:
    action: str
    content: str
    index: int
    plausibility: float
    stance: int

class OptimizedUtils:
    def __init__(self, config: Dict):
        self.config = config
        self.user_cache = {}
        self.community_cache = {}
        
    @lru_cache(maxsize=1000)
    def load_si(self, user_id: int) -> int:
        """优化的社交影响力加载函数"""
        if user_id not in self.user_cache:
            try:
                with open(self.config['paths']['humanfile'], 'r') as f:
                    humans = json.load(f)
                with open(self.config['paths']['botfile'], 'r') as f:
                    bots = json.load(f)
                self.user_cache.update({u['id']: u['social influence'] for u in humans + bots})
            except Exception as e:
                return 0
        return self.user_cache.get(user_id, 0)

    @staticmethod
    def calculate_tt_score(tt_ij: float, support_sum: float, oppose_sum: float, 
                          gamma: float = 0.5, alpha: float = 0.05, beta: float = 0.1) -> float:
        """优化的信任阈值计算"""
        try:
            enhancement = gamma * (1 - np.exp(-beta * support_sum))
            decay = (1 - gamma) * (1 - np.exp(-alpha * oppose_sum))
            return np.clip(tt_ij + enhancement - decay, 0, 1)
        except:
            return 0

    def process_user_batch(self, users: List[int], comm: str) -> List[Tuple[int, float]]:
        """批量处理用户信任阈值"""
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for user_id in users:
                futures.append(
                    executor.submit(self.process_single_user, user_id, comm)
                )
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        return results

    @lru_cache(maxsize=1000)
    def get_user_history(self, user_id: int, comm: str) -> Tuple[float, float]:
        """优化的用户历史数据获取"""
        if user_id not in self.user_cache:
            try:
                with open(self.config['newpaths']['HumanHistricalInfo'], 'r') as f:
                    history_data = json.load(f)
                    user_data = next(u for u in history_data if u['id'] == user_id)
                    self.user_cache[user_id] = user_data
            except:
                return 0.0, 0.0
                
        user_data = self.user_cache[user_id]
        support_sum = 0.0
        oppose_sum = 0.0
        
        if comm in user_data['receive info']:
            for item in user_data['receive info'][comm]:
                si = self.load_si(item['from_id'])
                if item['stance'] == 1:
                    support_sum += si * item['info score']
                else:
                    oppose_sum += si * item['info score']
                    
        return support_sum, oppose_sum

    def share_decision(self, user_id: int, info: str, comm: str) -> UserAction:
        """优化的分享决策"""
        if random.random() < 0.5:  # 简化的决策逻辑
            action = "Repost" if random.random() < 0.7 else "Quote"
            content = info if action == "Repost" else f"Comment: AI generated comment. Original: {info}"
            return UserAction(
                action=action,
                content=content,
                index=0,
                plausibility=random.random(),
                stance=random.randint(0, 1)
            )
        return None 