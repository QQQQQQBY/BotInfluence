import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import json
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import random
from dataclasses import dataclass
from functools import lru_cache
import yaml
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from tqdm import tqdm
from datetime import datetime
from langchain.callbacks import get_openai_callback
@dataclass
class Config:
    paths: Dict
    newpaths: Dict
    History: Dict
    parameters: Dict
    disinfopaths: Dict
    promptpaths: Dict
    log: Dict
    disinfo_claim: Dict

class LLMManager:
    """LLM管理器，统一管理所有LLM相关操作"""
    def __init__(self):
        self.total_tokens = 0  # 添加token计数器
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="deepseek-v3",
            temperature=0.9,
            api_key="",
            base_url='',
            max_retries=2,
            http_client=httpx.Client(
                timeout=20.0,
                limits=httpx.Limits(max_connections=100),
                transport=httpx.HTTPTransport(retries=2)
            )
        )

    def get_token_usage(self) -> int:
        """获取总token使用量"""
        return self.total_tokens

    def reset_token_counter(self) -> None:
        """重置token计数器"""
        self.total_tokens = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
        ),
        reraise=True
    )
    def process_prompt(self, prompt: PromptTemplate, user_data: dict, logger: logging.Logger, max_retries: int = 3) -> dict:
        """处理提示词并返回结果
        
        Args:
            prompt: 提示词模板
            user_data: 用户数据
            max_retries: 最大重试次数，默认为3次
            
        Returns:
            dict: LLM处理结果
        """
        retries = 0
        while retries < max_retries:
            try:
                chain = (prompt | self.llm | JsonOutputParser())
                with get_openai_callback() as cb:
                    result = chain.invoke(user_data)
                    self.total_tokens += cb.total_tokens
                # 记录token使用量
                return result
            except Exception as e:
                retries += 1
                logger.error(f"LLM处理失败 (尝试 {retries}/{max_retries}): {str(e)}")
                if retries == max_retries:
                    logger.error("达到最大重试次数，返回空结果")
                    return {}
            

class DisseminationOptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.edge_index_cache = None
        self.mbot_cache = self._get_mbot_historical_data()
        self.lbot_cache = self._get_lbot_historical_data()
        self.human_cache = self._get_user_historical_data()
        self.llm_manager = LLMManager()
        self.token_usage_history = []  # Add token usage history record
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logger"""
        logger = logging.getLogger('dissemination')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.config.log['Disseminate_log_file'])
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_user_historical_data(self) -> dict:
        """Get user historical data"""
        try:
            human_cache = {}
            with open(self.config.History['HumanHistricalInfoPolitics'], 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            for user_id in history_data.keys():
                user_data = history_data[user_id]
                if int(user_data['id']) not in human_cache:
                    human_cache[int(user_data['id'])] = user_data

            return human_cache
            
        except Exception as e:
            self.logger.error(f"Failed to get user historical data: {str(e)}")
            return {}

    def _get_mbot_historical_data(self) -> dict:
        """Get bot historical data"""
        try:
            mbot_cache = {}
            with open(self.config.History['MBotHistricalInfoPolitics'], 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            for mbot_id in history_data.keys():
                mbot_data = history_data[mbot_id]
                if int(mbot_data['id']) not in mbot_cache:
                    mbot_cache[int(mbot_data['id'])] = mbot_data

            return mbot_cache
            
        except Exception as e:
            self.logger.error(f"Failed to get bot historical data: {str(e)}")
            return {}
    
    def _get_lbot_historical_data(self) -> dict:
        """Get bot historical data"""
        try:
            lbot_cache = {}
            with open(self.config.History['LBotHistricalInfoPolitics'], 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            for lbot_id in history_data.keys():
                lbot_data = history_data[lbot_id]
                if int(lbot_data['id']) not in lbot_cache:
                    lbot_cache[int(lbot_data['id'])] = lbot_data

            return lbot_cache
        
        except Exception as e:
            self.logger.error(f"Failed to get bot historical data: {str(e)}")
            return {}
        
    @staticmethod
    def to_undirected_numpy(edge_index: np.ndarray) -> np.ndarray:
        """将边索引转换为无向图"""
        if edge_index.size == 0:
            return edge_index
        reverse_edges = np.flip(edge_index, axis=0)
        return np.unique(np.concatenate([edge_index, reverse_edges], axis=1), axis=1)

    def load_neighbour_nodes(self, centernodes: Set[int], edge_index: np.ndarray) -> Dict[int, List[int]]:
        """获取邻居节点"""
        if self.edge_index_cache is None:
            self.edge_index_cache = self.to_undirected_numpy(edge_index).T
        
        disseminationnodes = defaultdict(list)
        for edge in self.edge_index_cache:
            for node in centernodes:
                if node in edge:
                    neighbor = edge[0] if edge[1] == node else edge[1]
                    disseminationnodes[node].append(int(neighbor))
                    disseminationnodes[node] = list(set(disseminationnodes[node]))
        return dict(disseminationnodes)


    def load_mbot_info(self, mbot_id: int, topic: str) -> Tuple[str, float, int]:
        """加载机器人最新的分享信息"""
        mbot_data = self.mbot_cache[mbot_id]
        
        latest_info = mbot_data['share info'][topic][-1]
        return (latest_info['content'], 
                latest_info['information Plausibility'], 
                latest_info['stance'])

    def load_lbot_info(self, lbot_id: int, topic: str) -> Tuple[str, float, int]:
        """加载机器人最新的分享信息"""
        lbot_data = self.lbot_cache[lbot_id]
        
        latest_info = lbot_data['share info'][topic][-1]
        return (latest_info['content'], 
                latest_info['information Plausibility'], 
                latest_info['stance'])
    
    def load_human_info(self, human_id: int, topic: str) -> Tuple[str, float, int]:
        """加载人类用户最新的分享信息"""
        human_data = self.human_cache[human_id]
        # 加载中心节点最新分享的信息
        latest_info = human_data['share info'][topic][-1]
        return (latest_info['content'],
                latest_info['information Plausibility'],
                latest_info['stance'])

    
    def validate_yaml(self, promptfile: str) -> bool:
        """验证提示词文件是否有效"""
        try:
            path = Path(promptfile)
            with open(path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            # print("YAML file format is correct")
            return True
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e.problem} (row{e.problem_mark.line+1})")
            return False
        except Exception as e:
            print(f"Failed to read the file: {str(e)}")
            return False
                
    def load_prompt(self, promptfile: str) -> Tuple[PromptTemplate, List[str], List[str]]:
        """加载提示词模板"""
        try:
            if self.validate_yaml(promptfile):
                with open(promptfile, 'r', encoding='utf-8') as f:
                    yaml_content = yaml.safe_load(f)
                json_prompt_template = yaml_content["template"]
                input_variables = yaml_content["input_variables"]
                attribute = yaml_content["attribute"]
                prompt = PromptTemplate.from_template(json_prompt_template)
                return prompt, input_variables, attribute
            else:
                self.logger.error(f"提示词文件验证失败: {promptfile}")
                return None, [], []
        except Exception as e:
            self.logger.error(f"加载提示词失败: {str(e)}")
            return None, [], []

    def process_llm_request(self, prompt_file: str, user_data: dict) -> dict:
        """Unified LLM request processing"""
        try:
            prompt, input_vars, attribute = self.load_prompt(prompt_file)
            if not prompt:
                return {}
                
            result = self.llm_manager.process_prompt(prompt, user_data, self.logger)
            if result is None:
                self.logger.error("LLM returned empty result")
                return {}
            
            # Record token usage
            self.log_token_usage(f"Processing {prompt_file} request")
            
            # Validate if returned result contains required attributes
            if all(attr in result for attr in attribute):
                return result
            else:
                self.logger.error(f"LLM result missing required attributes: {attribute}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to process LLM request: {str(e)}")
            return {}

    
    def DWC_main(self, topic: str, exposed_nodes: Dict[int, List[int]], 
                 center_nodes: Set[int], index: int, mbot_ids: Set[int], 
                 lbot_ids: Set[int], human_ids: Set[int], disinfo: str, 
                 human_num: int, mbot_num: int, lbot_num: int, correctstrategy: str) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Main information dissemination process
        
        Args:
            topic: Current topic
            exposed_nodes: Exposed nodes dictionary {center node: [exposed nodes list]}
            center_nodes: Center nodes set
            index: Current time step
            bot_ids: Bot IDs set
            human_ids: Human user IDs set
            disinfo: Current disseminated false information claim

        Returns:
            Tuple[List[int], List[int], List[int], List[int]]: 
            (All exposed human nodes, Trust nodes, Untrust nodes, Uninterested nodes)
        """
        self.logger.info(f'Starting to process {topic} topic at time step {index}')
        
        # Initialize result sets
        all_exposed_humans = set()
        all_trust_nodes = set()
        all_unbelieve_nodes = set()

        try:
            # 0. Load false information content
            if index == 1:
                # 0.1 Load false information content
                current_disinfo, plausibility = self.load_initial_disinfo(topic)
                current_disinfo, plausibility = self.rewrite_disinfo(current_disinfo)
                # 0.2 For bot nodes, add initial sharing information
                for node in center_nodes & (mbot_ids | lbot_ids):  # Intersection operation, only process center nodes that are bots
                    mbot_share_info = {
                        'action': 'Post',
                        'content': current_disinfo,
                        'index': index,
                        'information Plausibility': plausibility,
                        'stance': 1  # Initial false information stance released by bot is 1
                    }
                    self.add_mbot_share_info(node, mbot_share_info, topic)

            # 1. Process each center node serially
            for node in tqdm(center_nodes, desc="Processing center nodes"):
                try:
                    # 2. Process single center node's information dissemination
                    # If the node has received information, it can share
                    if node in human_ids:
                        # if self.human_cache[node]['share info'][topic] != []:
                            exposed, trust, unbelieve = self.process_center_node(
                                node,
                                exposed_nodes.get(node, []),
                                mbot_ids,
                                lbot_ids,
                                human_ids,
                                index,
                                topic,
                                disinfo
                                )
                    if node in mbot_ids:
                        if self.mbot_cache[node]['share info'][topic] != []:
                            exposed, trust, unbelieve = self.process_center_node(
                                node,
                                exposed_nodes.get(node, []),
                                mbot_ids,
                                lbot_ids,
                                human_ids,
                                index,
                                topic,
                                disinfo
                            )
                        if self.mbot_cache[node]['share info'][topic] == []:
                            current_disinfo, plausibility = self.load_initial_disinfo(topic)
                            current_disinfo, plausibility = self.rewrite_disinfo(current_disinfo)
                            mbot_share_info = {
                            'action': 'Post',
                            'content': current_disinfo,
                            'index': index,
                            'information Plausibility': plausibility,
                            'stance': 1  # Initial false information stance released by bot is 1
                                }
                            self.add_mbot_share_info(node, mbot_share_info, topic)
                            exposed, trust, unbelieve = self.process_center_node(
                                node,
                                exposed_nodes.get(node, []),
                                mbot_ids,
                                lbot_ids,
                                human_ids,
                                index,
                                topic,
                                disinfo
                            )
                    if node in lbot_ids:
                        if self.lbot_cache[node]['share info'][topic] != []:
                            exposed, trust, unbelieve = self.process_center_node(
                            node,
                            exposed_nodes.get(node, []),
                            mbot_ids,
                            lbot_ids,
                            human_ids,
                            index,
                            topic,
                            disinfo
                            )
                        if self.lbot_cache[node]['share info'][topic] == []:
                            correct_info, plausibility = self.load_lbot_correct_info(topic, correctstrategy)
                            correct_info, plausibility = self.rewrite_correct_info(correct_info)
                            lbot_share_info = {
                                'action': 'Post',
                                'content': correct_info,
                                'index': index,
                                'information Plausibility': plausibility,
                                'stance': 0  # Initial false information stance released by bot is 0
                            }
                            self.add_lbot_share_info(node, lbot_share_info, topic)
                            exposed, trust, unbelieve = self.process_center_node(
                                node,
                                exposed_nodes.get(node, []),
                                mbot_ids,
                                lbot_ids,
                                human_ids,
                                index,
                                topic,
                                disinfo
                            )
                    # 3. Update results
                    all_exposed_humans.update(exposed)
                    all_trust_nodes.update(trust)
                    all_unbelieve_nodes.update(unbelieve)
                except Exception as e:  
                    self.logger.error(f"Error processing node {node}: {str(e)}")

            self.logger.info(f'Completed processing {topic} topic at time step {index}')
            
            # 4. Update trust threshold for all trust nodes, at current index time step
            self.update_all_trust_thresholds(
                all_trust_nodes, 
                all_unbelieve_nodes, 
                topic, 
                index
            )

            # Update dissemination tendency
            self.update_dissemination_tendency(all_trust_nodes, all_unbelieve_nodes, topic, index)

            # 5. Update bot and human historical information
            with open(self.config.History['MBotHistricalInfoPolitics'], 'w', encoding='utf-8') as f:   
                json.dump(self.mbot_cache, f, indent=4, ensure_ascii=False)
            
            with open(self.config.History['HumanHistricalInfoPolitics'], 'w', encoding='utf-8') as f:   
                json.dump(self.human_cache, f, indent=4, ensure_ascii=False)

            with open(self.config.History['LBotHistricalInfoPolitics'], 'w', encoding='utf-8') as f:   
                json.dump(self.lbot_cache, f, indent=4, ensure_ascii=False)

            # 6. Get token usage
            total_tokens = self.llm_manager.get_token_usage()
            self.logger.info(f"{index} time step, DWC_main completed, token usage: {total_tokens}")

            return (
                list(all_exposed_humans),
                list(all_trust_nodes),
                list(all_unbelieve_nodes),
                total_tokens
            )

        except Exception as e:
            self.logger.error(f"DWC_main execution error: {str(e)}")
            return [], [], [], [], 0

    def rewrite_disinfo(self, disinfo: str) -> Tuple[str, float]:
        """
        改写（重新描述当前的虚假信息）
        """
        try:
            result = self.process_llm_request(
                self.config.promptpaths['RewriteDisinformation'],
                {"disinformation": disinfo}
            )
            plausibility = self.get_information_plausibility(result.get('NewDisinformation', disinfo))
            return result.get('NewDisinformation', disinfo), plausibility
        except Exception as e:
            self.logger.error(f"改写虚假信息失败: {str(e)}")
            return disinfo, 0.5

    def load_initial_disinfo(self, topic: str) -> Tuple[str, float]:
        """
        0.1 从Dataset/Disinformation目录下加载初始虚假信息
        
        Args:
            topic: 话题名称 (Business, Education, Entertainment, Politics, Sports, Technology)

        Returns:
            Tuple[str, float]: (虚假信息内容, 可信度分数)
        """
        try:
            # 构建正确的文件路径
            disinfo_file = self.config.disinfopaths[topic]
            
            self.logger.info(f"尝试从 {disinfo_file} 加载虚假信息")
            
            # 检查文件是否存在
            if not Path(disinfo_file).exists():
                self.logger.error(f"虚假信息文件不存在: {disinfo_file}")
                return "Default disinformation", 0.5
            
            # 加载虚假信息文件
            with open(disinfo_file, 'r', encoding='utf-8') as f:
                disinfo_data = yaml.safe_load(f)
            
            if not isinstance(disinfo_data, dict):
                self.logger.error(f"虚假信息文件格式错误: {disinfo_file}")
                return "Default disinformation", 0.5
            
            # 获取虚假信息列表和可信度列表
            disinfo_list = disinfo_data.get("DisinformationDescribeList", [])
            plausibility_list = disinfo_data.get("DisinformationPlausibility", [])
            
            if not disinfo_list or not plausibility_list:
                self.logger.error(f"虚假信息文件内容为空: {disinfo_file}")
                return "Default disinformation", 0.5
            
            # 随机选择一条虚假信息
            index = random.randint(0, len(disinfo_list) - 1)
            disinfo = disinfo_list[index]
            plausibility = plausibility_list[index]
            
            self.logger.info(f"成功加载 {topic} 话题的虚假信息")
            return disinfo, plausibility
        
        except Exception as e:
            self.logger.error(f"加载 {topic} 话题的虚假信息失败: {str(e)}")
            return "Default disinformation", 0.5

    def load_lbot_correct_info(self, topic: str, correctstrategy: str) -> Tuple[str, float]:
        """加载合法机器人最新的分享信息"""
        filename = self.config.disinfopaths[topic]
        with open(filename, 'r', encoding='utf-8') as f:
            correctinfo_data = yaml.safe_load(f)

        info_list = correctinfo_data['CorrectiveInformation'].get(correctstrategy, [])
       
        index = random.randint(0, len(info_list) - 1)
        info = info_list[index]
        plausibility = correctinfo_data['CorrectiveInformationPlausibility'].get(correctstrategy, [])[index]
        return info, plausibility
    
    def process_center_node(self, node: int, exposed_nodes: List[int],
                          mbot_ids: Set[int], lbot_ids: Set[int], human_ids: Set[int],
                          index: int, topic: str, disinfo: str) -> Tuple[List[int], List[int], List[int], List[int]]:
        """2. 处理单个中心节点的信息传播"""
        try:
            # 2.1 获取节点信息，中心节点，将这些节点的信息分享给暴露节点（一阶邻居节点）
            if node in mbot_ids:
                info, plausibility, stance = self.load_mbot_info(node, topic)
            # elif node in lbot_ids:
            #     info, plausibility, stance = self.load_lbot_info(node, topic)
            else:
                info, plausibility, stance = self.load_human_info(node, topic)
            exposed_nodes = list(set(exposed_nodes))
            # 2.2 处理暴露节点（一阶邻居节点）

            exposed_human_nodes = []
            for n in tqdm(exposed_nodes, desc="Processing exposed nodes"):
                if n in human_ids:
                    exposed_human_nodes.append(n)
                    # 2.2.1 向人类邻居节点添加接收信息记录
                    self.add_human_receive_info(n, node, index, info, topic, disinfo, stance)

            # 2.4 处理人类节点的分享决策
            # 2.4.1 获取在当前时刻决定分享节点 节点都会接收到消息，但只选择最后一条消息进行分享？接收到消息一定会看？ 当前时间段激活了才能接收？
            # 激活时间之后就看不到了。。也会影响其置信度分数？决定分享，但是不会分享出去，但是对其置信度有影响？，激活时间是有action的时间，最后一个时间看到了？对其有了影响？
            share_nodes = self.get_share_nodes(exposed_human_nodes, topic)
            # 2.4.2 处理分享节点，将分享节点划分为相信当前接收的信息或者不相信当前接收的信息
            trust_nodes, unbelieve_nodes = self.process_share_nodes(
                share_nodes, topic, plausibility, stance, index
            )
            # 2.4.3 处理信任节点和非信任节点的分享行为
            trust_actions, unbelieve_actions = self.process_human_share_actions(
                trust_nodes, unbelieve_nodes, info, topic
            )
            # 2.4.4 向分享节点添加分享信息记录
            self.add_human_share_info(trust_actions, unbelieve_actions, topic, index)    

            # 2.4.5 获取未感兴趣节点（没有进行分享的节点
            # uninterested_nodes = list(set(exposed_human_nodes) - set(share_nodes))
            
            # 2.4.6 返回结果
            return exposed_human_nodes, trust_nodes, unbelieve_nodes

        except Exception as e:
            self.logger.error(f"处理中心节点 {node} 时发生错误: {str(e)}")
            return [], [], [], []

    def add_human_receive_info(self, human_id: int, from_id: int, 
                             index: int, info: str, topic: str, disinfo: str, stance: int) -> None:
        """2.2.1 向人类邻居节点添加接收信息记录"""
        """
        format:
        {
            'from_id': from_id,
            'index': index,
            'text_info': info,
            'social influence': int,
            'info score': float,
            'stance': int
        }
        """
        # 2.2.1.1 计算分享信息的用户对当前用户（human_id）的影响力
        social_influence_score = self.get_social_influence(human_id, from_id)
        # 2.2.1.2 计算信息对人类的可信度
        credibility_score = self.calculate_info_score_to_human(info, human_id)
        # 2.2.1.3 判断人类的立场
        # stance = self.judge_stance_of_human(info, disinfo)
        try:
            receive_info = {
                'from_id': from_id,
                'index': index,
                'text_info': info,
                'social influence': social_influence_score,
                'info score': credibility_score,
                'stance': stance
            }

            self.human_cache[human_id]['receive info'][topic].append(receive_info)

        except Exception as e:
            self.logger.error(f"添加人类接收信息记录失败: {str(e)}")

    def get_social_influence(self, user_id: int, from_id: int) -> int:
        """2.2.1.1 获取用户的社交影响力"""
        try:
            user_data = self.human_cache[user_id]['neighbor_nodes_social_influence'][str(from_id)]
            if not user_data:
                return 0
            return user_data
        except Exception as e:
            self.logger.error(f"获取社交影响力失败: {str(e)}")
            return 0

    def calculate_info_score_to_human(self, info: str, user_id: int) -> float:
        """2.2.1.2 计算信息对用户的影响分数"""
        try:
            # 获取用户历史数据
            user_data = self.human_cache[user_id]
            if not user_data:
                return 0.0

            # 构建LLM输入数据
            llm_input = {
                "text_information": info,
                "history_info": user_data.get('user_text', {}),
            }

            # 使用统一的LLM处理方法
            result = self.process_llm_request(
                self.config.promptpaths['CalculateInfoScore'],
                llm_input
            )

            return result.get('Score', 0.5)

        except Exception as e:
            self.logger.error(f"计算信息影响分数失败: {str(e)}")
            return 0.5


    def judge_stance_of_human(self, info: str, disinfo: str) -> int:
        """2.2.1.3 判断信息的立场"""
        try:
            llm_input = {
                "original_info": disinfo,
                "current_info": info
            }
            
            result = self.process_llm_request(
                self.config.promptpaths['JudgeStance'],
                llm_input
            )
            
            return result.get('consistency_score', random.randint(0, 1))
            
        except Exception as e:
            self.logger.error(f"判断信息立场失败: {str(e)}")
            return random.randint(0, 1)

    def add_mbot_receive_info(self, mbot_id: int, receive_info: dict, topic: str) -> None:
        """2.2.2 向机器人邻居节点添加接收信息记录"""
        try:
            self.mbot_cache[mbot_id]['receive info'][topic].append(receive_info)
        except Exception as e:
            self.logger.error(f"添加机器人接收信息时发生错误: {str(e)}")

    def process_mbot_share_actions(self, exposed_mbot_nodes: List[int], node: int, info: str, index: int, topic: str, mbotpostsratio: float) -> None:
        """2.3.1 处理机器人节点的行为,生成机器人节点的分享信息"""
        """
        format:
        {
            'action': 'Post',
            'content': str,
            'index': int,
            'information Plausibility': float,
            'stance': int
        }
        """
        mbotpostnodes = []
        mbotsharenodes = []
        # 2.3.1.1 生成机器人节点的Post和Share行为
        for i, mbotnode in enumerate(exposed_mbot_nodes):
            if self.mbot_cache[mbotnode]['receive info'][topic][i]['stance'] == 0:
                mbotpostnodes.append(mbotnode)
            if self.mbot_cache[mbotnode]['receive info'][topic][i]['stance'] == 1:
                v = 1 if random.random() < mbotpostsratio else 0
                if v == 1:
                    mbotpostnodes.append(mbotnode)
                else:
                    mbotsharenodes.append(mbotnode)
        actions = {}
        # 2.3.1.2 生成机器人节点的Post行为
        for mbotnode in mbotpostnodes:
            content = self.load_initial_disinfo(topic)
            actions[mbotnode] = {'action': 'Post', 'content': content, "index": index, 'information Plausibility': self.get_information_plausibility(content), 'stance': 1}
        # 2.3.1.3 生成机器人节点的Share行为
        for mbotnode in mbotsharenodes:
            data = {"information": info}
            result = self.process_share_llm_request(self.config.promptpaths['BotSharePrompt'], data)
            if result['Action'] == 'Repost':
                if "Repost:" not in info:
                    content = 'Repost:' + info
                else:
                    content = info
                stance = 1
                # 2.3.1.3.1 生成机器人节点的Repost行为，计算信息的自身说服力
                information_plausibility = self.get_information_plausibility(content)
            elif result['Action'] == 'Quote':
                content = "Comment: " + result['Comment'] + "Original post: " + info
                stance = 1
                # 2.3.1.3.2 生成机器人节点的Quote行为，计算信息的自身说服力
                information_plausibility = self.get_information_plausibility(content)
            actions[mbotnode] = {'action': result['Action'], 'content': content, "index": index, 'information Plausibility': information_plausibility, 'stance': stance}
        return actions

    def add_mbot_share_info(self, node: int, mbot_share_info: dict, topic: str) -> None:
        """0.2/2.3.2 向机器人邻居节点添加分享的信息记录"""
        """
        format:
        {
            'action': 'Post',
            'content': str,
            'index': int,
            'information Plausibility': float,
            'stance': int
        }
        """
        try:
            self.mbot_cache[node]['share info'][topic].append(mbot_share_info)
        except Exception as e:
            self.logger.error(f"添加机器人分享信息失败: {str(e)}")

    def add_lbot_share_info(self, node: int, lbot_share_info: dict, topic: str) -> None:
        """0.2/2.3.2 向机器人邻居节点添加分享的信息记录"""
        try:
            self.lbot_cache[node]['share info'][topic].append(lbot_share_info)
        except Exception as e:
            self.logger.error(f"添加机器人分享信息失败: {str(e)}")

    def get_share_nodes(self, exposed_nodes: List[int], topic: str) -> List[int]:
        """2.4.1 获取会分享信息的节点"""
        share_nodes = []
        try:
            # 2.4.1.1 在暴露节点中选择获取会分享信息的节点
            for node in exposed_nodes:
                dissemination_tendency = self.human_cache[node]['dissemination tendency'][-1][list(self.human_cache[node]['dissemination tendency'][-1].keys())[0]]
                # 设置随机种子
                # random.seed(node)
                # 生成随机数
                random_num = random.random()
                # 如果随机数小于传播倾向，则将节点加入分享节点列表
                if random_num < dissemination_tendency[topic]:
                    share_nodes.append(node)
        except Exception as e:
            self.logger.error(f"获取分享节点时发生错误: {str(e)}")
        
        return share_nodes

    def process_share_nodes(self, share_nodes: List[int], topic: str, 
                          plausibility: float, stance: int, index: int) -> Tuple[List[int], List[int]]:
        """2.4.2 处理分享节点的信任度计算"""
        trust_nodes = []
        unbelieve_nodes = []
        
        try:
            for node in share_nodes:

                # 计算信任阈值
                # tt_score = self.calculate_trust_threshold(node, comm, self.config.parameters['gamma'], self.config.parameters['alpha'], self.config.parameters['beta'])
                tt_key = list(self.human_cache[node]['trust threshold'].keys())[-1]
                tt_score = self.human_cache[node]['trust threshold'][tt_key][topic]
                
                # 根据信任阈值和信息可信度决定节点的信任状态
                if stance == 1:
                    detect_fake_prob = 1 - (1 - tt_score) * plausibility
                else:
                    detect_fake_prob = 1 - (1 - tt_score) * (1 - plausibility)
                
                if random.random() < detect_fake_prob:
                    unbelieve_nodes.append(node)
                else:
                    trust_nodes.append(node)
                
                # 更新节点的信任阈值
                # self.update_trust_threshold(node, topic, tt_score, index)
                
        except Exception as e:
            self.logger.error(f"处理分享节点时发生错误: {str(e)}")
            
        return trust_nodes, unbelieve_nodes

    def process_human_share_actions(self, trust_nodes: Set[int], unbelieve_nodes: Set[int], info: str, topic: str) -> Tuple[List[int], List[int]]:
        """2.4.3 形成人类用户分享信息"""
        # 2.4.3.1 形成人类用户分享信息
        trust_actions = {}
        unbelieve_actions = {}

        for n in trust_nodes:
            user_data = {
                "information": info,"repost_number": self.human_cache[n]['retweet_num'], "quote_number": self.human_cache[n]['quote_num'],
                "history_info": self.human_cache[n]['user_text']
            }
            # 2.4.3.2 使用LLM生成人类用户分享信息
            result = self.process_share_llm_request(self.config.promptpaths['TrustPrompt'], user_data)
            if result['Action'] == 'Repost':
                if "Repost:" not in info:
                    content = "User:" + str(n) + " Repost:" + info
                else:
                    content = info
                
            elif result['Action'] == 'Quote':  
                if "Original post:" not in info:
                    content = "User:" + str(n) + " Comment: " + result['Comment'] + "\n" + "Original post: " + info
                else:
                    content = "User:" + str(n) + " Comment: " + result['Comment'] + "\n" + info
            trust_actions[n] = {"Action": result['Action'], "Content": content}
        
        for n in unbelieve_nodes:
            user_data = {
                "information": info,
                "history_info": self.human_cache[n]['user_text']
            }
            # 2.4.3.3 使用LLM生成人类用户引用信息
            result = self.process_llm_request(self.config.promptpaths['UnbelievePrompt'], user_data)
            if len(result) == 0:
                self.logger.error(f"生成人类用户引用信息失败")
                result = {'Action': 'Quote', 'Comment': "I do not believe this information"}
            if "Original post:" not in info:
                content = "User:" + str(n) + " Comment: " + result['Comment'] + "\n" + "Original post: " + info
            else:
                content = "User:" + str(n) + " Comment: " + result['Comment'] + "\n" + info
            unbelieve_actions[n] = {"Action": result['Action'], "Content": content}


        return trust_actions, unbelieve_actions

    def process_share_llm_request(self, prompt_file: str, user_data: dict) -> dict:
        """处理分享行为的LLM请求"""
        try:
            prompt, input_vars, attribute = self.load_prompt(prompt_file)
            if not prompt:
                return {'Action': 'Repost', 'Reason': ""}
                
            result = self.llm_manager.process_prompt(prompt, user_data, self.logger)
            if not result:
                return {'Action': 'Repost', 'Reason': ""}
            
            # 定义不同动作所需的属性
            required_attributes = {
                'Repost': ['Action', 'Reason'],
                'Quote': ['Action', 'Comment']
            }
            
            action = result.get('Action', 'Repost')
            required_attrs = required_attributes.get(action, ['Action', 'Reason'])
            
            if not all(attr in result for attr in required_attrs):
                self.logger.error(f"LLM返回结果缺少必要属性: {required_attrs}")
                return {'Action': 'Repost', 'Reason': ""}
                
            return result
            
        except Exception as e:
            self.logger.error(f"处理分享行为的LLM请求失败: {str(e)}")
            return {'Action': 'Repost', 'Reason': ""}


    def add_human_share_info(self, trust_actions: Dict[int, Dict[str, str]], unbelieve_actions: Dict[int, Dict[str, str]], topic: str, index: int) -> None:
        """2.4.4 添加人类用户分享信息记录"""
        try:
            
            for node in trust_actions:
                humanaddshareinfo = {}
                humanaddshareinfo['index'] = index
                humanaddshareinfo['action'] = trust_actions[node]['Action']
                humanaddshareinfo['content'] = trust_actions[node]['Content']
                humanaddshareinfo['information Plausibility'] = self.get_information_plausibility(trust_actions[node]['Content'])
                humanaddshareinfo['stance'] = 1
                self.human_cache[node]['share info'][topic].append(humanaddshareinfo)
            for node in unbelieve_actions:
                humanaddshareinfo = {}
                humanaddshareinfo['index'] = index
                humanaddshareinfo['action'] = unbelieve_actions[node]['Action']
                humanaddshareinfo['content'] = unbelieve_actions[node]['Content']
                humanaddshareinfo['information Plausibility'] = self.get_information_plausibility(unbelieve_actions[node]['Content'])
                humanaddshareinfo['stance'] = 0
                self.human_cache[node]['share info'][topic].append(humanaddshareinfo)
                        
        except Exception as e:
            self.logger.error(f"添加人类分享信息记录失败: {str(e)}")

    

    def update_all_trust_thresholds(self, trust_nodes: Set[int], 
                                  unbelieve_nodes: Set[int], 
                                  topic: str, index: int) -> None:
        """4. 批量更新所有节点的信任阈值"""
        try:
            # 4.1 串行更新所有信任节点的信任阈值
            for node in trust_nodes:
                self.update_trust_threshold(
                    node, topic, 
                    self.calculate_trust_threshold(node, topic, self.config.parameters['gamma'], self.config.parameters['delta'], self.config.parameters['beta']),  # 4.1.1 计算当前节点的信任阈值
                    index
                )
                
            # 4.2 串行更新所有不信任节点的信任阈值
            for node in unbelieve_nodes:
                self.update_trust_threshold(
                    node, topic, 
                    self.calculate_trust_threshold(node, topic, self.config.parameters['gamma'], self.config.parameters['delta'], self.config.parameters['beta']),  # 4.2.1 计算当前节点的信任阈值
                    index
                )
        except Exception as e:
            self.logger.error(f"批量更新信任阈值失败: {str(e)}")

    def update_trust_threshold(self, node: int, topic: str, 
                             new_score: float, index: int) -> None:
        """4.1/4.2 更新节点的信任阈值"""
        try:
            if str(index) not in self.human_cache[node]['trust threshold']:
                self.human_cache[node]['trust threshold'][str(index)] = {}
            self.human_cache[node]['trust threshold'][str(index)][topic] = new_score.item()

        except Exception as e:
            self.logger.error(f"更新信任阈值时发生错误: {str(e)}")

            
    def calculate_trust_threshold(self, node: int, topic: str, gamma: float, delta: float, beta: float) -> float:
        """4.1.1/4.2.1 计算节点的信任阈值"""
        try:
            support_sum_si_fj = 0
            oppose_sum_si_fj = 0
            # 4.1.1.2/4.2.1.2 加载最初的信任阈值
            begin_trust_threshold = self.human_cache[node]['trust threshold'].get(str(0), {}).get(topic, 0.5)
            # 4.1.1.3/4.2.1.3 加载节点接收到的信息
            for item in self.human_cache[node]['receive info'][topic]:
                if item['stance'] == 1:
                    support_sum_si_fj = support_sum_si_fj + item['social influence'] * item['info score']
                else:
                    oppose_sum_si_fj = oppose_sum_si_fj + item['social influence'] * item['info score']
            # 4.1.1.5/4.2.1.5 计算信任阈值, 支持虚假信息（置信度下降），反对虚假信息信任度上升
            decay = gamma * (1 - np.exp(-beta * support_sum_si_fj))
            enhancement = (1 - gamma) * (1 - np.exp(-delta * oppose_sum_si_fj))
            trust_threshold = np.clip(begin_trust_threshold + enhancement - decay, 0, 1)
            return trust_threshold

        except Exception as e:
            self.logger.error(f"计算信任阈值时发生错误: {str(e)}")
            return 0.5
    

    def get_information_plausibility(self, info: str) -> float:
        """虚假信息的plausibility"""
        try:
            result = self.process_llm_request(self.config.promptpaths['InformationPlausibility'], {"DisinformationText": info})
            if result is None:
                self.logger.error("LLM返回结果为空")
                return 0.5
            return result.get('CredibilityScore', 0.5) # 返回虚假信息可信度
        except Exception as e:
            self.logger.error(f"计算虚假信息可信度时发生错误: {str(e)}")
            return 0.5

    def log_token_usage(self, operation: str) -> None:
        """记录token使用情况"""
        current_tokens = self.llm_manager.get_token_usage()
        self.token_usage_history.append({
            'operation': operation,
            'tokens': current_tokens,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        self.logger.info(f"{operation} 使用了 {current_tokens} tokens")

    def get_token_usage_summary(self) -> str:
        """获取token使用情况摘要"""
        if not self.token_usage_history:
            return "没有token使用记录"
        
        total_tokens = self.llm_manager.get_token_usage()
        summary = f"总token使用量: {total_tokens}\n"
        summary += "详细使用记录:\n"
        for record in self.token_usage_history:
            summary += f"- {record['operation']}: {record['tokens']} tokens ({record['timestamp']})\n"
        return summary

    def reset_token_counters(self) -> None:
        """重置token计数器"""
        self.llm_manager.reset_token_counter()
        self.token_usage_history = []
    
    def update_dissemination_tendency(self, trust_nodes: Set[int], unbelieve_nodes: Set[int], topic: str, index: int) -> None:
        """更新传播倾向"""
        try:
            # 1. 获取用户初始传播倾向值
            for node in trust_nodes:
                # 获取用户最近的传播倾向值
                dissemination_tendency = self.human_cache[node]['dissemination tendency'][-1][list(self.human_cache[node]['dissemination tendency'][-1].keys())[0]].copy()
                if len(self.human_cache[node]['receive info'][topic]) > self.config.parameters['n0']:
                                        # 更新传播倾向值,依据接触的次数
                    dissemination_tendency[topic] = float(dissemination_tendency[topic] * np.exp(-self.config.parameters['xi'] * 1))

                self.human_cache[node]['dissemination tendency'].append({index:dissemination_tendency})

            for node in unbelieve_nodes:
                # 获取用户最近的传播倾向值
                dissemination_tendency = self.human_cache[node]['dissemination tendency'][-1][list(self.human_cache[node]['dissemination tendency'][-1].keys())[0]].copy()
                if len(self.human_cache[node]['receive info'][topic]) > self.config.parameters['n0']:
                    
                    # 更新传播倾向值,依据接触的次数
                    dissemination_tendency[topic] = float(dissemination_tendency[topic] * np.exp(-self.config.parameters['xi'] * 1))
                self.human_cache[node]['dissemination tendency'].append({index:dissemination_tendency})
            
        except Exception as e:
            self.logger.error(f"更新传播倾向时发生错误: {str(e)}")
            
