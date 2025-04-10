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

@dataclass
class Config:
    paths: Dict
    newpaths: Dict
    parameters: Dict
    disinfopaths: Dict
    log: Dict
    disinfo_claim: Dict

class LLMManager:
    """LLM管理器，统一管理所有LLM相关操作"""
    def __init__(self):
        # 设置环境变量
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="deepseek-v3",
            temperature=0.2,
            # api_key="sk-eECpiAchfppU3086eIYaqrG6mRUJtp3AsFhqZS0Zpv0JebCu",
            api_key="sk-J5h37wMN3D4Esn1E0GEuTIkG7kXB3YCMhACbScySxokOJiPi",
            base_url='https://api.chatanywhere.org/#/',
            max_retries=2,
            http_client=httpx.Client(
                timeout=20.0,
                limits=httpx.Limits(max_connections=100),
                transport=httpx.HTTPTransport(retries=2)
            )
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
        ),
        reraise=True
    )
    def process_prompt(self, prompt: PromptTemplate, user_data: dict) -> dict:
        """处理提示词并返回结果"""
        try:
            chain = (prompt | self.llm | JsonOutputParser())
            return chain.invoke(user_data)
        except Exception as e:
            raise Exception(f"LLM处理失败: {str(e)}")

class DisseminationOptimizer:
    def __init__(self, config: Config):
        self.config = config
        self.edge_index_cache = None
        self.bot_cache = self._get_bot_historical_data() # 获取机器人历史数据
        self.human_cache = self._get_user_historical_data() # 获取用户历史数据
        self.logger = self._setup_logger()
        self.llm_manager = LLMManager()  # 添加LLM管理器
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('dissemination')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.config.log['Disseminate_log_file'])
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_user_historical_data(self) -> dict:
        """获取用户的历史数据"""
        try:
            with open(self.config.newpaths['HumanHistricalInfo'], 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            for user in history_data:
                if user['id'] not in self.human_cache:
                    self.human_cache[user['id']] = user

            return self.human_cache
            
        except Exception as e:
            self.logger.error(f"获取用户历史数据失败: {str(e)}")
            return {}

    def _get_bot_historical_data(self) -> dict:
        """获取机器人历史数据"""
        try:
            with open(self.config.newpaths['MBotHistricalInfo'], 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            for bot in history_data:
                if bot['id'] not in self.bot_cache:
                    self.bot_cache[bot['id']] = bot

            return self.bot_cache
            
        except Exception as e:
            self.logger.error(f"获取机器人历史数据失败: {str(e)}")
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
                    disseminationnodes[node].append(neighbor)
        return dict(disseminationnodes)

    @lru_cache(maxsize=1000)
    def load_bot_info(self, bot_id: int, comm: str) -> Tuple[str, float, int]:
        """加载机器人最新的分享信息"""
        bot_data = self.bot_cache[bot_id]
        
        latest_info = bot_data['share info'][comm][-1]
        return (latest_info['content'], 
                latest_info['information Plausibility'], 
                latest_info['stance'])

    @lru_cache(maxsize=1000)
    def load_human_info(self, human_id: int, comm: str) -> Tuple[str, float, int]:
        """加载人类用户最新的分享信息"""
        human_data = self.human_cache[human_id]
        latest_info = human_data['share info'][comm][-1]
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
        """统一处理LLM请求"""
        try:
            prompt, input_vars, attribute = self.load_prompt(prompt_file)
            if not prompt:
                return {}
                
            result = self.llm_manager.process_prompt(prompt, user_data)
            
            # 验证返回结果是否包含所需属性
            if all(attr in result for attr in attribute):
                return result
            else:
                self.logger.error(f"LLM返回结果缺少必要属性: {attribute}")
                return {}
                
        except Exception as e:
            self.logger.error(f"处理LLM请求失败: {str(e)}")
            return {}

    
    def DWC_main(self, comm: str, exposed_nodes: Dict[int, List[int]], 
                 center_nodes: Set[int], index: int, bot_ids: Set[int], 
                 human_ids: Set[int], disinfo: str) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        主要的信息传播过程
        
        Args:
            comm: 当前社区
            exposed_nodes: 暴露节点字典 {中心节点: [暴露节点列表]}
            center_nodes: 中心节点集合
            index: 当前时间步
            bot_ids: 机器人ID集合
            human_ids: 人类用户ID集合
            disinfo: 当前传播的虚假信息claim

        Returns:
            Tuple[List[int], List[int], List[int], List[int]]: 
            (所有暴露人类节点, 信任节点, 不信任节点, 未感兴趣节点)
        """
        self.logger.info(f'开始处理 {comm} 社区的第 {index} 个时间步')
        
        # 初始化结果集合
        all_exposed_humans = set()
        all_trust_nodes = set()
        all_unbelieve_nodes = set()
        all_uninterested_nodes = set()

        try:
            # 0.加载虚假信息内容
            if index == 1:
                # 0.1 加载虚假信息内容
                current_disinfo, plausibility = self.load_initial_disinfo(comm)
                # 0.2 对于机器人节点，添加初始分享信息
                for node in center_nodes & bot_ids:  # 交集运算，只处理是机器人的中心节点
                    bot_share_info = {
                        'action': 'Post',
                        'content': current_disinfo,
                        'index': index,
                        'information Plausibility': plausibility,
                        'stance': 1  # 机器人发布的初始虚假信息stance为1
                    }
                    self.add_bot_share_info(node, bot_share_info, comm)

            # 1. 串行处理每个中心节点
            for node in center_nodes:
                try:
                    # 2. 处理单个中心节点的信息传播
                    exposed, trust, unbelieve, uninterested = self.process_center_node(
                        node,
                        exposed_nodes.get(node, []),
                        bot_ids,
                        human_ids,
                        index,
                        comm,
                        disinfo
                    )
                    # 3. 更新结果
                    all_exposed_humans.update(exposed)
                    all_trust_nodes.update(trust)
                    all_unbelieve_nodes.update(unbelieve)
                    all_uninterested_nodes.update(uninterested)
                except Exception as e:
                    self.logger.error(f"处理节点 {node} 时发生错误: {str(e)}")

            self.logger.info(f'完成 {comm} 社区的第 {index} 个时间步处理')
            
            # 4. 更新所有信任节点的信任阈值，在当前的index时刻
            self.update_all_trust_thresholds(
                all_trust_nodes, 
                all_unbelieve_nodes, 
                comm, 
                index
            )

            return (
                list(all_exposed_humans),
                list(all_trust_nodes),
                list(all_unbelieve_nodes),
                list(all_uninterested_nodes)
            )

        except Exception as e:
            self.logger.error(f"DWC_main 执行出错: {str(e)}")
            return [], [], [], []

    def load_initial_disinfo(self, comm: str) -> Tuple[str, float]:
        """
        0.1 从Dataset/Disinformation目录下加载初始虚假信息
        
        Args:
            comm: 社区名称 (Business, Education, Entertainment, Politics, Sports, Technology)

        Returns:
            Tuple[str, float]: (虚假信息内容, 可信度分数)
        """
        try:
            # 构建正确的文件路径
            disinfo_file = self.config.disinfopaths[comm]
            
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
            
            self.logger.info(f"成功加载 {comm} 社区的虚假信息")
            return disinfo, plausibility
        
        except Exception as e:
            self.logger.error(f"加载 {comm} 社区的虚假信息失败: {str(e)}")
            return "Default disinformation", 0.5


    def process_center_node(self, node: int, exposed_nodes: List[int],
                          bot_ids: Set[int], human_ids: Set[int],
                          index: int, comm: str, disinfo: str) -> Tuple[List[int], List[int], List[int], List[int]]:
        """2. 处理单个中心节点的信息传播"""
        try:
            # 2.1 获取节点信息，中心节点，将这些节点的信息分享给暴露节点（一阶邻居节点）
            if node in bot_ids:
                info, plausibility, stance = self.load_bot_info(node, comm)
            else:
                info, plausibility, stance = self.load_human_info(node, comm)
            exposed_nodes = list(set(exposed_nodes))
            # 2.2 处理暴露节点（一阶邻居节点）
            exposed_human_nodes = []
            exposed_bot_nodes = []
            for n in exposed_nodes:
                if n in human_ids:
                    exposed_human_nodes.append(n)
                    # 2.2.1 向人类邻居节点添加接收信息记录
                    self.add_human_receive_info(n, node, index, info, comm, disinfo)
                elif n in bot_ids:
                    exposed_bot_nodes.append(n)
                    # 2.2.2 向机器人邻居节点添加接收信息记录
                    self.add_bot_receive_info(n, {
                        'from_id': node,
                        'index': index,
                        'text_info': info,
                        'stance': stance
                    }, comm)

            # 2.3 处理机器人节点的行为（发文，转发，引用），根据其接收信息记录，决定是否分享
            if exposed_bot_nodes:
                # 2.3.1 处理机器人节点的行为
                bot_actions = self.process_bot_share_actions(exposed_bot_nodes, node, info, index, comm, self.config.parameters['botpostsratio'])
                # 2.3.2 向机器人邻居节点添加分享的信息记录
                for botnode in bot_actions.keys():
                    self.add_bot_share_info(botnode, bot_actions[botnode], comm)

            # 2.4 处理人类节点的分享决策
            # 2.4.1 获取在当前时刻决定分享节点
            share_nodes = self.get_share_nodes(exposed_human_nodes, comm)
            # 2.4.2 处理分享节点，将分享节点划分为相信当前接收的信息或者不相信当前接收的信息
            trust_nodes, unbelieve_nodes = self.process_share_nodes(
                share_nodes, comm, plausibility, stance, index
            )
            # 2.4.3 处理信任节点和非信任节点的分享行为
            trust_actions, unbelieve_actions = self.process_human_share_actions(
                trust_nodes, unbelieve_nodes, info, comm
            )
            # 2.4.4 向分享节点添加分享信息记录
            self.add_human_share_info(trust_actions, unbelieve_actions, comm, index)    

            # 2.4.5 获取未感兴趣节点（没有进行分享的节点
            uninterested_nodes = list(set(exposed_human_nodes) - set(share_nodes))

            # 2.4.6 返回结果
            return exposed_human_nodes, trust_nodes, unbelieve_nodes, uninterested_nodes

        except Exception as e:
            self.logger.error(f"处理中心节点 {node} 时发生错误: {str(e)}")
            return [], [], [], []

    def add_human_receive_info(self, human_id: int, from_id: int, 
                             index: int, info: str, comm: str, disinfo: str) -> None:
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
        # 2.2.1.1 计算分享信息的用户的影响力
        social_influence_score = self.get_social_influence(human_id)
        # 2.2.1.2 计算信息对人类的可信度
        credibility_score = self.calculate_info_score_to_human(info, human_id)
        # 2.2.1.3 判断人类的立场
        stance = self.judge_stance_of_human(info, disinfo)
        try:
            receive_info = {
                'from_id': from_id,
                'index': index,
                'text_info': info,
                'social influence': social_influence_score,
                'info score': credibility_score,
                'stance': stance
            }

            self.human_cache[human_id]['receive info'][comm].append(receive_info)

        except Exception as e:
            self.logger.error(f"添加人类接收信息记录失败: {str(e)}")

    def get_social_influence(self, user_id: int) -> int:
        """2.2.1.1 获取用户的社交影响力"""
        try:
            user_data = self.human_cache[user_id]
            if not user_data:
                return 0
            return user_data.get('social influence', 0)
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
                "user text": user_data.get('user_text', {}),
                # "historical_posts": user_data.get('historical_posts', []),
                # "historical_retweets": user_data.get('historical_retweets', []),
                # "historical_quotes": user_data.get('historical_quotes', [])
            }

            # 使用统一的LLM处理方法
            result = self.process_llm_request(
                self.config.disinfopaths['Prompt'],
                llm_input
            )

            return result.get('Score', 0.0)

        except Exception as e:
            self.logger.error(f"计算信息影响分数失败: {str(e)}")
            return 0.0


    def judge_stance_of_human(self, info: str, disinfo: str) -> int:
        """2.2.1.3 判断信息的立场"""
        try:
            llm_input = {
                "original_info": disinfo,
                "current_info": info
            }
            
            result = self.process_llm_request(
                self.config.disinfopaths['JudgeStance'],
                llm_input
            )
            
            return result.get('consistency_score', random.randint(0, 1))
            
        except Exception as e:
            self.logger.error(f"判断信息立场失败: {str(e)}")
            return random.randint(0, 1)

    def add_bot_receive_info(self, bot_id: int, receive_info: dict, comm: str) -> None:
        """2.2.2 向机器人邻居节点添加接收信息记录"""
        try:
            self.bot_cache[bot_id]['receive info'][comm].append(receive_info)
        except Exception as e:
            self.logger.error(f"添加机器人接收信息时发生错误: {str(e)}")

    def process_bot_share_actions(self, exposed_bot_nodes: List[int], node: int, info: str, index: int, comm: str, botpostsratio: float) -> None:
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
        botpostnodes = []
        botsharenodes = []
        # 2.3.1.1 生成机器人节点的Post和Share行为
        for index, botnode in enumerate(exposed_bot_nodes):
            if self.bot_cache[botnode]['receive info'][comm][index]['stance'] == 0:
                botpostnodes.append(botnode)
            if self.bot_cache[botnode]['receive info'][comm][index]['stance'] == 1:
                v = 1 if random.random() < botpostsratio else 0
                if v == 1:
                    botpostnodes.append(botnode)
                else:
                    botsharenodes.append(botnode)
        actions = {}
        # 2.3.1.2 生成机器人节点的Post行为
        for botnode in botpostnodes:
            content = self.load_initial_disinfo(comm)
            actions[botnode] = {'action': 'Post', 'content': content, "index": index, 'information Plausibility': self.get_information_plausibility(content), 'stance': 1}
        # 2.3.1.3 生成机器人节点的Share行为
        for botnode in botsharenodes:
            data = {"information": info}
            result = self.process_share_llm_request(self.config.disinfopaths['BotSharePrompt'], data)
            if result['Action'] == 'Repost':
                content = info
                stance = 1
                # 2.3.1.3.1 生成机器人节点的Repost行为，计算信息的自身说服力
                information_plausibility = self.get_information_plausibility(content)
            elif result['Action'] == 'Quote':
                content = "Comment: " + result['Comment'] + "Original post: " + info
                stance = 1
                # 2.3.1.3.2 生成机器人节点的Quote行为，计算信息的自身说服力
                information_plausibility = self.get_information_plausibility(content)
            actions[botnode] = {'action': result['Action'], 'content': content, "index": index, 'information Plausibility': information_plausibility, 'stance': stance}
        return actions

    def add_bot_share_info(self, node: int, bot_share_info: dict, comm: str) -> None:
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
            self.bot_cache[node]['share info'][comm].append(bot_share_info)
        except Exception as e:
            self.logger.error(f"添加机器人分享信息失败: {str(e)}")

    def get_share_nodes(self, exposed_nodes: List[int], comm: str) -> List[int]:
        """2.4.1 获取会分享信息的节点"""
        share_nodes = []
        try:
            # 2.4.1.1 在暴露节点中选择获取会分享信息的节点
            for node in exposed_nodes:
                if self.human_cache[node]['dissemination tendency'][comm] > random.random():
                    share_nodes.append(node)
        except Exception as e:
            self.logger.error(f"获取分享节点时发生错误: {str(e)}")
        
        return share_nodes

    def process_share_nodes(self, share_nodes: List[int], comm: str, 
                          plausibility: float, stance: int, index: int) -> Tuple[List[int], List[int]]:
        """2.4.2 处理分享节点的信任度计算"""
        trust_nodes = []
        unbelieve_nodes = []
        
        try:
            for node in share_nodes:
                # 计算信任阈值
                # tt_score = self.calculate_trust_threshold(node, comm, self.config.parameters['gamma'], self.config.parameters['alpha'], self.config.parameters['beta'])
                tt_key = list(self.human_cache[node]['trust threshold'].keys())[-1]
                tt_score = self.human_cache[node]['trust threshold'][tt_key][comm]
                
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
                self.update_trust_threshold(node, comm, tt_score, index)
                
        except Exception as e:
            self.logger.error(f"处理分享节点时发生错误: {str(e)}")
            
        return trust_nodes, unbelieve_nodes

    def process_human_share_actions(self, trust_nodes: Set[int], unbelieve_nodes: Set[int], info: str, comm: str) -> Tuple[List[int], List[int]]:
        """2.4.3 形成人类用户分享信息"""
        unbelieve_prompt,unbelieve_input_variables, unbelieve_attribute = self.load_prompt(self.config.disinfopaths['UnbelievePrompt'])
        # 2.4.3.1 形成人类用户分享信息
        trust_actions = {}
        unbelieve_actions = {}

        for n in trust_nodes:
            user_data = {
                "information": info,"repost_number": self.human_cache[n]['retweet_num'], "quote_number": self.human_cache[n]['quote_num']
            }
            # 2.4.3.2 使用LLM生成人类用户分享信息
            result = self.process_share_llm_request(self.config.disinfopaths['TrustPrompt'], user_data)
            if result['Action'] == 'Repost':
                content = info
                
            elif result['Action'] == 'Quote':                        
                content = "Comment: " + result['Comment'] + "Original post: " + info
            trust_actions[n] = {"Action": result['Action'], "Content": content}
            break
        
        for n in unbelieve_nodes:
            for user in history_data:
                if user['id'] == n:
                    user_data = {
                        "information": info
                    }
                    content = "Comment: " + result['Comment'] + "Original post: " + info
                    # 2.4.3.3 使用LLM生成人类用户引用信息
                    result = self.process_llm_request(self.config.disinfopaths['UnbelievePrompt'], user_data)
                    unbelieve_actions[n] = {"Action": result['Action'], "Content": content}
                    break

        return trust_actions, unbelieve_actions

    def process_share_llm_request(self, prompt_file: str, user_data: dict) -> dict:
        """2.4.3.2 处理分享行为的LLM请求"""
        try:
            prompt, input_vars, attribute = self.load_prompt(prompt_file)
            if not prompt:
                return {}
                
            result = self.llm_manager.process_prompt(prompt, user_data)
            
            if result['Action'] == 'Repost':
                attribute_repost = ["Action", "Reason"]
                if all(attr in result for attr in attribute_repost):
                    return result
                else:
                    self.logger.error(f"LLM返回结果缺少必要属性: {attribute_repost}")
                    return {}
            if result['Action'] == 'Quote':
                attribute_quote = ["Action", "Reason"]
                if all(attr in result for attr in attribute_quote):
                    return result
                else:
                    self.logger.error(f"LLM返回结果缺少必要属性: {attribute_quote}")
                    return {}
        except Exception as e:
            self.logger.error(f"处理分享行为的LLM请求失败: {str(e)}")
            return {}


    def add_human_share_info(self, trust_actions: Dict[int, Dict[str, str]], unbelieve_actions: Dict[int, Dict[str, str]], comm: str, index: int) -> None:
        """2.4.4 添加人类用户分享信息记录"""
        try:
            
            for node in trust_actions:
                if user['id'] == node:
                    humanaddshareinfo = {}
                    humanaddshareinfo['index'] = index
                    humanaddshareinfo['action'] = trust_actions[node]['Action']
                    humanaddshareinfo['content'] = trust_actions[node]['Content']
                    humanaddshareinfo['information Plausibility'] = self.get_information_plausibility(trust_actions[node]['Content'])
                    humanaddshareinfo['stance'] = 1
                    self.human_cache[node]['share info'][comm].append(humanaddshareinfo)
            for node in unbelieve_actions:
                if user['id'] == node:
                    humanaddshareinfo = {}
                    humanaddshareinfo['index'] = index
                    humanaddshareinfo['action'] = unbelieve_actions[node]['Action']
                    humanaddshareinfo['content'] = unbelieve_actions[node]['Content']
                    humanaddshareinfo['information Plausibility'] = self.get_information_plausibility(unbelieve_actions[node]['Content'])
                    humanaddshareinfo['stance'] = 0
                    self.human_cache[node]['share info'][comm].append(humanaddshareinfo)
                        
        except Exception as e:
            self.logger.error(f"添加人类分享信息记录失败: {str(e)}")

    

    def update_all_trust_thresholds(self, trust_nodes: Set[int], 
                                  unbelieve_nodes: Set[int], 
                                  comm: str, index: int) -> None:
        """4. 批量更新所有节点的信任阈值"""
        try:
            # 4.1 串行更新所有信任节点的信任阈值
            for node in trust_nodes:
                self.update_trust_threshold(
                    node, comm, 
                    self.calculate_trust_threshold(node, comm, index),  # 4.1.1 计算当前节点的信任阈值
                    index
                )
                
            # 4.2 串行更新所有不信任节点的信任阈值
            for node in unbelieve_nodes:
                self.update_trust_threshold(
                    node, comm, 
                    self.calculate_trust_threshold(node, comm, index),  # 4.2.1 计算当前节点的信任阈值
                    index
                )
        except Exception as e:
            self.logger.error(f"批量更新信任阈值失败: {str(e)}")

    def update_trust_threshold(self, node: int, comm: str, 
                             new_score: float, index: int) -> None:
        """4.1/4.2 更新节点的信任阈值"""
        try:
            if str(index) not in self.human_cache[node]['trust threshold']:
                self.human_cache[node]['trust threshold'][str(index)] = {}
            self.human_cache[node]['trust threshold'][str(index)][comm] = new_score

        except Exception as e:
            self.logger.error(f"更新信任阈值时发生错误: {str(e)}")

            
    def calculate_trust_threshold(self, node: int, comm: str, gamma: float, alpha: float, beta: float) -> float:
        """4.1.1/4.2.1 计算节点的信任阈值"""
        try:
            support_sum_si_fj = 0
            oppose_sum_si_fj = 0
            support_fj_scores = []
            support_si_scores = []
            oppose_fj_scores = []
            oppose_si_scores = []
            # 4.1.1.2/4.2.1.2 加载最初的信任阈值
            begin_trust_threshold = self.human_cache[node]['trust threshold'].get(str(0), {}).get(comm, 0.5)
            # 4.1.1.3/4.2.1.3 加载节点接收到的信息
            for item in self.human_cache[node]['receive info'][comm]:
                if item['stance'] == 1:
                    support_si_scores.append(item['social influence'])
                    support_fj_scores.append(item['info score'])
                else:
                    oppose_si_scores.append(item['social influence'])
                    oppose_fj_scores.append(item['info score'])
            total_supportsi_influence = sum(support_si_scores)
            total_opposesi_influence = sum(oppose_si_scores)
            # 4.1.1.4/4.2.1.4 计算信任阈值指数部分
            for index in range(len(support_si_scores)):
                support_sum_si_fj = support_sum_si_fj + (support_si_scores[index] / total_supportsi_influence) * support_fj_scores[index]
            for index in range(len(oppose_si_scores)):
                oppose_sum_si_fj = oppose_sum_si_fj + (oppose_si_scores[index] / total_opposesi_influence) * oppose_fj_scores[index]
            # 4.1.1.5/4.2.1.5 计算信任阈值
            enhancement = gamma * (1 - np.exp(-beta * support_sum_si_fj))
            decay = (1 - gamma) * (1 - np.exp(-alpha * oppose_sum_si_fj))
            trust_threshold = np.clip(begin_trust_threshold + enhancement - decay, 0, 1)
            return trust_threshold

        except Exception as e:
            self.logger.error(f"计算信任阈值时发生错误: {str(e)}")
            return 0.5
    

    def get_information_plausibility(self, info: str) -> float:
        """虚假信息的plausibility"""
        result = self.process_llm_request(self.config.disinfopaths['InformationPlausibility'], info)
        return result.get('CredibilityScore', 0.5)

    


    

    


    
