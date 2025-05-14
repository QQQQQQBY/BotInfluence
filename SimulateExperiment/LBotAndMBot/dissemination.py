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
    """LLM Manager, unified management of all LLM related operations"""
    def __init__(self):
        self.total_tokens = 0  # Add token counter
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="deepseek-v3",
            temperature=0.9,
            api_key="sk-BsNA4JYA07xoFhalsusjTLbFLQ7gzFIA2ZDBlULod0gjPa2T",
            base_url='https://api.chatanywhere.tech/v1',
            max_retries=2,
            http_client=httpx.Client(
                timeout=20.0,
                limits=httpx.Limits(max_connections=100),
                transport=httpx.HTTPTransport(retries=2)
            )
        )

    def get_token_usage(self) -> int:
        """Get total token usage"""
        return self.total_tokens

    def reset_token_counter(self) -> None:
        """Reset token counter"""
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
        """Process prompt and return result
        
        Args:
            prompt: Prompt template
            user_data: User data
            max_retries: Maximum retry attempts, default is 3
            
        Returns:
            dict: LLM processing result
        """
        retries = 0
        while retries < max_retries:
            try:
                chain = (prompt | self.llm | JsonOutputParser())
                with get_openai_callback() as cb:
                    result = chain.invoke(user_data)
                    self.total_tokens += cb.total_tokens
                # Record token usage
                return result
            except Exception as e:
                retries += 1
                logger.error(f"LLM processing failed (attempt {retries}/{max_retries}): {str(e)}")
                if retries == max_retries:
                    logger.error("Maximum retry attempts reached, returning empty result")
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
        """Convert edge index to undirected graph"""
        if edge_index.size == 0:
            return edge_index
        reverse_edges = np.flip(edge_index, axis=0)
        return np.unique(np.concatenate([edge_index, reverse_edges], axis=1), axis=1)

    def load_neighbour_nodes(self, centernodes: Set[int], edge_index: np.ndarray) -> Dict[int, List[int]]:
        """Get neighbor nodes"""
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
        """Load bot's latest sharing information"""
        mbot_data = self.mbot_cache[mbot_id]
        
        latest_info = mbot_data['share info'][topic][-1]
        return (latest_info['content'], 
                latest_info['information Plausibility'], 
                latest_info['stance'])

    def load_lbot_info(self, lbot_id: int, topic: str) -> Tuple[str, float, int]:
        """Load bot's latest sharing information"""
        lbot_data = self.lbot_cache[lbot_id]
        
        latest_info = lbot_data['share info'][topic][-1]
        return (latest_info['content'], 
                latest_info['information Plausibility'], 
                latest_info['stance'])
    
    def load_human_info(self, human_id: int, topic: str) -> Tuple[str, float, int]:
        """Load human user's latest sharing information"""
        human_data = self.human_cache[human_id]
        # Load center node's latest shared information
        latest_info = human_data['share info'][topic][-1]
        return (latest_info['content'],
                latest_info['information Plausibility'],
                latest_info['stance'])

    
    def validate_yaml(self, promptfile: str) -> bool:
        """Validate if prompt file is valid"""
        try:
            path = Path(promptfile)
            with open(path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e.problem} (row{e.problem_mark.line+1})")
            return False
        except Exception as e:
            print(f"Failed to read the file: {str(e)}")
            return False
                
    def load_prompt(self, promptfile: str) -> Tuple[PromptTemplate, List[str], List[str]]:
        """Load prompt template"""
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
                self.logger.error(f"Prompt file validation failed: {promptfile}")
                return None, [], []
        except Exception as e:
            self.logger.error(f"Failed to load prompt: {str(e)}")
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
                        'stance': 1  # The initial false information stance released by the bot is 1
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
                            'stance': 1  # The initial false information stance released by the bot is 1
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
                                'stance': 0  # The initial false information stance released by the bot is 1
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
        """Rewrite false information"""
        try:
            result = self.process_llm_request(
                self.config.promptpaths['RewriteDisinformation'],
                {"disinformation": disinfo}
            )
            plausibility = self.get_information_plausibility(result.get('NewDisinformation', disinfo))
            return result.get('NewDisinformation', disinfo), plausibility
        except Exception as e:
            self.logger.error(f"Failed to rewrite false information: {str(e)}")
            return disinfo, 0.5

    def rewrite_correct_info(self, correct_info: str) -> Tuple[str, float]:
        """Rewrite legal bot sharing information"""
        try:
            result = self.process_llm_request(
                self.config.promptpaths['RewriteCorrectInformation'],
                {"correct_information": correct_info}
            )
            plausibility = self.get_information_plausibility(result.get('NewCorrectInformation', correct_info))
            return result.get('NewCorrectInformation', correct_info), plausibility
        except Exception as e:
            self.logger.error(f"Failed to rewrite legal bot sharing information: {str(e)}")
            return correct_info, 0.5
        
    def load_initial_disinfo(self, topic: str) -> Tuple[str, float]:
        """
        0.1 Load initial false information from Dataset/Disinformation directory
        
        Args:
            topic: Topic name (Business, Education, Entertainment, Politics, Sports, Technology)

        Returns:
            Tuple[str, float]: (False information content, Credibility score)
        """
        try:
            # Build correct file path
            disinfo_file = self.config.disinfopaths[topic]
            
            self.logger.info(f"Attempting to load false information from {disinfo_file}")
            
            # Check if file exists
            if not Path(disinfo_file).exists():
                self.logger.error(f"False information file does not exist: {disinfo_file}")
                return "Default disinformation", 0.5
            
            # Load false information file
            with open(disinfo_file, 'r', encoding='utf-8') as f:
                disinfo_data = yaml.safe_load(f)
            
            if not isinstance(disinfo_data, dict):
                self.logger.error(f"False information file format error: {disinfo_file}")
                return "Default disinformation", 0.5
            
            # Get false information list and credibility list
            disinfo_list = disinfo_data.get("DisinformationDescribeList", [])
            plausibility_list = disinfo_data.get("DisinformationPlausibility", [])
            
            if not disinfo_list or not plausibility_list:
                self.logger.error(f"False information file is empty: {disinfo_file}")
                return "Default disinformation", 0.5
            
            # Randomly select a false information
            index = random.randint(0, len(disinfo_list) - 1)
            disinfo = disinfo_list[index]
            plausibility = plausibility_list[index]
            
            self.logger.info(f"Successfully loaded false information for {topic} topic")
            return disinfo, plausibility
        
        except Exception as e:
            self.logger.error(f"Failed to load false information for {topic} topic: {str(e)}")
            return "Default disinformation", 0.5

    def load_lbot_correct_info(self, topic: str, correctstrategy: str) -> Tuple[str, float]:
        """Load bot's latest sharing information"""
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
        """Process single center node's information dissemination"""
        try:
            # 2.1 Get node information, center node, share this node's information with exposed nodes (first-degree neighbor nodes)
            if node in mbot_ids:
                info, plausibility, stance = self.load_mbot_info(node, topic)
            elif node in lbot_ids:
                info, plausibility, stance = self.load_lbot_info(node, topic)
            else:
                info, plausibility, stance = self.load_human_info(node, topic)
            exposed_nodes = list(set(exposed_nodes))
            # 2.2 Process exposed nodes (first-degree neighbor nodes)

            exposed_human_nodes = []
            for n in tqdm(exposed_nodes, desc="Processing exposed nodes"):
                if n in human_ids:
                    exposed_human_nodes.append(n)
                    # 2.2.1 Add received information record to human neighbor nodes
                    self.add_human_receive_info(n, node, index, info, topic, disinfo, stance)

            # 2.4 Process human node's sharing decision
            # 2.4.1 Get the nodes that decide to share at the current time
            share_nodes = self.get_share_nodes(exposed_human_nodes, topic)
            # 2.4.2 Process share nodes, divide share nodes into nodes that believe the current received information or do not believe the current received information
            trust_nodes, unbelieve_nodes = self.process_share_nodes(
                share_nodes, topic, plausibility, stance, index
            )
            # 2.4.3 Process trust nodes and non-trust nodes' sharing behavior
            trust_actions, unbelieve_actions = self.process_human_share_actions(
                trust_nodes, unbelieve_nodes, info, topic
            )
            # 2.4.4 Add sharing information record to share nodes
            self.add_human_share_info(trust_actions, unbelieve_actions, topic, index)    
            
            # 2.4.6 Return results
            return exposed_human_nodes, trust_nodes, unbelieve_nodes

        except Exception as e:
            self.logger.error(f"Error processing center node {node}: {str(e)}")
            return [], [], [], []

    def add_human_receive_info(self, human_id: int, from_id: int, 
                             index: int, info: str, topic: str, disinfo: str, stance: int) -> None:
        """Add received information record to human neighbor nodes"""
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
        # 2.2.1.1 Calculate the influence of the sharing information user on the current user (human_id)
        social_influence_score = self.get_social_influence(human_id, from_id)
        # 2.2.1.2 Calculate the credibility of information to humans
        credibility_score = self.calculate_info_score_to_human(info, human_id)
        # 2.2.1.3 Judge the stance of humans
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
            self.logger.error(f"Failed to add human receive information record: {str(e)}")

    def get_social_influence(self, user_id: int, from_id: int) -> int:
        """2.2.1.1 Get user's social influence"""
        try:
            user_data = self.human_cache[user_id]['neighbor_nodes_social_influence'][str(from_id)]
            if not user_data:
                return 0
            return user_data
        except Exception as e:
            self.logger.error(f"Failed to get user's social influence: {str(e)}")
            return 0

    def calculate_info_score_to_human(self, info: str, user_id: int) -> float:
        """2.2.1.2 Calculate the influence score of information to users"""
        try:
            # Get user's historical data
            user_data = self.human_cache[user_id]
            if not user_data:
                return 0.0

            # Build LLM input data
            llm_input = {
                "text_information": info,
                "history_info": user_data.get('user_text', {}),
            }

            # Use a unified LLM processing method
            result = self.process_llm_request(
                self.config.promptpaths['CalculateInfoScore'],
                llm_input
            )

            return result.get('Score', 0.5)

        except Exception as e:
            self.logger.error(f"Failed to calculate information influence score: {str(e)}")
            return 0.5


    def judge_stance_of_human(self, info: str, disinfo: str) -> int:
        """2.2.1.3 Judge the stance of information"""
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
            self.logger.error(f"Failed to judge the stance of information: {str(e)}")
            return random.randint(0, 1)

    def add_mbot_receive_info(self, mbot_id: int, receive_info: dict, topic: str) -> None:
        """2.2.2 Add received information record to bot neighbor nodes"""
        try:
            self.mbot_cache[mbot_id]['receive info'][topic].append(receive_info)
        except Exception as e:
            self.logger.error(f"Failed to add bot receive information: {str(e)}")

    def process_mbot_share_actions(self, exposed_mbot_nodes: List[int], node: int, info: str, index: int, topic: str, mbotpostsratio: float) -> None:
        """2.3.1 Process bot node's behavior, generate bot node's sharing information"""
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
        # 2.3.1.1 Generate bot node's Post and Share behavior
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
        # 2.3.1.2 Generate bot node's Post behavior
        for mbotnode in mbotpostnodes:
            content = self.load_initial_disinfo(topic)
            actions[mbotnode] = {'action': 'Post', 'content': content, "index": index, 'information Plausibility': self.get_information_plausibility(content), 'stance': 1}
        # 2.3.1.3 Generate bot node's Share behavior
        for mbotnode in mbotsharenodes:
            data = {"information": info}
            result = self.process_share_llm_request(self.config.promptpaths['BotSharePrompt'], data)
            if result['Action'] == 'Repost':
                if "Repost:" not in info:
                    content = 'Repost:' + info
                else:
                    content = info
                stance = 1
                # 2.3.1.3.1 Generate bot node's Repost behavior, calculate the credibility of information
                information_plausibility = self.get_information_plausibility(content)
            elif result['Action'] == 'Quote':
                content = "Comment: " + result['Comment'] + "Original post: " + info
                stance = 1
                # 2.3.1.3.2 Generate bot node's Quote behavior, calculate the credibility of information
                information_plausibility = self.get_information_plausibility(content)
            actions[mbotnode] = {'action': result['Action'], 'content': content, "index": index, 'information Plausibility': information_plausibility, 'stance': stance}
        return actions

    def add_mbot_share_info(self, node: int, mbot_share_info: dict, topic: str) -> None:
        """0.2/2.3.2 Add sharing information record to bot neighbor nodes"""
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
            self.logger.error(f"Failed to add bot sharing information: {str(e)}")

    def add_lbot_share_info(self, node: int, lbot_share_info: dict, topic: str) -> None:
        """0.2/2.3.2 Add sharing information record to bot neighbor nodes"""
        try:
            self.lbot_cache[node]['share info'][topic].append(lbot_share_info)
        except Exception as e:
            self.logger.error(f"Failed to add bot sharing information: {str(e)}")

    def get_share_nodes(self, exposed_nodes: List[int], topic: str) -> List[int]:
        """2.4.1 Get nodes that will share information"""
        share_nodes = []
        try:
            # 2.4.1.1 Select nodes that will share information from exposed nodes
            for node in exposed_nodes:
                dissemination_tendency = self.human_cache[node]['dissemination tendency'][-1][list(self.human_cache[node]['dissemination tendency'][-1].keys())[0]]
                if dissemination_tendency[topic] > random.random():
                    share_nodes.append(node)
        except Exception as e:
            self.logger.error(f"Failed to get share nodes: {str(e)}")
        
        return share_nodes

    def process_share_nodes(self, share_nodes: List[int], topic: str, 
                          plausibility: float, stance: int, index: int) -> Tuple[List[int], List[int]]:
        """2.4.2 Process the trust calculation of share nodes"""
        trust_nodes = []
        unbelieve_nodes = []
        
        try:
            for node in share_nodes:

                # Calculate trust threshold
                # tt_score = self.calculate_trust_threshold(node, comm, self.config.parameters['gamma'], self.config.parameters['alpha'], self.config.parameters['beta'])
                tt_key = list(self.human_cache[node]['trust threshold'].keys())[-1]
                tt_score = self.human_cache[node]['trust threshold'][tt_key][topic]
                
                # Determine the trust state of the node based on the trust threshold and the credibility of the information
                if stance == 1:
                    detect_fake_prob = 1 - (1 - tt_score) * plausibility
                else:
                    detect_fake_prob = 1 - (1 - tt_score) * (1 - plausibility)
                
                if random.random() < detect_fake_prob:
                    unbelieve_nodes.append(node)
                else:
                    trust_nodes.append(node)
                
                # Update the trust threshold of the node
                # self.update_trust_threshold(node, topic, tt_score, index)
                
        except Exception as e:
            self.logger.error(f"Failed to process share nodes: {str(e)}")
            
        return trust_nodes, unbelieve_nodes

    def process_human_share_actions(self, trust_nodes: Set[int], unbelieve_nodes: Set[int], info: str, topic: str) -> Tuple[List[int], List[int]]:
        """2.4.3 Form human user sharing information"""
        # 2.4.3.1 Form human user sharing information
        trust_actions = {}
        unbelieve_actions = {}

        for n in trust_nodes:
            user_data = {
                "information": info,"repost_number": self.human_cache[n]['retweet_num'], "quote_number": self.human_cache[n]['quote_num'],
                "history_info": self.human_cache[n]['user_text']
            }
            # 2.4.3.2 Use LLM to generate human user sharing information
            result = self.process_share_llm_request(self.config.promptpaths['TrustPrompt'], user_data)
            if result['Action'] == 'Repost':
                if "Repost:" not in info:
                    content = "User:" + str(n) + 'Repost:' + info
                else:
                    content = info
                
            elif result['Action'] == 'Quote':  
                if "Original post:" not in info:
                    content ="User:" + str(n) + "Comment: " + result['Comment'] +  "\n" + "Original post: " + info
                else:
                    content = "User:" + str(n) + "Comment: " + result['Comment'] + "\n" + info
            trust_actions[n] = {"Action": result['Action'], "Content": content}
        
        for n in unbelieve_nodes:
            user_data = {
                "information": info,
                "history_info": self.human_cache[n]['user_text']
            }
            # 2.4.3.3 Use LLM to generate human user quote information
            result = self.process_llm_request(self.config.promptpaths['UnbelievePrompt'], user_data)
            if len(result) == 0:
                self.logger.error(f"Failed to generate human user quote information")
                result = {'Action': 'Quote', 'Comment': "I do not believe this information"}
            if "Original post:" not in info:
                content = "User:" + str(n) + "Comment: " + result['Comment'] + "\n" + "Original post: " + info
            else:
                content = "User:" + str(n) + "Comment: " + result['Comment'] + "\n" + info
            unbelieve_actions[n] = {"Action": result['Action'], "Content": content}


        return trust_actions, unbelieve_actions

    def process_share_llm_request(self, prompt_file: str, user_data: dict) -> dict:
        """Process the LLM request for sharing behavior"""
        try:
            prompt, input_vars, attribute = self.load_prompt(prompt_file)
            if not prompt:
                return {'Action': 'Repost', 'Reason': ""}
                
            result = self.llm_manager.process_prompt(prompt, user_data, self.logger)
            if not result:
                return {'Action': 'Repost', 'Reason': ""}
            
            # Define the attributes required for different actions
            required_attributes = {
                'Repost': ['Action', 'Reason'],
                'Quote': ['Action', 'Comment']
            }
            
            action = result.get('Action', 'Repost')
            required_attrs = required_attributes.get(action, ['Action', 'Reason'])
            
            if not all(attr in result for attr in required_attrs):
                self.logger.error(f"LLM returned result missing required attributes: {required_attrs}")
                return {'Action': 'Repost', 'Reason': ""}
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process the LLM request for sharing behavior: {str(e)}")
            return {'Action': 'Repost', 'Reason': ""}


    def add_human_share_info(self, trust_actions: Dict[int, Dict[str, str]], unbelieve_actions: Dict[int, Dict[str, str]], topic: str, index: int) -> None:
        """2.4.4 Add human user sharing information record"""
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
            self.logger.error(f"Failed to add human sharing information record: {str(e)}")

    

    def update_all_trust_thresholds(self, trust_nodes: Set[int], 
                                  unbelieve_nodes: Set[int], 
                                  topic: str, index: int) -> None:
        """4. Batch update the trust threshold of all nodes"""
        try:
            # 4.1 Serial update the trust threshold of all trust nodes
            for node in trust_nodes:
                self.update_trust_threshold(
                    node, topic, 
                    self.calculate_trust_threshold(node, topic, self.config.parameters['gamma'], self.config.parameters['delta'], self.config.parameters['beta']),  # 4.1.1 计算当前节点的信任阈值
                    index
                )
                
            # 4.2 Serial update the trust threshold of all unbelieve nodes
            for node in unbelieve_nodes:
                self.update_trust_threshold(
                    node, topic, 
                    self.calculate_trust_threshold(node, topic, self.config.parameters['gamma'], self.config.parameters['delta'], self.config.parameters['beta']),  # 4.2.1 计算当前节点的信任阈值
                    index
                )
        except Exception as e:
            self.logger.error(f"Failed to batch update the trust threshold: {str(e)}")

    def update_trust_threshold(self, node: int, topic: str, 
                             new_score: float, index: int) -> None:
        """4.1/4.2 Update the trust threshold of the node"""
        try:
            if str(index) not in self.human_cache[node]['trust threshold']:
                self.human_cache[node]['trust threshold'][str(index)] = {}
            self.human_cache[node]['trust threshold'][str(index)][topic] = new_score.item()

        except Exception as e:
            self.logger.error(f"Failed to update the trust threshold: {str(e)}")

            
    def calculate_trust_threshold(self, node: int, topic: str, gamma: float, delta: float, beta: float) -> float:
        """4.1.1/4.2.1 Calculate the trust threshold of the node"""
        try:
            support_sum_si_fj = 0
            oppose_sum_si_fj = 0
            # 4.1.1.2/4.2.1.2 Load the initial trust threshold
            begin_trust_threshold = self.human_cache[node]['trust threshold'].get(str(0), {}).get(topic, 0.5)
            # 4.1.1.3/4.2.1.3 Load the information received by the node
            for item in self.human_cache[node]['receive info'][topic]:
                if item['stance'] == 1:
                    support_sum_si_fj = support_sum_si_fj + item['social influence'] * item['info score']
                else:
                    oppose_sum_si_fj = oppose_sum_si_fj + item['social influence'] * item['info score']
            # 4.1.1.5/4.2.1.5 Calculate the trust threshold, support false information (confidence下降), oppose false information trust degree上升
            decay = gamma * (1 - np.exp(-beta * support_sum_si_fj))
            enhancement = (1 - gamma) * (1 - np.exp(-delta * oppose_sum_si_fj))
            trust_threshold = np.clip(begin_trust_threshold + enhancement - decay, 0, 1)
            return trust_threshold

        except Exception as e:
            self.logger.error(f"Failed to calculate the trust threshold: {str(e)}")
            return 0.5
    

    def get_information_plausibility(self, info: str) -> float:
        """False information plausibility"""
        try:
            result = self.process_llm_request(self.config.promptpaths['InformationPlausibility'], {"DisinformationText": info})
            if result is None:
                self.logger.error("LLM returned result is empty")
                return 0.5
            return result.get('CredibilityScore', 0.5) # Return the credibility of false information
        except Exception as e:
            self.logger.error(f"Failed to calculate the plausibility of false information: {str(e)}")
            return 0.5

    def log_token_usage(self, operation: str) -> None:
        """Record token usage"""
        current_tokens = self.llm_manager.get_token_usage()
        self.token_usage_history.append({
            'operation': operation,
            'tokens': current_tokens,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        self.logger.info(f"{operation} 使用了 {current_tokens} tokens")

    def get_token_usage_summary(self) -> str:
        """Get token usage summary"""
        if not self.token_usage_history:
            return "No token usage record"
        
        total_tokens = self.llm_manager.get_token_usage()
        summary = f"Total token usage: {total_tokens}\n"
        summary += "Detailed usage records:\n"
        for record in self.token_usage_history:
            summary += f"- {record['operation']}: {record['tokens']} tokens ({record['timestamp']})\n"
        return summary

    def reset_token_counters(self) -> None:
        """Reset token counter"""
        self.llm_manager.reset_token_counter()
        self.token_usage_history = []
    
    def update_dissemination_tendency(self, trust_nodes: Set[int], unbelieve_nodes: Set[int], topic: str, index: int) -> None:
        """Update dissemination tendency"""
        try:
            # 1. Get the initial dissemination tendency value
            for node in trust_nodes:
                dissemination_tendency = self.human_cache[node]['dissemination tendency'][-1][list(self.human_cache[node]['dissemination tendency'][-1].keys())[0]].copy()
                if len(self.human_cache[node]['receive info'][topic]) > self.config.parameters['n0']:   
                    # Update dissemination tendency, based on the number of contacts
                    dissemination_tendency[topic] = float(dissemination_tendency[topic] * np.exp(-self.config.parameters['xi'] * 1))
                self.human_cache[node]['dissemination tendency'].append({index:dissemination_tendency})

            for node in unbelieve_nodes:
                dissemination_tendency = self.human_cache[node]['dissemination tendency'][-1][list(self.human_cache[node]['dissemination tendency'][-1].keys())[0]].copy()
                if len(self.human_cache[node]['receive info'][topic]) > self.config.parameters['n0']:
                    # Update dissemination tendency, based on the number of contacts
                    dissemination_tendency[topic] = float(dissemination_tendency[topic] * np.exp(-self.config.parameters['xi'] * 1))
                self.human_cache[node]['dissemination tendency'].append({index:dissemination_tendency})
            
        except Exception as e:
            self.logger.error(f"Failed to update dissemination tendency: {str(e)}")
            
