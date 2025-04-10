# Defines functions that are repeatedly called within the current folder
from dotenv import load_dotenv
load_dotenv()
import numpy as np
#prepare llm
from langchain_openai import ChatOpenAI
import os
# get environment variables
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
api_key = os.getenv("OPENAI_API_KEY") 
api_base = os.getenv("OPENAI_API_BASE") 

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

llm = ChatOpenAI(
    model="deepseek-v3", # Better for JSON generation
    temperature=0.2,  # Lower for more consistent structure
    api_key = "sk-eECpiAchfppU3086eIYaqrG6mRUJtp3AsFhqZS0Zpv0JebCu",
    base_url = 'https://api.chatanywhere.org/#/',
    max_retries=2,  # Middle-level retry, Automatically retry the response with HTTP status code 5xx
    http_client=httpx.Client( # Configure the HTTP client explicitly
        timeout=20.0, # Timeout of a single request
        limits=httpx.Limits(max_connections=100), # Bottom retry
        transport=httpx.HTTPTransport(retries=2) # TCP connection disconnection is automatically handled
    )
)

# Outer retry
@retry(
    stop=stop_after_attempt(3),  # The maximum number of retries is three, Maximum retry times of the service layer
    wait=wait_exponential(multiplier=1, min=2, max=10),  # Avalanche avoidance
    retry=retry_if_exception_type(
        (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
    ), # triggering condition
    reraise=True
)
def safe_chain_invoke(chain, input_data):
    return chain.invoke(input_data)

# load config
def load_config(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)

import yaml
from pathlib import Path
from langchain.prompts import PromptTemplate
def validate_yaml(file_path: str):
    """Verify the YAML file validity"""
    try:
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        print("YAML file format is correct")
        return True
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e.problem} (row{e.problem_mark.line+1})")
        return False
    except Exception as e:
        print(f"Failed to read the file: {str(e)}")
        return False

def load_prompt(file_path):
    if validate_yaml(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
            json_prompt_template = yaml_content["template"]
            input_variables = yaml_content["input_variables"]
            attribute = yaml_content["attribute"]
        prompt = PromptTemplate.from_template(json_prompt_template)
        
        return prompt, input_variables, attribute
    else:
        return "Prompt error"

# Validate the JSON structure meets requirements
def validate_json_response(data, attribute):
    required_domains = {"Entertainment", "Technology", "Sports", 
                       "Business", "Politics", "Education"}
    
    if not isinstance(data, dict):
        return False
    if attribute not in data:
        return False
    
    received_domains = {item["domain"] for item in data[attribute]}
    return required_domains == received_domains

# Recursively converts all int64
def convert(obj):
    if isinstance(obj, np.int64):
        return int(obj)  
    elif isinstance(obj, list):
        return [convert(i) for i in obj]  # Recursively process the list
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}  # Recursively process the dictionary
    else:
        return obj

# Format the content list as an ordinal string
def format_numbered_items(items, title="", max_items=80):
    if not items:
        return f"{title} [no data]\n"
    
    numbered_items = []
    for idx, item in enumerate(items[:max_items], 1):
        # Clean up the text and add a sequence number
        cleaned = str(item).replace('\n', ' ').strip()
        numbered_items.append(f"{idx}. {cleaned}")
    
    return f"{title}:\n" + "\n" + " ".join(numbered_items) + "\n" + " "

# Process users' data and return validated JSON
from langchain_core.output_parsers import JsonOutputParser
def process_user(prompt, llm, required_keys, attribute, user_data):
    try:
        # Check for missing keys
        missing_keys = required_keys - set(user_data.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
        # Get and parse response
        # formatted_prompt = prompt.format(**user_data)
        chain = (prompt | llm | JsonOutputParser())
        # json_response = chain.invoke(user_data)        
        try:
            json_response = safe_chain_invoke(chain, user_data)
        except Exception as e:
            print(f"All retry failed: {type(e).__name__}: {str(e)}")
            return None

        # Validate response structure
        if validate_json_response(json_response, attribute):
            return json_response
        else:
            raise ValueError("Invalid response structure")
            
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return None
