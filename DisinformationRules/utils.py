import numpy as np
#prepare llm
from langchain_openai import ChatOpenAI
import os
# get environment variables
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

llm = ChatOpenAI(
    model="gpt-4o-mini", # Better for JSON generation
    temperature=0.2,  # Lower for more consistent structure
    api_key = "sk-eECpiAchfppU3086eIYaqrG6mRUJtp3AsFhqZS0Zpv0JebCu",
    base_url = 'https://api.chatanywhere.tech/v1',
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
        prompt = PromptTemplate.from_template(json_prompt_template)
        
        return prompt, input_variables
    else:
        return "Prompt error"


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

from langchain_core.output_parsers import JsonOutputParser
def process(prompt, llm, input_variables, prompt_data):
    try:
        chain = (prompt | llm | JsonOutputParser())
        try:
            json_response = safe_chain_invoke(chain, prompt_data)
            return json_response
        except Exception as e:
            print(f"All retry failed: {type(e).__name__}: {str(e)}")
            return None
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        return None