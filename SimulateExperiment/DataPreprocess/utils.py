import numpy as np
import yaml

# load config
def load_config(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)
    
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