import json
from langchain_openai import ChatOpenAI
import os
from tqdm import tqdm
from utils import load_config
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def summarize_humantext(config):
    llm = ChatOpenAI(
            model="deepseek-v3",
            temperature=0.2,
            api_key="sk-eECpiAchfppU3086eIYaqrG6mRUJtp3AsFhqZS0Zpv0JebCu",
            base_url='https://api.chatanywhere.org/#/',
        )

    with open(config['paths']['textinfo'], 'r', encoding='utf-8') as f:
        human_text = json.load(f)

    output_file = config['newpaths']['HumanSummerizeText']
    if os.path.exists(output_file):
        return
    
    results = {}
    for textkey in tqdm(list(human_text.keys())):
        historical_posts = human_text[textkey]['historical_posts']
        historical_retweets = human_text[textkey]['historical_retweets']
        historical_quotes = human_text[textkey]['historical_quotes']
        prompt = f"""
    Read the user's past posts, retweets and quotes, and provide a concise summary of their interests, frequently discussed topics, communication style, and possible opinions or attitudes.

    Output Requirements:
    1.Write in a natural, fluid style;
    2.Do not list posts one by one — just summarize the key patterns;
    3.Limit the final summary to no more than 300 words.
    4.Response must strictly follow the [Output Format]
    5.Output must be a string, don't include any other text.

    User's posts:
    {historical_posts}
    User's retweets:
    {historical_retweets}
    User's quotes:
    {historical_quotes}

    [Output Format]:
    Required Output Format:
        "TextSummary"

    """
        response = llm.invoke(prompt)
        results[textkey] = response.content
    # print(response)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # 1. load config
    config = load_config("SimulateExperiment/config.yaml")
    # 2. run
    summarize_humantext(config)


