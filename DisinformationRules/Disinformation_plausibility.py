# discriminate the disinformation plausibiliy
from utils import load_config, load_prompt, llm, convert, process
import yaml


def disinformation_plausibility_main():
    BusinessDisinformation = config['paths']['BusinessDisinformation']
    EducationDisinformation = config['paths']['EducationDisinformation']
    EntertainmentDisinformation = config['paths']['EntertainmentDisinformation']
    PoliticsDisinformation = config['paths']['PoliticsDisinformation']
    SportsDisinformation = config['paths']['SportsDisinformation']
    TechnologyDisinformation = config['paths']['TechnologyDisinformation']
    Comm_Paths = [BusinessDisinformation, EducationDisinformation, EntertainmentDisinformation, PoliticsDisinformation, SportsDisinformation, TechnologyDisinformation]
    Comms = ["Business", "Education","Entertainment", "Politics", "Sports", "Technology"]

    for index, input_file in enumerate(Comm_Paths):
        prompt,input_variables = load_prompt(config['paths']['DisinformationPlausibilityPrompt'])
        with open(input_file, 'r', encoding='utf-8') as f:
            comm_info = yaml.safe_load(f)
            prompt_datas = comm_info['DisinformationDescribeList']
        results = []
        for prompt_data in prompt_datas:
            result = process(prompt, llm, input_variables, prompt_data)
            results.append(result['CredibilityScore'])
        print(Comms[index])
        print(results)

if __name__ == "__main__":
    config = load_config("DisinformationRules/config.yaml")
    disinformation_plausibility_main(config)
