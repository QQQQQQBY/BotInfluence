# truncated power law distribition + maximum normalization interest community 
from set_logger import *
import time_utils
import glob
from utils import load_config, load_prompt, llm, convert, validate_json_response, format_numbered_items, process_user
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
import json
import powerlaw
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
from tqdm import tqdm
# calculate truncated power law distribition

def dispose_input(interest_domain_scores):
    """
    Disposal of input data
    """
    result = {}
    for items in interest_domain_scores:
        try:
            result[items['domain']] = float(items['score'])
        except:
            # prevent the error of divide by zero
            result[items['domain']] = 1
    return result

def DT_main(config, logger):
    logger.info(f'DT Program start time:{time_utils.get_now_time()}')
    logger.info(" ")
    IC_scores_list = []
    jsons_files = glob.glob(config['paths']['RawDataset'] + '*.json')
    share_num = []
    users_list = []

    for input_file in tqdm(jsons_files):
        if "concathumantext" in input_file:
            continue
        logger.info(f'Start time of {input_file} file:{time_utils.get_now_time()}')
        logger.info(" ")
        with open(input_file, 'r', encoding='utf-8') as f:
            user_data_batch = json.load(f)

        # Formula: truncated power law score + interest community score
        # 1. load interest community score
        IC_file = config['newpaths']['IC_Json'] + (input_file.split(config['paths']['RawDataset'])[-1]).split("_time_truncation")[0] + "_interest_community.json"
        with open(IC_file, 'r', encoding='utf-8') as f:
            IC_data = json.load(f)
        
        # 2. calculate truncated power law score
        out_file = input_file.split("_time_truncation")[0]
        out_file = config['newpaths']['DT_Json'] + out_file.split(config['paths']['RawDataset'])[-1] + "_disseminate_tendency.json"

        # check if the output file exists
        if os.path.exists(out_file):
            logger.info(f"{out_file} file already exists, skipping...")
            continue

        # collect the number of retweets and quotes 
        results = []
        count = 0
        for user_datas in tqdm(user_data_batch):
            try:
                result = {}
                result['user_id'] = user_datas['user_id']
                result['id'] = count + 1
                count = count + 1
                # 1.1 load interest community score
                for IC_content in IC_data:
                    if str(user_datas['user_id']) == str(IC_content['user_id']):
                        result['interest_community'] = dispose_input(IC_content['interest_domain_scores'])
                        break

                # 1.2 Maximum normalization
                # max_IC_score = max(result['interest_community'].values())
                max_IC_score = 10
                for key, value in result['interest_community'].items():
                    result['interest_community'][key] = value/max_IC_score

                # 1.3 calculate truncated power law score
                result['post_num'] = len(user_datas['Posts'])
                result['retweet_num'] = len(user_datas['Retweet'])
                result['quote_num'] = len(user_datas['Quote'])

                # prepare for score calculation
                IC_scores_list.append(result['interest_community'])
                share_num.append(len(user_datas['Retweet']) + len(user_datas['Quote']))
                users_list.append(user_datas['user_id'])
                results.append(result)
            except Exception as e:
                logger.info(f"Error processing user {user_datas['user_id']}: {e}")
                pass
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=4, ensure_ascii=False)
        logger.info(f'Save to {out_file} file:{time_utils.get_now_time()}')
        logger.info(" ")
        logger.info(f'End time of {input_file} file:{time_utils.get_now_time()}')
    logger.info(f'End time of DT Program:{time_utils.get_now_time()}')
    return share_num, users_list, IC_scores_list

def truncated_power_law(x, alpha, lambda_):
    """Probability density function of truncated power law distribution (unnormalized)"""
    p = x ** (-alpha) * np.exp(-lambda_ * x)
    return p

def cdf(x, alpha, lambda_):
    """
        Calculate CDF of discrete truncated power law distribution
        :param x: Target value (integer)
        :param alpha: power law exponent
        :param lambda_: truncation parameter
        :param xmin: minimum threshold
        :param xmax: Summation upper bound
        :return: CDF value
    """
    pdf = truncated_power_law(x, alpha, lambda_)
    cdf = np.cumsum(pdf) / np.sum(pdf) 
    return cdf

def power_law(data):
    # 1. Generate histogram data, Histograms group data into discrete intervals, Visualize the distribution shape of the data
    fit = powerlaw.Fit(data, discrete=True)
    xmin = fit.power_law.xmin
    alpha = fit.power_law.alpha
    hist, bin_edges = np.histogram(data, bins=np.arange(xmin, max(data) + 2), density=True) #Box boundaries (from xmin to maximum +2, step size is 1) # density=True: Normalized to probability density
    # 2. Calculate the center of the histogram
    x_hist = (bin_edges[:-1] + bin_edges[1:])/2 
    
    # 3. Based on the least square method, the residual sum of squares between the predicted value and the actual data is minimized by adjusting the function parameters
    popt, _ = curve_fit(
        truncated_power_law, # Function to be fitted
        x_hist,  # independent variable
        hist,   # dependent variable
        p0=[alpha, 0.01],  # Initial value: standard power law alpha + lambda'
        bounds=([1, 0], [10, 1])  # Parameter range constraint
        )
    alpha_trunc, lambda_trunc = popt

    # 4. Calculate the normalized constant
    x_values = np.arange(xmin, max(data) + 1)
    unnormalized_pdf = truncated_power_law(x_values, alpha_trunc, lambda_trunc)
    C_trunc = 1 / np.sum(unnormalized_pdf)

    print(f"Optimal x_min: {xmin}")
    print("\n[Truncated power law distribution]")
    print(f"power exponent (alpha): {alpha_trunc:.3f}")
    print(f"truncation parameter (lambda): {lambda_trunc:.5f}")
    print(f"normalization constant (C): {C_trunc:.3f}")

    # 5. Distribution fitting comparison
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=np.arange(xmin, max(data) + 2), density=True, alpha=0.6, label="Empirical Data")
    P_trunc = C_trunc * truncated_power_law(x_values, alpha_trunc, lambda_trunc)
    # Draw fitting curves
    plt.plot(x_values, P_trunc, label=f"Truncated Power Law (α={alpha_trunc:.2f}, λ={lambda_trunc:.4f})", color="blue") 

    plt.xlabel("x: dissemination number")
    plt.ylabel("P(x): probability density")
    plt.title("Truncated Power Law Fit")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.savefig("UserBotAgentAttributes/Figure/truncated_power_law.png")

    # 6. CDF, cumulative distribution function
    cdf_values = cdf(x_values, alpha_trunc, lambda_trunc)
    plt.figure(figsize=(10, 6))
    plt.step(x_values, cdf_values, where='post', color='r', label='discrete CDF')
    plt.plot(x_values, cdf_values, 'b-', label='CDF')
    plt.xlabel('x (dissemination number)')
    plt.ylabel('CDF: P(X ≤ x)')
    plt.title('Cumulative distribution function of truncated power law distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig("UserBotAgentAttributes/Figure/truncated_power_law_cdf.png")

    return C_trunc, alpha_trunc, lambda_trunc, xmin, x_values, cdf_values

def calculate_DT(config, share_num, users_list, IC_scores_list, alpha, lambdas, xmin, theta, x_values, cdf_values, logger):
    output_file = config['newpaths']['DT_Json'] + "DT_Score.json"
    if os.path.exists(output_file):
        logger.info(f"{output_file} file already exists, skipping...")
        return "complete calculate DT score"
    x_values = list(x_values)
    cdf_values = list(cdf_values)
    DT_score_list = []
    community_list = ['Entertainment', 'Technology', 'Sports', 'Business', 'Education', 'Politics']
    for i in range(len(share_num)):
        x = share_num[i]
        if x < xmin:
            formulate_power_law_cdf = 0.0        
        if x >= xmin:
            formulate_power_law_cdf = cdf_values[x_values.index(x)]
        total_score = {}
        for community in community_list:
            total_score[community] = theta * formulate_power_law_cdf + (1 - theta) * IC_scores_list[i][community]
        user_DT = {}
        user_DT['user_id'] = users_list[i]
        user_DT['DTScore'] = total_score
        DT_score_list.append(user_DT)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert(DT_score_list), f, indent = 4, ensure_ascii = False)
    return DT_score_list

def concat_DT_score(config, DT_score_file, logger):
    output_file = config['newpaths']['DT_Json'] + "concat_disseminate_tendency.json"
    if os.path.exists(output_file):
        logger.info(f"{output_file} file already exists, skipping...")
        return "complete concatenate DT score"
    with open(DT_score_file, 'r', encoding='utf-8') as f:
        DT_score_list = json.load(f)
    jsons_files = glob.glob(config['newpaths']['DT_Json'] + '*.json')
    outcome = []
    count = 0
    for input_file in tqdm(jsons_files):
        if "DT_Score.json" in input_file:
            continue
        with open(input_file, 'r', encoding='utf-8') as f:
            user_data_batch = json.load(f)
        for user_data in user_data_batch:
            for DT_score in DT_score_list:
                if user_data['user_id'] == DT_score['user_id']:
                    user_data['id'] = count + 1
                    count = count + 1
                    user_data['DTScore'] = DT_score['DTScore']
                    user_data['post_num'] = user_data['post_num']
                    user_data['retweet_num'] = user_data['retweet_num']
                    user_data['quote_num'] = user_data['quote_num']     
                    outcome.append(user_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert(outcome), f, indent = 4, ensure_ascii = False)
    return "complete concatenate DT score"

def disseminate_tendency_main():
    # 1. load config
    config = load_config("UserBotAgentAttributes/config.yaml")
    logger = set_logger(config['log']['DT_log_file'])
    # 2. process all users
    share_num, users_list, IC_scores_list = DT_main(config, logger)
    # 3. power law distribution
    # 3.1 Filter out 0 values (power law distribution requires x > 0)
    if len(share_num) != 0 and len(users_list) != 0 and len(IC_scores_list) != 0:
        data = np.array(share_num)
        data = data[data > 0]
        # 3.2 Calculate the power law distribution
        C, alpha, lambdas, xmin, x_values, cdf_values = power_law(data)
        # 4. calculate DT score
        theta = config['hyparameters']['theta']
        DT_score_list = calculate_DT(config, share_num, users_list, IC_scores_list, alpha, lambdas, xmin, theta, x_values, cdf_values, logger)
    # 5. concate the DT score with the original data
    DT_score_file = config['newpaths']['DT_Json'] + "DT_Score.json"
    outcome = concat_DT_score(config, DT_score_file, logger)
    logger.info(outcome)

if __name__ == '__main__':
    disseminate_tendency_main()
