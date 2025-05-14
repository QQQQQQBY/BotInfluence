# Statistics the share number of each user

import json

def share_num_main():
    with open('Dataset/ConcatRawDataset/concatrawdata.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    share_num = {}
    for user_key in data.keys():
        user = data[user_key]
        if user['user_id'] not in share_num:
            share_num[user['user_id']] = {}
        share_num[user['user_id']]['retweet_num'] = len(user['Retweet'])
        share_num[user['user_id']]['quote_num'] = len(user['Quote'])

    with open('Dataset/ExperimentDataset/share_num.json', 'w', encoding='utf-8') as f:
        json.dump(share_num, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    share_num_main()

