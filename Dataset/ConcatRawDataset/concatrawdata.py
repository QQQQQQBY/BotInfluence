# Concat the raw dataset

import json
import glob

def concat_raw_data():
    concat_data = {}
    json_files = glob.glob('BotInfluence-main/Dataset/RawDataset/*.json')
    count = 0
    for file in json_files:
        if "concathumantext" in file:
            continue
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for user in data:
            if str(user['user_id']) not in concat_data:
                concat_data[str(user['user_id'])] = {}
            concat_data[str(user['user_id'])] = user
            concat_data[str(user['user_id'])]['id'] = count
            count += 1

    output_file = 'BotInfluence-main/Dataset/ConcatRawDataset/concatrawdata.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(concat_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    concat_raw_data()

