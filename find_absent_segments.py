import json
from tqdm import tqdm

if __name__ == '__main__':
    with open('preprocessed_data/train_table_segments.json', 'r') as f:
        data = json.load(f)

    with open('link_generator/all_passage_query.json', 'r') as f:
        querys = json.load(f)

    absent_rows = []
    table_names = list(data.keys())
    for cur_table_name in tqdm(table_names):
        for row_index, row in enumerate(data[cur_table_name]):
            if cur_table_name + f'_{row_index}' not in querys:
                absent_rows.append(cur_table_name + f'_{row_index}')

    with open('link_generator/absent_rows.json', 'w') as f:
        json.dump(absent_rows, f, indent=2)