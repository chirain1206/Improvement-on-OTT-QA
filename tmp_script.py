import json

if __name__ == '__main__':
    with open('preprocessed_data/train_table_segments.json', 'r') as f:
        data = json.load(f)

    with open('link_generator/all_passage_query.json', 'r') as f:
        querys = json.load(f)

    absent_row = []
    table_names = list(data.keys())
    for name in table_names:
        if name not in querys:
            absent_row.append(name)

    with open('absent_row.json', 'w') as f:
        json.dump(absent_row, f, indent=2)