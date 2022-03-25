import json

if __name__ == '__main__':
    linked_passages = {}
    for i in range(8):
        with open(f'link_generator/row_passage_query.json-0000{i}-of-00008', 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = json.loads(line)
                linked_passages[tmp[0]] = tmp[1]

    with open(f'link_generator/all_passage_query.json', 'w') as f:
        json.dump(linked_passages, f)