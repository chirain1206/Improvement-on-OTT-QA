from preprocessing import *
from drqa import retriever
import json
import sys
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, type=str)
    args = parser.parse_args()
    with open(f'data/all_plain_tables.json', 'r') as f:
        plain_tables = json.load(f)

    n_threads = 64
    if args.split == 'train':
        dict_results = {}
        print("using {}".format(args.split))
        if not os.path.exists(f'preprocessed_data/{args.split}_table_segments.json'):
            with open(f'released_data/{args.split}.traced.json', 'r') as f:
                data = json.load(f)

            # extract tables corresponding to each training question
            data = [plain_tables[cur['table_id']] for cur in data]
            results = []
            with Pool(n_threads) as p:
                results = list(
                    tqdm(
                        p.imap(split_table_segment, data, chunksize=16),
                        total=len(data),
                        desc="convert table to table segments",
                    )
                )

            for table in tqdm(results):
                dict_results[table[0]] = table[1:]  # first element in table indicate the uid of the table

            with open(f'preprocessed_data/{args.split}_table_segments.json', 'w') as f:
                json.dump(dict_results, f, indent=2)
            
    elif args.split == 'all':
        dict_results = {}
        print("using {}".format(args.split))
        if not os.path.exists(f'preprocessed_data/{args.split}_table_segments.json'):
            with open(f'released_data/{args.split}.traced.json', 'r') as f:
                data = json.load(f)

            # extract tables corresponding to each training question
            data = list(plain_tables.values())
            results = []
            with Pool(n_threads) as p:
                results = list(
                    tqdm(
                        p.imap(split_table_segment, data, chunksize=16),
                        total=len(data),
                        desc="convert table to table segments",
                    )
                )

            for table in tqdm(results):
                dict_results[table[0]] = table[1:]  # first element in table indicate the uid of the table

            with open(f'preprocessed_data/{args.split}_table_segments.json', 'w') as f:
                json.dump(dict_results, f, indent=2)
    else:
        raise NotImplementedError
