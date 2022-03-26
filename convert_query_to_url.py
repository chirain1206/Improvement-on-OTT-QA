from drqa import retriever
from tqdm import tqdm
from transformers import BertTokenizer
from multiprocessing import Pool
import argparse
import logging
import json
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--option', type=str, default='bm25')
parser.add_argument('--split', type=str, default='train')
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class(args.option)(bm25_path=args.model, strict=False)

def query_to_url(table_segment):
    segment_name = table_segment[0]
    query_lst = table_segment[1]
    url_lst = []

    for query in query_lst:
        if len(ranker.closest_docs(query, 1)[0]) > 0:
            url_lst.append(ranker.closest_docs(query, 1)[0][0])

    return [segment_name, url_lst]

if __name__ == '__main__':
    n_threads = os.cpu_count()
    with open('link_generator/all_passage_query.json', 'r') as f:
        data = json.load(f)

    if args.split == 'all':
        data = [[name, data[name]] for name in data]

        with Pool(n_threads) as p:
            all_query_url = list(
                tqdm(
                    p.imap(query_to_url, data),
                    total=len(data),
                    desc='Use BM25 to find url for each query',
                )
            )
        all_query_url = {segment_name:url_lst for segment_name, url_lst in all_query_url}

        with open(f'link_generator/{args.split}_url.json', 'w') as f:
            json.dump(all_query_url, f, indent=2)
    elif args.split == 'train':
        train_names = []
        with open('preprocessed_data/train_table_segments.json', 'r') as f:
            train_segments = json.load(f)

        # extract names of table segments to be trained
        for segment_name in train_segments:
            for row_index, row in enumerate(train_segments[segment_name]):
                train_names.append(segment_name + f'_{row_index}')

        data = [[name, data[name]] for name in train_names if name in data]

        with Pool(n_threads) as p:
            all_query_url = list(
                tqdm(
                    p.imap(query_to_url, data),
                    total=len(data),
                    desc='Use BM25 to find url for each query',
                )
            )
        all_query_url = {segment_name:url_lst for segment_name, url_lst in all_query_url}

        with open(f'link_generator/{args.split}_url.json', 'w') as f:
            json.dump(all_query_url, f, indent=2)