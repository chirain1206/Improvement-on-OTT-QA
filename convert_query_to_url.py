from drqa import retriever
from tqdm import tqdm
from transformers import BertTokenizer
from multiprocessing import Pool
import argparse
import logging
import json

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

if __name__ == '__main__':
    logger.info('Initializing ranker...')
    ranker = retriever.get_class(args.option)(bm25_path=args.model, strict=False)
    all_query_url = {}
    with open('link_generator/all_passage_query.json', 'r') as f:
        querys = json.load(f)

    for segment_name, query_lst in tqdm(querys.items(), desc='Use BM25 to find url for each query',):
        all_query_url[segment_name] = []
        for query in query_lst:
            if len(ranker.closest_docs(query, 1)[0]) > 0:
                all_query_url[segment_name].append(ranker.closest_docs(query, 1)[0][0])

    with open('link_generator/all_url.json', 'w') as f:
        json.dump(all_query_url, f, indent=2)