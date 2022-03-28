from drqa import retriever
from tqdm import tqdm
from transformers import BertTokenizer
from multiprocessing import Pool
import argparse
import logging
import json
import random
import nltk.data
import drqa.drqa_tokenizers
import math
import os
from multiprocessing.util import Finalize
from collections import Counter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--max_query_len', type=int, default=128)
args = parser.parse_args()

def generate_train_sample(trace_question):
    question = trace_question['question']
    table_id = trace_question['table_id']
    answer_node = trace_question['answer-node']

    # treat the answer row with most answer nodes as the ground-truth answer row
    answer_row = []
    for node in answer_node:
        answer_row.append(node[1][0])
    answer_row = Counter(answer_row).most_common(1)[0][0]
    ground_truth_block = fused_blocks[table_id + f'_{answer_row}']

    # preprocess the question
    query_tokens = '[CLS] ' + question + ' [SEP]'
    query_tokens = bert_tokenizer.tokenize(query_tokens)
    query_types = [0] * len(query_tokens)
    query_masks = [1] * len(query_tokens)

    # truncate query length
    if len(query_tokens) > args.max_query_len:
        query_tokens = query_tokens[:args.max_query_len]
        query_tokens[-1] = '[SEP]'
        query_types = query_types[:args.max_query_len]
        query_masks = query_mask[:args.max_query_len]

    query = [query_tokens, query_types, query_masks]

    return query, ground_truth_block

if __name__ == '__main__':
    n_threads = os.cpu_count()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='/tmp/')

    with open('released_data/train.traced.json', 'r') as f:
        data = json.load(f)

    with open('preprocessed_data/train_fused_blocks.json', 'r') as f:
        fused_blocks = json.load(f)

    with Pool(n_threads) as p:
        results = list(
            tqdm(
                p.imap_unordered(generate_train_sample, data),
                total=len(data),
                desc='Generate fine-tune samples for retriever',
            )
        )

    with open('retriever/fine_tune_pretrain_data.json', 'w') as f:
        json.dump(results, f, indent=2)