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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--max_block_len', type=int, default=512)
args = parser.parse_args()

def find_sublst(lst, sublst):
    sublst_len = len(sublst)
    for ind in (i for i, e in enumerate(lst) if e == sublst[0]):
        if lst[ind:ind + sublst_len] == sublst:
            return ind, ind + sublst_len - 1
    return -1, -1

def generate_reader_train_sample(trace_question):
    question = tokenizer.tokenize('[CLS] ' + trace_question['question'] + ' [SEP]')
    answer_nodes = trace_question['answer-node']
    table_id = trace_question['table_id']
    used_row = set()
    for node in answer_nodes:
        answer_row = node[1][0]
        used_row.add(answer_row)
        fused_block_name = table_id + f'_{answer_row}'
        if fused_block_name not in fused_blocks:
            continue
        answer_block = fused_blocks[fused_block_name]
        block_len_limit = args.max_block_len - len(question)



if __name__ == '__main__':
    n_threads = os.cpu_count()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='/tmp/')
    tokenizer.add_tokens(["[TAB]", "[TITLE]", "[ROW]", "[MAX]", "[MIN]", "[EAR]", "[LAT]"])
    with open('released_data/train.traced.json', 'r') as f:
        data = json.load(f)

    with open('preprocessed_data/train_fused_blocks.json', 'r') as f:
        fused_blocks = json.load(f)

    with Pool(n_threads) as p:
        results = list(
            tqdm(
                p.imap_unordered(generate_pseudo_train_sample, data),
                total=len(data),
                desc='Generate training samples for reader',
            )
        )

    if not os.path.exists('reader'):
        os.makedirs('reader')
    with open('reader/fine_tune_data.json', 'w') as f:
        json.dump(results, f, indent=2)