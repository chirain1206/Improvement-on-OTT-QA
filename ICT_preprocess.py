from drqa import retriever
from tqdm import tqdm
from transformers import BertTokenizer
from multiprocessing import Pool
import argparse
import logging
import json
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

def generate_pseudo_train_sample(cur_fused_block):


    return pseudo_query, cur_fused_block[1]

if __name__ == '__main__':
    n_threads = 64

    with open('preprocessed_data/train_fused_blocks.json', 'r') as f:
        data = json.load(f)

    block_names = list(data.keys())
    random.shuffle(block_names)
    shuffle_data = [[name, data[name]] for name in block_names]

    with Pool(n_threads) as p:
        results = list(
            tqdm(
                p.imap_unordered(generate_pseudo_train_sample, shuffle_data),
                total=len(block_names),
                desc='Generate training samples for ICT',
            )
        )