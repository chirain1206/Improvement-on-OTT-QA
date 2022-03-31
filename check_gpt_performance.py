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
parser.add_argument('--evaluation_size', type=int, default=1000)
args = parser.parse_args()


if __name__ == '__main__':
    n_threads = os.cpu_count()

    with open('link_generator/train_url.json', 'r') as f:
        data = json.load(f)

    block_names = list(data.keys())
    random.shuffle(block_names)
    block_names = block_names[:args.evaluation_size]
    num_ground_truth_links = 0
    num_predict_correct = 0
    i = 0

    for name in block_names:
        i += 1
        table_name = name[:name.rfind('_')]
        row_index = int(name[name.rfind('_')+1:])
        with open(f'data/traindev_tables_tok/{table_name}', 'r') as f:
            cur_table = json.load(f)
        row_data = []
        ground_truth_links = []
        for cell in cur_table['data'][row_index]:
            ground_truth_links += cell[1]
        ground_truth_links = set(ground_truth_links)

        # only predict the percentage of links that are correctly predicted
        # (even if model generated too many links but it still won't affect the accuracy )
        for predict_links in data[name]:
            if predict_links in ground_truth_links:
                num_predict_correct += 1
        num_ground_truth_links += len(ground_truth_links)
        sys.stdout.write('finished {}/{}; Accuracy: = {} \r'.format(i, len(block_names), num_predict_correct / num_ground_truth_links))
    print('finished {}/{}; Accuracy: = {} \r'.format(i, len(block_names), num_predict_correct / num_ground_truth_links))