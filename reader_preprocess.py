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
    answer = tokenizer.tokenize(trace_question['answer-text'])
    table_id = trace_question['table_id']
    used_row = set()
    start_index = end_index = -1
    for node in answer_nodes:
        answer_row = node[1][0]
        if answer_row in used_row:
            continue
        used_row.add(answer_row)

        fused_block_name = table_id + f'_{answer_row}'
        assert fused_block_name in fused_blocks

        answer_block = fused_blocks[fused_block_name]
        block_len_limit = args.max_block_len - len(question)
        answer_block_tokens = answer_block[0][:block_len_limit]

        # find answer position in the fused block (the answer should not be in title information)
        row_token_index = answer_block_tokens.index("[row]")
        start_index, end_index = find_sublst(answer_block_tokens[row_token_index:], answer)
        if start_index != -1:
            start_index += row_token_index + len(question)
            end_index += row_token_index + len(question)
            break

    if start_index == -1:
        return None
    else:
        output_tokens = question + answer_block_tokens
        output_types = [1] * (len(question) - 1) + [0] + answer_block[1][:block_len_limit]
        output_masks = [1] * len(question) + answer_block[2][:block_len_limit]

        assert output_tokens[start_index:end_index + 1] == answer
        assert len(output_tokens) == len(output_types)
        assert len(output_types) == len(output_masks)
        return [output_tokens, output_types, output_masks, start_index, end_index]

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
                p.imap_unordered(generate_reader_train_sample, data),
                total=len(data),
                desc='Generate training samples for reader',
            )
        )

    if not os.path.exists('reader'):
        os.makedirs('reader')
    with open('reader/fine_tune_data.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Sucessfully generate {len(results)} training samples.")