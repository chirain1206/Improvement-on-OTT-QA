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
parser.add_argument('--max_query_len', type=int, default=128)
args = parser.parse_args()

PROCESS_TOK = None

# randomly drop half of the words from the sentence and return
def half_tokenize(text):
    global PROCESS_TOK
    tokens = PROCESS_TOK.tokenize(text).ngrams(n=1, uncased=True)
    indices = random.sample(range(len(tokens)), math.ceil(len(tokens) / 2))
    tokens = [tokens[i] for i in sorted(indices)]
    return ' '.join(tokens)

def init(tokenizer_class):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)

def generate_pseudo_train_sample(cur_fused_block):
    # corrupt the table segment
    tokens = '[CLS] '
    table_name = cur_fused_block[0][:cur_fused_block[0].rfind('_')]
    with open(f'data/traindev_tables_tok/{table_name}.json', 'r') as f:
        cur_table = json.load(f)
    half_title = half_tokenize(cur_table['title'])
    tokens += half_title + ' '
    row_index = int(cur_fused_block[0][cur_fused_block[0].rfind('_')+1:])
    for j, cell in enumerate(cur_table['data'][row_index]):
        half_cell = half_tokenize(cell[0])
        tokens += cur_table['header'][j][0] + ' is ' + half_cell + ' '
    tokens = bert_tokenizer.tokenize(tokens)
    token_type = [0] * len(tokens)

    # sample a sentence from the fused passage
    segment_urls = linked_urls[cur_fused_block[0]] if cur_fused_block[0] in linked_urls else []
    if len(segment_urls) > 0:
        sample_url = random.choice(segment_urls)
        sample_passage = passages[sample_url]
        if len(sample_passage) > 0:
            sample_sentence = '[SEP] ' + random.choice(sentence_tokenizer.tokenize(sample_passage)) + ' [SEP]'
        else:
            sample_sentence = '[SEP]'
    else:
        sample_sentence = '[SEP]'

    # produce pseudo query
    tokens += bert_tokenizer.tokenize(sample_sentence)
    token_type += [1] * (len(tokens) - len(token_type))
    token_mask = [1] * len(tokens)

    # truncate query length
    if len(tokens) > args.max_query_len:
        tokens = tokens[:args.max_query_len]
        tokens[-1] = '[SEP]'
        token_type = token_type[:args.max_query_len]
        token_mask = token_mask[:args.max_query_len]

    pseudo_query = [tokens, token_type, token_mask]

    # add [CLS] token to front of the fused block
    block_tokens = ["[CLS]"] + cur_fused_block[1][0]
    block_type = [0] + cur_fused_block[1][1]
    block_mask = [1] + cur_fused_block[1][2]
    block_repr = [block_tokens, block_type, block_mask]

    # return pseudo query and the original block
    return pseudo_query, block_repr

if __name__ == '__main__':
    n_threads = os.cpu_count()
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    with open('preprocessed_data/train_fused_blocks.json', 'r') as f:
        data = json.load(f)

    with open('link_generator/train_url.json', 'r') as f:
        linked_urls = json.load(f)

    with open('data/all_passages.json', 'r') as f:
        passages = json.load(f)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir='/tmp/')
    block_names = list(data.keys())
    random.shuffle(block_names)
    shuffle_data = [[name, data[name]] for name in block_names]

    tok_class = drqa.drqa_tokenizers.get_class('simple')
    with Pool(n_threads, initializer=init, initargs=(tok_class,)) as p:
        results = list(
            tqdm(
                p.imap_unordered(generate_pseudo_train_sample, shuffle_data),
                total=len(block_names),
                desc='Generate training samples for ICT',
            )
        )

    with open('retriever/ICT_pretrain_data.json', 'w') as f:
        json.dump(results, f, indent=2)