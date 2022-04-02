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
# parser.add_argument('--model', type=str, required=True)
parser.add_argument('--option', type=str, default='bm25')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--max_block_len', type=int, default=512)
parser.add_argument('--retain_passage', action="store_true", default=False, help="Whether or not to retain passages following the improvement strategy")
args = parser.parse_args()

# logger.info('Initializing ranker...')
# ranker = retriever.get_class(args.option)(bm25_path=args.model, strict=False)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True, cache_dir='/tmp/')
tokenizer.add_tokens(["[TAB]", "[TITLE]", "[ROW]", "[MAX]", "[MIN]", "[EAR]", "[LAT]"])
with open('link_generator/all_passage_query.json', 'r') as f:
    querys = json.load(f)
with open('data/all_passages.json', 'r') as f:
    passages = json.load(f)

def fusion(cur_table_name):
    fused_block_dict = {}

    for row_index, row in enumerate(data[cur_table_name]):
        tokens = row[0]
        lst_type = row[1][-1]
        token_type = row[1]
        segment_name = cur_table_name + f'_{row_index}'
        extra_segment_name = "0"

        # if fused block size already greater than the limit before fusion, return it directly
        if len(tokens) > args.max_block_len:
            tokens = tokens[:args.max_block_len]
            token_type = token_type[:args.max_block_len]
            token_mask = [1] * len(tokens)
            if args.retain_passage:
                fused_block_dict[segment_name + '@' + extra_segment_name] = [tokens, token_type, token_mask]
            else:
                fused_block_dict[segment_name] = [tokens, token_type, token_mask]
            continue

        # find linked passages
        if segment_name in train_urls:
            linked_url = train_urls[segment_name]
            linked_passages = [passages[url] for url in linked_url]
        else:
            linked_passages = []

        # concatenate passages to end of table segment
        for cur_passage in linked_passages:
            passage_token = tokenizer.tokenize("[SEP] " + cur_passage)
            tokens += passage_token
            token_type += [int(not lst_type)] * len(passage_token)
            lst_type = int(not lst_type)
            if len(tokens) >= args.max_block_len:
                tokens = tokens[:args.max_block_len]
                token_type = token_type[:args.max_block_len]

                # Improvement strategy enables truncated passages to still link with original table segment
                if args.retain_passage:
                    token_mask = [1] * len(tokens)
                    fused_block_dict[segment_name + '@' + extra_segment_name] = [tokens, token_type, token_mask]
                    tokens = row[0]
                    lst_type = row[1][-1]
                    token_type = row[1]
                    extra_segment_name = str(int(extra_segment_name) + 1)
                else:
                    break

        token_mask = [1] * len(tokens)
        if args.retain_passage:
            fused_block_dict[segment_name + '@' + extra_segment_name] = [tokens, token_type, token_mask]
        else:
            fused_block_dict[segment_name] = [tokens, token_type, token_mask]

    return fused_block_dict

if __name__ == '__main__':
    fused_blocks = {}
    # one space leave for [CLS] token
    args.max_block_len = args.max_block_len - 1

    n_threads = 64
    if args.split == 'train' or args.split == 'dev':
        with open(f'preprocessed_data/{args.split}_table_segments.json', 'r') as f:
            data = json.load(f)
        with open(f'link_generator/{args.split}_url.json', 'r') as f:
            train_urls = json.load(f)

        with Pool(n_threads) as p:
            table_names = list(data.keys())
            results = list(
                tqdm(
                    p.imap(fusion, table_names),
                    total=len(table_names),
                    desc='Fusion Procedure',
                )
            )

        for _ in results:
            for row in _.keys():
                fused_blocks[row] = _[row]

        if args.retain_passage:
            with open(f'preprocessed_data/{args.split}_fused_blocks_retained.json', 'w') as f:
                json.dump(fused_blocks, f, indent=2)
        else:
            with open(f'preprocessed_data/{args.split}_fused_blocks.json', 'w') as f:
                json.dump(fused_blocks, f, indent=2)