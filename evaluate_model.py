import argparse
import logging
import json
import os
import sys
import random
from train_retriever import VectorizeModel
from transformers import (BertConfig, BertTokenizer, BertModel, BertForQuestionAnswering)
from torch import nn
import torch

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--eval_size', type=int, default=1000)
parser.add_argument(
    "--load_model_path",
    default="",
    type=str,
    required=True,
    help="Path to the retriever model that have been trained"
)
parser.add_argument(
    "--load_reader_model_path",
    default="",
    type=str,
    required=True,
    help="Path to the reader model that have been trained"
)
parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pre-trained model or shortcut name selected in the list"
)
parser.add_argument(
    "--cache_dir",
    default="/tmp/",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--eval_option",
    default="",
    type=str,
    help="retriever, reader, or both",
)
parser.add_argument(
    "--top_k",
    default=15,
    type=int,
    help="k value for top_k retrieval",
)
parser.add_argument('--max_block_len', type=int, default=512)
args = parser.parse_args()
device = torch.device("cuda:0")
args.n_gpu = torch.cuda.device_count()
args.device = device

if __name__ == '__main__':
    with open('released_data/dev.traced.json', 'r') as f:
        data = json.load(f)
    random.shuffle(data)
    data = data[:args.eval_size]

    # load the matrix of candidates
    candidate_info = torch.load('./preprocessed_data/dev_candidates.pth')
    IDX2BLOCK = candidate_info['IDX2BLOCK']
    BLOCK2IDX = candidate_info['BLOCK2IDX']
    candidate_matrix = candidate_info['candidate_matrix']

    query_config = BertConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    query_tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=args.cache_dir
    )
    args.orig_dim = query_config.hidden_size
    args.proj_dim = 128

    query_model = VectorizeModel(BertModel, args.model_name_or_path, query_config, len(query_tokenizer),
                                 args.cache_dir,
                                 args.orig_dim, args.proj_dim)
    query_model_path = os.path.join(args.load_model_path, 'pytorch_model.bin')
    query_model.load_state_dict(torch.load(query_model_path))
    if args.n_gpu > 1:
        query_model = nn.DataParallel(query_model)
    query_model.to(args.device)
    query_model.eval()

    if args.eval_option == 'retriever':
        num_succ = 0
        num_fin_questions = 0
        for trace_question in data:
            answer_row = set()
            for node in trace_question['answer-node']:
                answer_row.add(node[1][0])
            answer_segment = set()
            for row in answer_row:
                answer_segment.add(trace_question['table_id'] + f'_{row}')

            # compute vector for the question
            query = trace_question['question']
            query_tokens = '[CLS] ' + query + ' [SEP]'
            query_tokens = query_tokenizer.tokenize(query_tokens)
            query_input_tokens = torch.LongTensor([query_tokenizer.convert_tokens_to_ids(query_tokens)]).to(args.device)
            query_input_types = torch.LongTensor([[0] * len(query_tokens)]).to(args.device)
            query_input_masks = torch.LongTensor([[1] * len(query_tokens)]).to(args.device)
            query_cls = query_model(query_input_tokens, query_input_types, query_input_masks).cpu()

            # compute similarity score
            retrieval_score = nn.functional.cosine_similarity(query_cls, candidate_matrix, dim=1)
            _, indices = torch.topk(retrieval_score, args.top_k)

            for i in range(args.top_k):
                if IDX2BLOCK[indices[i].item()] in answer_segment:
                    num_succ += 1
                    break
            num_fin_questions += 1
            sys.stdout.write('finished {}/{}; HITS@{} = {} \r'.format(num_fin_questions, len(data), args.top_k,
                                                                      num_succ / num_fin_questions))

        print('finished {}/{}; HITS@{} = {} \r'.format(num_fin_questions, len(data), args.top_k,
                                                       num_succ / num_fin_questions))
    elif args.eval_option == 'both':
        with open('preprocessed_data/dev_fused_blocks.json', 'r') as f:
            fused_blocks = json.load(f)

        for trace_question in data:
            # compute vector for the question
            query = trace_question['question']
            query_tokens = '[CLS] ' + query + ' [SEP]'
            query_tokens = query_tokenizer.tokenize(query_tokens)
            query_input_tokens = torch.LongTensor([query_tokenizer.convert_tokens_to_ids(query_tokens)]).to(args.device)
            query_input_types = torch.LongTensor([[0] * len(query_tokens)]).to(args.device)
            query_input_masks = torch.LongTensor([[1] * len(query_tokens)]).to(args.device)
            query_cls = query_model(query_input_tokens, query_input_types, query_input_masks).cpu()

            # compute similarity score and retrieve top-k blocks
            retrieval_score = nn.functional.cosine_similarity(query_cls, candidate_matrix, dim=1)
            scores, indices = torch.topk(retrieval_score, args.top_k)
            retrieval_blocks = [IDX2BLOCK[indice.item()] for indice in indices]

            # load the reader model
            reader_tokenizer = BertTokenizer.from_pretrained(
                args.model_name_or_path,
                do_lower_case=True,
                cache_dir=args.cache_dir
            )
            reader_tokenizer.add_tokens(["[TAB]", "[TITLE]", "[ROW]", "[MAX]", "[MIN]", "[EAR]", "[LAT]"])
            reader_model = BertForQuestionAnswering.from_pretrained(
                args.model_name_or_path,
                config=query_config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            reader_model.resize_token_embeddings(len(reader_tokenizer))
            reader_model.load_state_dict(torch.load(os.path.join(args.load_reader_model_path, 'pytorch_model.bin')))
            if args.n_gpu > 1:
                reader_model = nn.DataParallel(reader_model)
            reader_model.to(args.device)
            reader_model.eval()

            for cur_block in retrieval_blocks:
                print(cur_block)
                block_len_limit = args.max_block_len - len(query_tokens)
                reader_input_tokens = query_tokens + fused_blocks[cur_block][0][:block_len_limit]
                reader_input_types = [1] * (len(query_tokens) - 1) + [0] + fused_blocks[cur_block][1][:block_len_limit]
                reader_input_masks = [1] * len(query_tokens) + fused_blocks[cur_block][2][:block_len_limit]

                reader_input_tokens = torch.LongTensor([reader_tokenizer.convert_tokens_to_ids(reader_input_tokens)]).to(args.device)
                reader_input_types = torch.LongTensor([reader_input_types]).to(args.device)
                reader_input_masks = torch.LongTensor([reader_input_masks]).to(args.device)

                reader_inputs = {
                    "input_ids": reader_input_tokens,
                    "attention_mask": reader_input_types,
                    "token_type_ids": reader_input_masks,
                }

                reader_outputs = reader_model(**reader_inputs)
                print(reader_outputs[0].size())

                break
            break