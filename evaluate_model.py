import argparse
import logging
import json
import os
import sys
import random
from train_retriever import VectorizeModel
from transformers import (BertConfig, BertTokenizer, BertModel, BertForQuestionAnswering)
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
import torch
from fuzzywuzzy import fuzz

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
    help="Path to the reader model that have been trained"
)
parser.add_argument(
    "--load_gpt_model_path",
    default="",
    type=str,
    help="Path to the GPT-2 model that have been trained"
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
parser.add_argument('--retain_passage', action="store_true", default=False, help="Whether or not to retain passages following the improvement strategy")
parser.add_argument('--predict_title', action="store_true", default=False, help="Whether or not to predict title following the improvement strategy")
parser.add_argument('--max_block_len', type=int, default=512)
args = parser.parse_args()
device = torch.device("cuda:0")
args.n_gpu = torch.cuda.device_count()
args.device = device

def sample_sequence(model, tokenizer, length, context, sub_args, temperature=1):
    generated = torch.LongTensor([tokenizer.encode(context + '[START]', add_special_tokens=False)]).to(args.device)
    predict_index = generated.size()[1]
    batch_size = generated.shape[0]

    finished_sentence = [False for _ in range(batch_size)]
    with torch.no_grad():
        for _ in range(length):
            outputs = model(generated, *sub_args)
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            else:
                next_token_logits = outputs[:, -1, :] / (temperature if temperature > 0 else 1.)

            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if all(finished_sentence):
                break

    prediction = generated[:, predict_index:].cpu().data.numpy()
    text = tokenizer.decode(prediction[0], clean_up_tokenization_spaces=True)
    decoded = []
    for _ in text[:text.find('[EOS]')].split(' # '):
        name = _.replace('#', '').strip()
        if len(name) > 1 and name not in decoded:
            decoded.append(name)

    return '(' + ','.join(decoded) + ')'

if __name__ == '__main__':
    with open('released_data/dev.traced.json', 'r') as f:
        data = json.load(f)
    random.shuffle(data)
    data = data[:args.eval_size]

    # load the matrix of candidates
    if args.retain_passage:
        candidate_info = torch.load('./preprocessed_data/dev_candidates_retained.pth')
        IDX2BLOCK = candidate_info['IDX2BLOCK']
        BLOCK2IDX = candidate_info['BLOCK2IDX']
        candidate_matrix = candidate_info['candidate_matrix']
    else:
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

    if args.predict_title:
        gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        gpt_tokenizer.add_tokens(['[SEP]', '[EOS]', '[START]', '[ENT]'])
        gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
        gpt_model.resize_token_embeddings(len(gpt_tokenizer))
        gpt_model.load_state_dict(torch.load(args.load_gpt_model_path))
        if args.n_gpu > 1:
            gpt_model = nn.DataParallel(gpt_model)
        gpt_model.to(args.device)
        gpt_model.eval()

    if args.eval_option == 'retriever':
        num_succ = 0
        num_fin_questions = 0
        for trace_question in data:
            if args.predict_title:
                input_title = 'In ' ' [SEP] ' + ' [SEP] ' + ' [ENT] ' + trace_question['question'] + ' [ENT] '
                prediction = sample_sequence(gpt_model, gpt_tokenizer, 16, input_title, [], temperature=0)
                query = trace_question['question'] + prediction
                print(trace_question['question'])
                print(prediction)
            else:
                query = trace_question['question']
            answer_row = set()
            for node in trace_question['answer-node']:
                answer_row.add(node[1][0])
            answer_segment = set()
            for row in answer_row:
                # answer_segment.add(trace_question['table_id'] + f'_{row}')
                answer_segment.add(trace_question['table_id'])

            # compute vector for the question
            query_tokens = '[CLS] ' + query + ' [SEP]'
            query_tokens = query_tokenizer.tokenize(query_tokens)
            query_input_tokens = torch.LongTensor([query_tokenizer.convert_tokens_to_ids(query_tokens)]).to(args.device)
            query_input_types = torch.LongTensor([[0] * len(query_tokens)]).to(args.device)
            query_input_masks = torch.LongTensor([[1] * len(query_tokens)]).to(args.device)
            query_cls = query_model(query_input_tokens, query_input_types, query_input_masks).cpu()

            # compute similarity score
            retrieval_score = nn.functional.cosine_similarity(query_cls, candidate_matrix, dim=1)
            _, indices = torch.topk(retrieval_score, args.top_k)

            break

            for i in range(args.top_k):
                # if IDX2BLOCK[indices[i].item()] in answer_segment:
                if IDX2BLOCK[indices[i].item()][:IDX2BLOCK[indices[i].item()].rfind('_')] in answer_segment:
                    num_succ += 1
                    break
            num_fin_questions += 1
            sys.stdout.write('finished {}/{}; HITS@{} = {:.2f}% \r'.format(num_fin_questions, len(data), args.top_k,
                                                                      100 * (num_succ / num_fin_questions)))

        print('finished {}/{}; HITS@{} = {:.2f}% \r'.format(num_fin_questions, len(data), args.top_k,
                                                       100 * (num_succ / num_fin_questions)))
    elif args.eval_option == 'both':
        if args.retain_passage:
            with open('preprocessed_data/dev_fused_blocks_retained.json', 'r') as f:
                fused_blocks = json.load(f)
        else:
            with open('preprocessed_data/dev_fused_blocks.json', 'r') as f:
                fused_blocks = json.load(f)

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

        num_succ = 0
        num_fin_questions = 0
        for trace_question in data:
            if args.predict_title:
                input_title = 'In ' ' [SEP] ' + ' [SEP] ' + ' [ENT] ' + trace_question['question'] + ' [ENT] '
                prediction = sample_sequence(gpt_model, gpt_tokenizer, 16, input_title, [], temperature=0)
                query = trace_question['question'] + prediction
            else:
                query = trace_question['question']

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
            candidate_answer = []
            candidate_answer_scores = []

            for cur_block in retrieval_blocks:
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
                start_probs = nn.functional.softmax(reader_outputs[0], dim=1)
                start_index = torch.argmax(start_probs, dim=1).item()
                start_score = start_probs[0][start_index].item()
                end_probs = nn.functional.softmax(reader_outputs[1], dim=1)[:, start_index:]
                # end_probs = reader_outputs[1][:, start_index:]
                end_index = torch.argmax(end_probs, dim=1).item()
                end_score = end_probs[0][end_index].item()
                end_index = end_index + start_index

                # extract answer span
                answer_span_prob = start_score * end_score
                answer_span = reader_tokenizer.decode(reader_input_tokens[0][start_index:end_index+1])
                candidate_answer.append(answer_span)
                candidate_answer_scores.append(answer_span_prob)

            candidate_answer_scores = torch.Tensor(candidate_answer_scores)
            sum_answer_scores = torch.mul(scores, candidate_answer_scores)
            output_answer = candidate_answer[torch.argmax(sum_answer_scores).item()]

            # if fuzz.partial_ratio(output_answer.lower(), trace_question['answer-text'].lower()) > 80:
            if output_answer.lower() == trace_question['answer-text'].lower():
                num_succ += 1
            num_fin_questions += 1
            sys.stdout.write('finished {}/{}; EM score = {:.2f}% \r'.format(num_fin_questions, len(data), 100 * (num_succ / num_fin_questions)))

        print('finished {}/{}; EM score = {:.2f}% \r'.format(num_fin_questions, len(data), 100 * (num_succ / num_fin_questions)))