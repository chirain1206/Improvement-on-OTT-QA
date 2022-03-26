import argparse
import logging
from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import torch.optim as optim
from tqdm import trange, tqdm
import math
from datetime import datetime
from utils import whitelist, is_year
import sys
import copy
from link_prediction import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt2', type=str)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--dataset', default=None, type=str, help="Whether to use dataset")
    parser.add_argument('--load_from', default=None, type=str, help="Whether to use dataset")
    parser.add_argument('--batch_size', default=128, type=int, help="Whether to use dataset")
    parser.add_argument('--every', default=50, type=int, help="Whether to use dataset")
    parser.add_argument('--max_source_len', default=32, type=int, help="Whether to use dataset")
    parser.add_argument('--max_target_len', default=16, type=int, help="Whether to use dataset")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_all', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--learning_rate', default=5e-6, type=float, help="whether to train or test the model")
    parser.add_argument('--shard', default=None, type=str, help="whether to train or test the model")

    args = parser.parse_args()

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.add_tokens(['[SEP]', '[EOS]', '[START]', '[ENT]'])
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    dataset = LinkGenearationDataset(args.dataset, 'all', tokenizer, args.max_source_len, args.max_target_len, remaining=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    print("Dataset Size = {}. Loader Size = {}".format(len(dataset), len(dataloader)))

    model.load_state_dict(torch.load(args.load_from))
    model = nn.DataParallel(model)
    model.to(args.device)
    model.eval()
    print("Loaded model from {}".format(args.load_from))

    mapping = {}
    for indexed_batch in tqdm(dev_dataloader, desc="Decoding"):
        batch = tuple(t.to(args.device) for t in indexed_batch[2:])
        row_ids = indexed_batch[0]
        links = indexed_batch[1]

        prefix, trg_inp, trg_out, mask = batch
        prefix = torch.cat([prefix, trg_inp[:, :1]], -1)
        samples = sample_sequence(model, 16, prefix, [], 1, temperature=0)
        samples = samples[:, prefix.shape[1]:]
        samples = samples.cpu().data.numpy()
        for row_id, link, s in zip(row_ids, links, samples):
            text = tokenizer.decode(s, clean_up_tokenization_spaces=True)

            decoded = []
            for _ in text[:text.find('[EOS]')].split(' # '):
                name = _.replace('#', '').strip()
                if len(name) > 1 and name not in decoded:
                    decoded.append(name)
            mapping[row_id] = mapping.get(row_id, []) + decoded

            continue

    for k, v in dataset.mapping.items():
        if k not in mapping:
            mapping[k] = v
        else:
            mapping[k].extend(v)

    f = open('link_generator/row_passage_query.json-0000{}-0000{}'.format(8, 8), 'w')
    for k, v in mapping.items():
        json_str = json.dumps((k, v))
        f.write(json_str + '\n')
    f.close()