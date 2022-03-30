import json
import argparse
import logging
import os
import sys
import random
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer,
                        BertModel, get_linear_schedule_with_warmup)
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
import pdb
import copy
from tqdm import tqdm, trange
from torch.autograd import Variable
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from datetime import datetime
import os
from torch.utils.data import DataLoader, Dataset, RandomSampler
import math
from train_stage12 import PretrainedModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def pad_collate_fn(data):
    query_input_tokens, query_input_types, query_input_masks, block_input_tokens, block_input_types, block_input_masks = zip(
        *data)
    max_query_len = max([len(_) for _ in query_input_tokens])
    max_block_len = max([len(_) for _ in block_input_tokens])

    # convert them into list and do manipulation
    query_input_tokens = list(query_input_tokens)
    query_input_types = list(query_input_types)
    query_input_masks = list(query_input_masks)
    block_input_tokens = list(block_input_tokens)
    block_input_types = list(block_input_types)
    block_input_masks = list(block_input_masks)

    for i in range(len(query_input_tokens)):
        query_input_tokens[i] += [0] * (max_query_len - len(query_input_tokens[i]))
        query_input_types[i] += [0] * (max_query_len - len(query_input_types[i]))
        query_input_masks[i] += [0] * (max_query_len - len(query_input_masks[i]))
    for i in range(len(block_input_tokens)):
        block_input_tokens[i] += [0] * (max_block_len - len(block_input_tokens[i]))
        block_input_types[i] += [0] * (max_block_len - len(block_input_types[i]))
        block_input_masks[i] += [0] * (max_block_len - len(block_input_masks[i]))

    query_input_tokens = torch.LongTensor(query_input_tokens)
    query_input_types = torch.LongTensor(query_input_types)
    query_input_masks = torch.LongTensor(query_input_masks)
    block_input_tokens = torch.LongTensor(block_input_tokens)
    block_input_types = torch.LongTensor(block_input_types)
    block_input_masks = torch.LongTensor(block_input_masks)

    return (query_input_tokens, query_input_types, query_input_masks, block_input_tokens, block_input_types, block_input_masks)

class VectorizeModel(PretrainedModel):
    def __init__(self, model_class, model_name_or_path, config, tokenizer_len, cache_dir, orig_dim, proj_dim,
                 for_block=False):
        super(VectorizeModel, self).__init__()

        self.base = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            cache_dir=cache_dir if cache_dir else None,
        )
        if for_block:
            self.base.resize_token_embeddings(tokenizer_len)
        self.projection = nn.Linear(orig_dim, proj_dim)

    def forward(self, input_tokens, input_types, input_masks):
        inputs = {"input_ids": input_tokens, "token_type_ids": input_types, "attention_mask": input_masks}
        _, cls_representation = self.base(**inputs)
        proj_cls = self.projection(cls_representation)

        # return tensor in [batch_size, proj_dim]
        return proj_cls

class retrieverDataset(Dataset):
    def __init__(self, data, query_tokenizer, block_tokenizer, shuffle=True):
        super(retrieverDataset, self).__init__()
        self.data = data
        self.shuffle = shuffle
        self.query_tokenizer = query_tokenizer
        self.block_tokenizer = block_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.shuffle:
            item = random.choice(self.data)
        else:
            item = self.data[index]

        query_tokens = item[0][0]
        fused_block_tokens = item[1][0]
        query_input_tokens = self.query_tokenizer.convert_tokens_to_ids(query_tokens)
        query_input_types = item[0][1]
        query_input_masks = item[0][2]
        block_input_tokens = self.block_tokenizer.convert_tokens_to_ids(fused_block_tokens)
        block_input_types = item[1][1]
        block_input_masks = item[1][2]

        # return query and ground-truth fused block (they have the forms (input_ids, token_types, token_masks))
        return query_input_tokens, query_input_types, query_input_masks, block_input_tokens, block_input_types, block_input_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--option",
        default=None,
        type=str,
        required=True,
        help="ICT or fine_tune",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--cache_dir",
        default="/tmp/",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--logging_steps", type=int, default=1024, help="Log every X updates steps.")
    parser.add_argument("--train_steps", default=10000, type=int, help="Total training steps.")
    parser.add_argument("--batch_size", default=2048, type=int, help="Batch sizes for each iteration.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--output_dir_query", default="query_model/", type=str, help="Directory to save the trained query model."
    )
    parser.add_argument(
        "--output_dir_block", default="block_model/", type=str, help="Directory to save the trained block model."
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list"
    )
    parser.add_argument(
        "--load_model_path",
        default="",
        type=str,
        help="Path to the model that have been trained"
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    args = parser.parse_args()
    args.output_dir = os.path.join('retriever', args.option + '_test')
    args.batch_size = math.ceil(math.sqrt(args.batch_size))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir_query = os.path.join(args.output_dir, args.output_dir_query)
    if not os.path.exists(args.output_dir_query):
        os.makedirs(args.output_dir_query)
    args.output_dir_block = os.path.join(args.output_dir, args.output_dir_block)
    if not os.path.exists(args.output_dir_block):
        os.makedirs(args.output_dir_block)

    device = torch.device("cuda:1")
    device_ids = [1,2,3]
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    # Set seed
    set_seed(args)

    query_config = BertConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    block_config = BertConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    query_tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir
    )
    block_tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir
    )
    block_tokenizer.add_tokens(["[TAB]", "[TITLE]", "[ROW]", "[MAX]", "[MIN]", "[EAR]", "[LAT]"])
    args.orig_dim = block_config.hidden_size
    args.proj_dim = 128

    if len(args.load_model_path) == 0:
        query_model = VectorizeModel(BertModel, args.model_name_or_path, query_config, len(query_tokenizer), args.cache_dir,
                                     args.orig_dim, args.proj_dim)
        block_model = VectorizeModel(BertModel, args.model_name_or_path, block_config, len(block_tokenizer), args.cache_dir,
                                     args.orig_dim, args.proj_dim, for_block=True)
    if args.n_gpu > 1:
        query_model = nn.DataParallel(query_model, device_ids=device_ids)
        block_model = nn.DataParallel(block_model, device_ids=device_ids)
    query_model.to(args.device)
    block_model.to(args.device)

    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
    dataset = retrieverDataset(train_data, query_tokenizer, block_tokenizer, shuffle=False)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8, pin_memory=True,
                        collate_fn=pad_collate_fn, drop_last=True)
    print("Dataset Size = {}. Loader Size = {}".format(len(dataset), len(loader)))

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    query_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in query_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in query_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    block_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in block_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in block_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    query_optimizer = AdamW(query_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    block_optimizer = AdamW(block_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    args.num_train_epoches = 1
    t_total = args.num_train_epoches * (len(dataset) // args.batch_size) * args.batch_size

    query_scheduler = get_linear_schedule_with_warmup(
        query_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    block_scheduler = get_linear_schedule_with_warmup(
        block_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    query_model.train()
    query_model.zero_grad()
    query_optimizer.zero_grad()
    block_model.train()
    block_model.zero_grad()
    block_optimizer.zero_grad()

    train_iterator = trange(0, int(args.num_train_epoches), desc="Epoch")
    for epoch in train_iterator:
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            query_input_tokens, query_input_types, query_input_masks, block_input_tokens, block_input_types, block_input_masks = tuple(
                Variable(t).to(args.device) for t in batch)

            query_cls = query_model(query_input_tokens, query_input_types, query_input_masks)
            block_cls = block_model(block_input_tokens, block_input_types, block_input_masks)

            for sub_iteration in range(query_cls.size()[0]):
                cur_query = query_cls[sub_iteration,:].unsqueeze(0)
                retrieval_score = nn.functional.cosine_similarity(cur_query, block_cls, dim=1).unsqueeze(0)
                label = torch.LongTensor([sub_iteration]).to(args.device)

                # compute loss and backward population
                loss = criterion(retrieval_score, label)
                tr_loss += loss.item()
                if sub_iteration == query_cls.size()[0] - 1:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(query_model.parameters(), args.max_grad_norm)
                query_optimizer.step()
                query_scheduler.step()

                torch.nn.utils.clip_grad_norm_(block_model.parameters(), args.max_grad_norm)
                block_optimizer.step()
                block_scheduler.step()

                query_model.zero_grad()
                block_model.zero_grad()

                global_step += 1
                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("{}_query_lr".format('train'), query_scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("{}_block_lr".format('train'), block_scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar("{}_loss".format('train'), (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info('Current learning rate: %.8f' % query_scheduler.get_last_lr()[0])
                    logger.info('Current loss: %.3f' % ((tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

        # if epoch + 1 in epoch_log_step:
        # Save model checkpoint
        output_dir_query = os.path.join(args.output_dir_query, "checkpoint-epoch{}".format(epoch))
        if not os.path.exists(output_dir_query):
            os.makedirs(output_dir_query)
        query_model_to_save = query_model.module if hasattr(query_model, "module") else query_model
        query_model_to_save.save_pretrained(output_dir_query)
        query_tokenizer.save_pretrained(output_dir_query)
        torch.save(args, os.path.join(output_dir_query, "training_args.bin"))

        output_dir_block = os.path.join(args.output_dir_block, "checkpoint-epoch{}".format(epoch))
        if not os.path.exists(output_dir_block):
            os.makedirs(output_dir_block)
        block_model_to_save = block_model.module if hasattr(block_model, "module") else block_model
        block_model_to_save.save_pretrained(output_dir_block)
        block_tokenizer.save_pretrained(output_dir_block)
        torch.save(args, os.path.join(output_dir_block, "training_args.bin"))

    tb_writer.close()