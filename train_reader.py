import json
import argparse
import logging
import os
import sys
import random
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer,
                        BertForQuestionAnswering, get_linear_schedule_with_warmup)
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
    input_tokens, input_types, input_masks, start_indices, end_indices = zip(*data)
    max_len = max([len(_) for _ in input_tokens])

    # convert them into list and do manipulation
    input_tokens = list(input_tokens)
    input_types = list(input_types)
    input_masks = list(input_masks)

    for i in range(len(input_tokens)):
        input_tokens[i] += [0] * (max_len - len(input_tokens[i]))
        input_types[i] += [0] * (max_len - len(input_types[i]))
        input_masks[i] += [0] * (max_len - len(input_masks[i]))

    input_tokens = torch.LongTensor(input_tokens)
    input_types = torch.LongTensor(input_types)
    input_masks = torch.LongTensor(input_masks)
    start_indices = torch.LongTensor(start_indices)
    end_indices = torch.LongTensor(end_indices)

    return (input_tokens, input_types, input_masks, start_indices, end_indices)

class readerDataset(Dataset):
    def __init__(self, data, tokenizer, shuffle=True):
        super(readerDataset, self).__init__()
        self.data = data
        self.shuffle = shuffle
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.shuffle:
            item = random.choice(self.data)
        else:
            item = self.data[index]

        input_tokens = self.tokenizer.convert_tokens_to_ids(item[0])
        input_types = item[1]
        input_masks = item[2]
        start_index = item[3]
        end_index = item[4]

        return input_tokens, input_types, input_masks, start_index, end_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "--load_optimizer_and_scheduler", action="store_true", help="Whether or not to load optimizer and scheduler."
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--logging_steps", type=int, default=30, help="Log every X updates steps.")
    parser.add_argument("--batch_size", default=24, type=int, help="Batch sizes for each iteration.")
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
        "--output_dir", default="reader/", type=str, help="Directory to save the trained reader model."
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
    parser.add_argument("--num_train_epoches", default=4, type=int, help="Number of epoches for training.")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:0")
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

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir
    )
    tokenizer.add_tokens(["[TAB]", "[TITLE]", "[ROW]", "[MAX]", "[MIN]", "[EAR]", "[LAT]"])
    model = BertForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.resize_token_embeddings(len(tokenizer))

    if len(args.load_model_path) > 0:
        model_path = os.path.join(args.load_model_path, 'pytorch_model.bin')
        model.load_state_dict(torch.load(model_path))
    # if args.n_gpu > 1:
    #     model = nn.DataParallel(model)
    model.to(args.device)

    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
    dataset = readerDataset(train_data, tokenizer, shuffle=False)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8, pin_memory=True,
                        collate_fn=pad_collate_fn, drop_last=True)
    print("Dataset Size = {}. Loader Size = {}".format(len(dataset), len(loader)))

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    t_total = args.num_train_epoches * len(loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Load in optimizer and scheduler states if needed
    if args.load_optimizer_and_scheduler:
        optimizer.load_state_dict(torch.load(os.path.join(args.load_model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.load_model_path, "scheduler.pt")))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    train_iterator = trange(0, int(args.num_train_epoches), desc="Epoch")
    for epoch in train_iterator:
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = tuple(Variable(t).to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            loss = outputs[0]

            # if args.n_gpu > 1:
            #     loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            tr_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            model.zero_grad()

            global_step += 1
            # Log metrics
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar("{}_lr".format('train'), scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar("{}_loss".format('train'), (tr_loss - logging_loss) / args.logging_steps, global_step)
                logger.info('Current learning rate: %.8f' % scheduler.get_last_lr()[0])
                logger.info('Current loss: %.3f' % ((tr_loss - logging_loss) / args.logging_steps))
                logging_loss = tr_loss

        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoint-epoch{}".format(epoch))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    tb_writer.close()