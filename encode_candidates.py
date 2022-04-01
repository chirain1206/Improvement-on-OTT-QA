import argparse
import logging
import json
import os
import sys
from transformers import (BertConfig, BertTokenizer, BertModel)
import torch
from torch import nn
from train_stage12 import PretrainedModel
from train_retriever import VectorizeModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_model_path",
    default="",
    type=str,
    required=True,
    help="Path to the model that have been trained"
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
    "--candidates_file",
    default=None,
    type=str,
    help="Path to the file includes candidate fused blocks",
)
parser.add_argument(
    "--do_lower_case",
    default=True,
    action="store_true",
    help="Path to the file includes candidate fused blocks",
)
args = parser.parse_args()
device = torch.device("cuda:0")
args.device = device

class candidateDataset(Dataset):
    def __init__(self, data, block_tokenizer):
        super(candidateDataset, self).__init__()
        self.data = data
        self.block_tokenizer = block_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tokens, token_type, token_mask = self.data[index]

        # add [CLS] token to front of the fused block
        tokens = ["[CLS]"] + tokens
        tokens = torch.LongTensor(block_tokenizer.convert_tokens_to_ids(tokens))
        type = torch.LongTensor([0] + token_type)
        mask = torch.LongTensor([1] + token_mask)

        return tokens, type, mask

if __name__ == '__main__':
    block_config = BertConfig.from_pretrained(
        args.model_name_or_path,
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
    block_model = VectorizeModel(BertModel, args.model_name_or_path, block_config, len(block_tokenizer), args.cache_dir,
                                 args.orig_dim, args.proj_dim, for_block=True)
    block_model_path = os.path.join(args.load_model_path, 'pytorch_model.bin')
    block_model.load_state_dict(torch.load(block_model_path))
    block_model.to(args.device)
    block_model.eval()
    candidate_matrix = None

    with open(args.candidates_file, 'r') as f:
        candidates = json.load(f)

    IDX2BLOCK = list(candidates.keys())
    data = list(candidates.values())
    BLOCK2IDX = {block_name: i for i, block_name in enumerate(IDX2BLOCK)}

    dataset = candidateDataset(data, block_tokenizer)
    loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, drop_last=False)
    tmp = []

    for batch in tqdm(loader, desc="Iteration"):
        tokens, token_type, token_mask = tuple(t.to(args.device) for t in batch)

        # torch.Size([1, 128])
        candidate_vec = block_model(tokens, token_type, token_mask)
        candidate_vec = candidate_vec.cpu()

        print(candidate_vec.size())

        # if candidate_matrix == None:
        #     candidate_matrix = candidate_vec
        # else:
        #     candidate_matrix = torch.cat((candidate_matrix, candidate_vec), 0)

    assert candidate_matrix.size()[0] == len(IDX2BLOCK)

    save_data = {"IDX2BLOCK":IDX2BLOCK, "BLOCK2IDX":BLOCK2IDX, "candidate_matrix":candidate_matrix}
    torch.save(save_data, 'preprocessed_data/dev_candidates.pth')