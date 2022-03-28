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
from ICT_preprocess import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--max_query_len', type=int, default=64)
args = parser.parse_args()