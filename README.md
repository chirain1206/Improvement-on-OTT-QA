# Improvement on Open Table-and-Text Question Answering (OTT-QA)

This respository contains the OTT-QA dataset used in [Open Question Answering over Tables and Text
](https://arxiv.org/abs/2010.10439) published in [ICLR2021](https://openreview.net/group?id=ICLR.cc/2021/Conference) and the implementation code for the fusion retriever and  the single-block reader. Based on that, several improving strategies are adopted.

![overview](./figures/demo.png)

## Repo Structure
- released_data: this folder contains the question/answer pairs for training, dev and test data.
- data/all_plain_tables.json: this file contains the 400K+ table candidates for the dev/test set.
- data/all_passages.json: this file contains the 5M+ open-domain passage candidates for the dev/test set.
- data/traindev_tables_tok: this folder contains the train/dev tables.
- data/traindev_request_tok: this folder cotains the linked passages for train/dev in-domain tables
- table_crawling/: the folder contains the table extraction steps from Wikipedia.
- retriever/: the folder contains the script to build sparse retriever index.

## Requirements
- [Transformers 2.2.1](https://github.com/huggingface/transformers)
- [Pytorch 1.4](https://pytorch.org/)
- [scipy](https://www.scipy.org/)

We suggest using virtual environment to install these dependencies.
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install transformers
pip install pexpect
```

## Step1: Preliminary Step
### Step1-1: Download the necessary files 
```
cd data/
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json
cd ../
```
This command will download the crawled tables and linked passages from Wikiepdia in a cleaned format.
### Step1-2: Build index for data preprocessing
```
cd retriever/
python build_tfidf.py --build_option text_title --out_dir text_title_bm25 --option bm25
```
This script will generate index files under retriever/ folder, which are used in the following experiments

## Step2: Data Preprocess
### Step2-1: Split training tables into table segments
```
python split_tables.py --split train
```
This command will split tables in training set into tables segments, which is the basic retrieval unit for table in subsequent steps.

### Step2-2: Fuse table segments with their linked passages
```
python find_absent_segments.py
```
This command finds table segments in training set that are not used by the GPT-2 model to generate link passages.

```
python predict_absent_rows.py --do_all --load_from link_generator/model-ep9.pt --dataset data/all_plain_tables.json --batch_size 256
```
This command used the trained GPT-2 model to generate querys for remaining table segments.

```
python preprocess_generated_query.py
```
This script simply combines generated querys from GPT-2 model into single file, which is originally stored in different files.

```
python convert_query_to_url.py --split train --model retriever/text_title_bm25/index-bm25-ngram\=2-hash\=16777216-tokenizer\=simple.npz
```
This script finds closest passages for generated querys in training set using BM25 criterion.

```
python fuse_segment_passage.py --split train
```
This command fuses table segments in training set with their linked passages into fused blocks.

### Step2-3: Generate training data for the retriever
```
python ICT_preprocess.py
```
It creates ICT training data for dual-encoders.

```
python fine_tune_preprocess.py
```
It creates fine-tune training data for dual-encoders.

### Step2-4: Generate training data for the reader
```
python reader_preprocess.py
```
It creates fine-tune data for the single-block reader.

## Step3: Training of models
### Step3-1: Train the fused block retriever
```
python train_retriever.py --option ICT --do_lower_case --train_file retriever/ICT_pretrain_data.json --batch_size 512 --learning_rate 1e-5 --train_steps 10000
python train_retriever.py --option fine_tune --do_lower_case --train_file retriever/fine_tune_pretrain_data.json --batch_size 512 --load_model_path retriever/ICT/2022_03_28_15_02_36/
```
First command uses ICT to pretrain the retriever model, then the second command fine-tunes the model on OTT-QA. Both encoders are based on BERT-base-uncased model from HugginFace implementation.

### Step3-2: Train the single-block reader
```
python train_reader.py --do_lower_case --train_file reader/fine_tune_data.json --batch_size 32 --learning_rate 1e-5 --num_train_epoches 10
```
This command fine-tunes the single-block reader model.

## Step4: Evaluation
### Step4-1: Data preprocess
```
python split_tables.py --split dev
python convert_query_to_url.py --split dev --model retriever/text_title_bm25/index-bm25-ngram\=2-hash\=16777216-tokenizer\=simple.npz
python fuse_segment_passage.py --split dev
```
Above commands create fused blocks for dev set.

### Step4-2: Encode candidates in the test set in vector form
```
python encode_candidates.py --load_model_path retriever/fine_tune/2022_03_30_04_07_24/block_model/checkpoint-epoch4 --candidates_file preprocessed_data/dev_fused_blocks.json
```
This commands utilizes the trained block model to encode each candidate fused block in the test set.

### Step4-3: Evaludate trained models
```
python evaluate_model.py --load_model_path retriever/fine_tune/2022_03_30_04_07_24/query_model/checkpoint-epoch4 --eval_option retriever
python evaluate_model.py --load_model_path retriever/fine_tune/2022_03_30_04_07_24/query_model/checkpoint-epoch4 --eval_option both --load_reader_model_path reader/2022_04_01_05_50_38/checkpoint-epoch3 --eval_size 200
```
First script evaluates the performance of the retriever model independetly and the second script evaluates the performance of the retriever and reader model jointly.

## Step5: Improvements
### Improvement Strategy 1: Retain remaining passages in the fusion procedure
In the fusion procedure, each table segment is linked with multiple passages. When the size of fused block reaches token limit, remaining table segments will be truncated. This strategy keeps remaining passages and links them with original table segment again to form other fused blocks, which could be done by following commands.
```
python fuse_segment_passage.py --split dev --retain_passage
python encode_candidates.py --load_model_path retriever/fine_tune/2022_03_30_04_07_24/block_model/checkpoint-epoch4 --candidates_file preprocessed_data/dev_fused_blocks_retained.json --retain_passage
```
To evaluate its effectiveness, add option ```--retain_passage``` in evaluation process.

### Improvement Strategy 2: Use GPT-2 model to augment the query
Use GPT-2 model to pre-augment the query in order to relieve the low lexical overlap issue in retriever. Add option ```--load_gpt_model_path link_generator/model-ep9.pt --predict_title``` in the evaluation process to enable it.

### Improvement Strategy 3: Use GENRE model to replace GPT-2 model in fusion procedure
Working on

## GPT-2 Link Prediction in Table
Below script predicts the linked passages from the given table segment based on the context using GPT-2 model. To train the model, please use the following command.
```
python link_prediction.py --dataset data/traindev_tables.json --do_train --batch_size 512
```
To generate links, run
```
python link_prediction.py --do_all --load_from link_generator/model-ep9.pt --dataset data/all_plain_tables.json --batch_size 256 --shard [current_iteration]@[total_iteration_number]
```
This command will generate all the link mapping in the link_generator/ folder.
```
python check_gpt_performance.py --evaluation_size 10000
```
This command provides a simple evaluation to the accuracy of prediction.

## Reference
If you find this project useful, please cite it using the following format

```
  @article{chen2021ottqa,
  title={Open Question Answering over Tables and Text},
  author={Wenhu Chen, Ming-wei Chang, Eva Schlinger, William Wang, William Cohen},
  journal={Proceedings of ICLR 2021},
  year={2021}
}
```
