# Open Table-and-Text Question Answering (OTT-QA)

This respository contains the OTT-QA dataset used in [Open Question Answering over Tables and Text
](https://arxiv.org/abs/2010.10439) published in [ICLR2021](https://openreview.net/group?id=ICLR.cc/2021/Conference) and the baseline code for the dataset [OTT-QA](https://ott-qa.github.io/). This dataset contains open questions which require retrieving tables and text from the web to answer. This dataset is re-annotated from the previous HybridQA dataset. The dataset is collected by UCSB NLP group and issued under MIT license. You can browse the examples through our [explorer](https://ott-qa.github.io/explore.html).

![overview](./figures/demo.png)

What's new compared to [HybridQA](http://hybridqa.github.io/):
- The questions are de-contextualized to be standalone without relying on the given context to understand.
- We add new dev/test set questions the newly crawled tables, which removes the potential bias in table retrieval.
- The groundtruth table and passage are not given to the model, it needs to retrieve from 400K+ candidates of tables and 5M candidates of passages to find the evidence.
- The tables in OTT-QA do not have groundtruth hyperlinks, which simulates a more general scenario outside Wikipedia.

## Results
Table Retrieval: We use page title + page section title + table schema as the representation of a table for retrieval
|     Split     |     HITS@1    |     HITS@5     |     HITS@10       |   HITS@20         | 
|---------------|---------------|----------------|-------------------|-------------------|
|Dev            | 41.0%         | 61.8%          | 68.5%              | 73.7%             |

QA Results: We use the retrieved table + retrieved text as the evidence to run HYBRIDER model (See https://arxiv.org/pdf/2004.07347.pdf for details), the results are shown as:
|     Model     |     Dev-EM        |     Dev-F1 |
|---------------|---------------|----------------|
| BERT-based-uncased |  8.7     |     10.9       |
| [BERT-large-uncased](https://drive.google.com/file/d/1a3I2HaOIP_9wES53E5kjbb2ST5IbgrVQ/view?usp=sharing) | 10.9      | 13.1         |

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

## Additional Information
If you want to know more about the crawling procedure, please refer to [crawling](https://github.com/wenhuchen/OpenHybridQA/tree/master/table_crawling) for details.

If you want to know more about the retrieval procedure, please refer to [retriever](https://github.com/wenhuchen/OpenDomainHybridQA/tree/master/retriever) for details.

Or you can skip these two steps to directly download the needed files from AWS in Step1.

## Step1: Preliminary Step
## Step1-1: Download the necessary files 
```
cd data/
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json
cd ../
```
This command will download the crawled tables and linked passages from Wikiepdia in a cleaned format.
## Step1-2: Build inedx for retriever
```
cd retriever/
python build_tfidf.py --build_option text_title --out_dir text_title_bm25 --option bm25
python build_tfidf.py --build_option title_sectitle_schema --out_dir title_sectitle_schema
```
This script will generate index files under retriever/ folder, which are used in the following experiments

## Step1-3: Reproducing the retrieval results
```
python evaluate_retriever.py --split dev --model retriever/title_sectitle_schema/index-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz  --format question_table
```
This script will produce the table retrieval results in terms of HITS@1,5,10,20,50.

## Step2: Training
### Step2-0: If you want to download the model from [Google Drive](https://drive.google.com/file/d/1a3I2HaOIP_9wES53E5kjbb2ST5IbgrVQ/view?usp=sharing), you can skip the following training procedure.
```
unzip models.zip
```
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
It creates pseudo-training data for dense retrieval.

### Step2-4: Train the fused block retriever
```
python train_retriever.py --option ICT --do_lower_case --train_file retriever/ICT_pretrain_data.json --batch_size 512
```
First command uses ICT to pretrain the retriever model, then the second command fine-tunes the model on OTT-QA. Both encoders are based on BERT-base-uncased model from HugginFace implementation.

## Step3: Evaluation
### Step3-1: Reconstruct Hyperlinked Table using built text title index
```
python evaluate_retriever.py --format table_construction --model retriever/text_title_bm25/index-bm25-ngram\=2-hash\=16777216-tokenizer\=simple.npz
python retrieve_and_preprocess.py --split dev_retrieve --model retriever/title_sectitle_schema/index-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz
python retrieve_and_preprocess.py --split test_retrieve --model retriever/title_sectitle_schema/index-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz
```
This step can potentially take a long time since it matches each cell in the 400K tables against the whole passage title pool.

### Step3-2: Evaluate with the trained model
```
python train_stage12.py --stage1_model stage1/[YOUR-MODEL-FOLDER] --stage2_model stage2/[YOUR-MODEL-FOLDER] --do_lower_case --predict_file preprocessed_data/dev_inputs.json --do_eval --option stage12 --model_name_or_path bert-large-uncased --table_path data/all_constructed_tables.json --request_path data/all_passages.json
python train_stage3.py --model_name_or_path stage3/[YOUR-MODEL-FOLDER] --do_stage3   --do_lower_case  --predict_file predictions.intermediate.json --per_gpu_train_batch_size 12  --max_seq_length 384   --doc_stride 128 --threads 8 --request_path data/all_passages.json
```
Once you have generated the predictions.json file, you can use the following command to see the results.
```
python evaluate_script.py predictions.json released_data/dev_reference.json
```
To replicate my results, please see the generated predictions.dev.json by my model.
```
python evaluate_script.py predictions.dev.json released_data/dev_reference.json
```

## CodaLab Evaluation
To obtain the score on the test set (released_data/test.blind.json), you need to participate the CodaLab challenge in [OTT-QA Competition](https://competitions.codalab.org/competitions/27324). Please submit your results to obtain your testing score. The submitted file should first be named "test_answers.json" and then zipped. The required format of the submission file is described as follows:
```
[
  {
    "question_id": xxxxx,
    "pred": XXX
  },
  {
    "question_id": xxxxx,
    "pred": XXX
  }
]
```
The reported scores are EM and F1.

## Link Prediction in Table
We also provide the script to predict the links from the given table based on the context using GPT-2 model. To train the model, please use the following command.
```
python link_prediction.py --dataset data/traindev_tables.json --do_train --batch_size 512
```
To generate links, please run
```
python link_prediction.py --do_all --load_from link_generator/model-ep9.pt --dataset data/all_plain_tables.json --batch_size 256 --shard [current_iteration]@[total_iteration_number]
```
This command will generate all the link mapping in the link_generator/ folder.

## Visualization
If you want to browse the tables, please go to [this website](https://wenhuchen.github.io/opendomaintables.github.io/) and type in your table_id like 'Serbia_at_the_European_Athletics_Championships_2', then you will see all the information related to the given table.

## Recent Papers

**Model**                                     |  **Organization**  |**Reference**                                                             | **Dev-EM** | **Dev-F1** | **Test-EM** | **Test-F1** | 
----------|---------------------------|-----------------------------------|---------------------------------------------------------------------------|---------|----------|------------------|
Fusion+Cross-Reader         | Google      | [Chen et al. (2021)](https://arxiv.org/abs/2010.10439)                    |  28.1 | 32.5  | 27.2  | 31.5 |
Dual Reader-Parser | Amazon | [Alexander et al. (2021)](https://assets.amazon.science/09/2b/7acf41f24c998cd3c2361681e9db/dual-reader-parser-on-hybrid-textual-and-tabular-evidence-for-open-domain-question-answering.pdf)                                              |   15.8   |    -  |  -       | -        |
BM25-HYBRIDER   | UCSB      | [Chen et al. (2021)](https://arxiv.org/abs/2010.10439)                    |    10.3     | 13.0  |  9.7           | 12.8  |

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
