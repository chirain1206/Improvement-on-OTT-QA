import json
from genre.trie import Trie
from genre.hf_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_hf as get_entity_spans
from tqdm import trange, tqdm
import argparse
import torch

def find_urls(search_sentence):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default='title', help='title or url')
    args = parser.parse_args()
    args.device = torch.device("cuda:0")

    model = GENRE.from_pretrained("models/hf_e2e_entity_linking_wiki_abs")
    model.to(args.device)
    model.eval()

    if args.option == 'title':
        with open('../Improvement-on-OTT-QA/released_data/dev.traced.json', 'r') as f:
            data = json.load(f)
        process_data = []
        process_size = 200
        for i in trange(process_size):
            traced_question = data[i]
            question = traced_question['question']
            prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(model, [question])

            process_question = model.sample(question, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)[0][0]['text']
            traced_question['question'] = process_question
            process_data.append(traced_question)

        with open('../Improvement-on-OTT-QA/released_data/dev.traced_GENRE.json', 'w') as f:
            json.dump(process_data, f, indent=2)
    elif args.option == 'url':
        with open('../Improvement-on-OTT-QA/data/all_passages.json', 'r') as f:
            passages = json.load(f)
        potential_url = list(passages.keys())
        cand_trie = Trie([model.encode(" }} [ {} ]".format(e))[1:].tolist() for e in potential_url])
        segment_url_dict = {}

        with open('../Improvement-on-OTT-QA/preprocessed_data/dev_table_segments.json', 'r') as f:
            dev_segments = json.load(f)
        table_names = dev_segments.keys()

        for name in tqdm(table_names):
            with open(f'../Improvement-on-OTT-QA/data/traindev_tables_tok/{name}.json', 'r') as f:
                cur_table = json.load(f)
            title = cur_table['title']
            sec_title = cur_table['section_title']
            for row_index, row in enumerate(cur_table['data']):
                segment_repr = f"In {title}, {sec_title}, "
                for col_index, header in enumerate(cur_table['header']):
                    if col_index < len(row) - 1:
                        segment_repr = segment_repr + header[0] + ' is ' + row[col_index][0] + ', '
                    else:
                        segment_repr = segment_repr + header[0] + ' is ' + row[col_index][0] + '.'
                prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(model, segment_repr, candidates_trie=cand_trie)
                generated = get_entity_spans(model, segment_repr, candidates_trie=cand_trie)
                generated2 = model.sample(segment_repr, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)[0][0]['text']
                print(generated2)
                print(generated)
                # segment_url_dict[name + f'_{row_index}'] = find_urls(generated)
                break
            break

        # with open(f'../Improvement-on-OTT-QA/link_generator/dev_url_GENRE.json', 'w') as f:
        #     json.dump(segment_url_dict, f, indent=2)