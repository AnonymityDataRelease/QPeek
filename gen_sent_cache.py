import sys
import pickle
from tqdm import tqdm
from nltk import tokenize
from sentence_transformers import SentenceTransformer, util
import os
import json

def read_pk_file(filename):
    data = pickle.load(open(filename,'rb'))
    return data
def write_pk_to_file(data,filename):
    pickle.dump(data,open(filename,'wb'))
def get_file_contents(filename, encoding='utf-8'):
    filename=filename.encode('utf-8')
    with open(filename, encoding=encoding) as f:
        content = f.read()
    return content
def read_json_file(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)
def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=2, sort_keys=True, ensure_ascii=False)

bi_model_name = 'all-mpnet-base-v2'
bi_embedder = SentenceTransformer(bi_model_name)

import random

for data_file in ['TRECDL_2019','TRECDL_2020','TRECDL_2021','Robust04_descript','Robust04_title']:
    orig_data=read_json_file("Data/{}.json".format(data_file))
    #----statistics information--------------
    all_qids = []
    for qid in tqdm(orig_data.keys()):
        all_qids.append(qid)
    random.shuffle(all_qids)
    for qid in tqdm(all_qids):
        #-----------------------original data--------------------------------
        # dict_keys(['gold_answer', 'query', 'top100context', 'top100docs'])
        top100doc_ids = orig_data[qid]['top100docs']
        top100doc_context = orig_data[qid]['top100context']
        query = orig_data[qid]['query']
        per_qid_scores = {}
        gold_answer=orig_data[qid]['gold_answer']
        
        sent_cache_path = 'tmp/sent/{}-{}-{}.pkl'.format(data_file,qid,bi_model_name.replace('/', '_'))

        #----load pre computed sentence embeddings (by bi-encoder)--------
        if not os.path.exists(sent_cache_path):
            query_embedding = bi_embedder.encode(query, convert_to_tensor=True)
            doc100_sents={}
            for docid in top100doc_ids:
                doc_context = top100doc_context[docid].split("\t")[-1]
                sents = tokenize.sent_tokenize(doc_context)
                sent_embeddings = bi_embedder.encode(sents, batch_size=128,show_progress_bar=True, convert_to_tensor=True)
                doc100_sents[docid]={'sentences': sents, 'sent_embeddings': sent_embeddings,'query_embedding':query_embedding}
            with open(sent_cache_path, "wb") as fOut:
                pickle.dump(doc100_sents, fOut)
