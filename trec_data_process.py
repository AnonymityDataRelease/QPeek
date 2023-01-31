import csv
import gzip
from tqdm import tqdm
from collections import defaultdict
import sys

import json
def get_21_document(document_id):
    (string1, string2, bundlenum, position) = document_id.split('_')
    assert string1 == 'msmarco' and string2 == 'doc'
    with open(f'MSmarcoData/msmarco_v2_doc/msmarco_doc_{bundlenum}', 'rt', encoding='utf8') as in_fh:
        in_fh.seek(int(position))
        json_string = in_fh.readline()
        document = json.loads(json_string)
        assert document['docid'] == document_id
        return document

def write_json_to_file(json_object, json_file, mode='w', encoding='utf-8'):
    with open(json_file, mode, encoding=encoding) as outfile:
        json.dump(json_object, outfile, indent=2, sort_keys=True, ensure_ascii=False)

#https://ir-datasets.com/trec-robust04.html
def getcontent(docid,docoffset, f):
    """getcontent(docid, f) will get content for a given docid (a string) from filehandle f.
    The content has four tab-separated strings: docid, url, title, body.
    """
    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), \
        f"Looking for {docid}, found {line}"
    return line.rstrip()
if __name__ == '__main__':
    
    data_type = "2021"#"2020" "2019"
    dir_path = "MSmarcoData"

    query_file = "{}/msmarco-test{}-queries.tsv".format(dir_path,data_type)
    rels_file = "{}/{}qrels-docs.txt".format(dir_path,data_type)
    top100_file = "{}/msmarco-doctest{}-top100".format(dir_path,data_type)
    doc_lookup_file = "{}/msmarco-docs-lookup.tsv.gz".format(dir_path)
    doc_file = "{}/msmarco-docs.tsv".format(dir_path)

    topicid_querystring = {}
    with open(query_file, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            topicid_querystring[topicid] = querystring_of_topicid

    # In the corpus tsv, each docid occurs at offset docoffset[docid]
    docoffset = {}
    with gzip.open(doc_lookup_file, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [docid, _, offset] in tsvreader:
            docoffset[docid] = int(offset)

    # For each topicid, the list of positive docids is topicid_answer[topicid]
    topicid_answer = {}
    with open(rels_file, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            if int(rel)>=1:
                if topicid in topicid_answer:
                    topicid_answer[topicid].append(docid)
                else:
                    topicid_answer[topicid] = [docid]

    # For each topicid, the list of all candidate docids is topicid_top100[topicid]
    topicid_top100 = {}
    with open(top100_file, 'rt', encoding='utf8') as top100f:
        for line in top100f:
            [topicid, _, unjudged_docid, rank, _, _] = line.split()
            if topicid in topicid_top100:
                topicid_top100[topicid].append(unjudged_docid)
            else:
                topicid_top100[topicid] = [unjudged_docid]
    
    ## queries with labelled documents
    total = 0
    recalled = 0
    has_gold_answer = set()
    for topicid,answer in topicid_answer.items():
        MyFlag = False
        if topicid in topicid_top100:
            total+=1
            cand = topicid_top100[topicid]
            for docid in answer:
                if docid in cand:
                    MyFlag=True
            if MyFlag:
                recalled +=1
                has_gold_answer.add(topicid)
    print(data_type,"Total: {}, Recalled: {}, {}".format(total,recalled,recalled/total))
    
    output_dataset = {}
    with open(doc_file,'rt', encoding="utf8") as f:
        for topicid in tqdm(has_gold_answer):
            query = topicid_querystring[topicid]
            top_100_docids = topicid_top100[topicid]
            gold_ans = topicid_answer[topicid]
            top_100_doc_context = {}
            for docid in top_100_docids:
                if data_type == "2021":
                    docdir =get_21_document(docid)
                    top_100_doc_context[docid] = "\t".join([docid,docdir['url'],docdir['title'],docdir['body']])
                else:
                    top_100_doc_context[docid] = getcontent(docid,docoffset,f)       
            output_dataset[topicid]={
                'query':query,
                'top100docs':top_100_docids,
                'top100context':top_100_doc_context,
                'gold_answer':gold_ans
            }
    write_json_to_file(output_dataset,"Data/TRECDL_{}.json".format(data_type))

