import sys
from myutils import read_json_file,write_json_to_file,write_pk_to_file
import pickle
from tqdm import tqdm
from nltk import tokenize
from sentence_transformers import SentenceTransformer, util
import os
#https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_quora_elasticsearch.py
#https://www.sbert.net/docs/pretrained_models.html
# '''
# One piece from different anchor-guided sentence chunks
# '''
def slide_with_anchor(input_list,slide_size,anchor_idx):# word slice window around anchor
    output_list = []
    for i in range(len(input_list) - slide_size + 1):
        if anchor_idx in input_list[i: i + slide_size]:
            output_list.append(input_list[i: i + slide_size])
    return(output_list)
def gen_slide_win(word_list,win_size,strip_len):
    max_len = len(word_list)
    slices = []
    start = 0
    while start+win_size<max_len:
        slices.append((start,start+win_size))
        start = start+strip_len
    slices.append((start,max_len))
    
    splited_context = []
    for (s,e) in slices:
        splited_context.append(" ".join(word_list[s:e]))
    return splited_context
def fast_blk_selection(anc_around_blk,anc_sent,sents,sents_embed,query_embed,lambda_combine):
    blk_scores = {}
    anc_idx = sents.index(anc_sent)
    anc_embed = sents_embed[anc_idx]
    for blk in anc_around_blk:
        blk_sent_info = []
        for sentidx in range(len(sents)):
            persent = sents[sentidx]
            if persent in blk:
                blk_sent_info.append((sentidx,len(persent.split(" "))))
        s_pos = blk.find(sents[blk_sent_info[0][0]])
        start_sent = blk[0:s_pos].strip()
        start_sent_idx = blk_sent_info[0][0]-1
        last2_sent = sents[blk_sent_info[-1][0]]
        
        e_pos = blk.find(last2_sent)+len(last2_sent)
        end_sent = blk[e_pos:]
        end_sent_idx = blk_sent_info[-1][0]+1
        if start_sent in sents[start_sent_idx]:
            blk_sent_info.append((start_sent_idx,len(start_sent.split(" "))))
        if end_sent_idx <len(sents):
            if end_sent in sents[end_sent_idx]:
                blk_sent_info.append((end_sent_idx,len(end_sent.split(" "))))
        query_scores = util.dot_score(query_embed,sents_embed)[0].cpu().tolist()
        anc_scores = util.dot_score(anc_embed,sents_embed)[0].cpu().tolist()
        
        score = sum([(query_scores[item[0]]*lambda_combine+(1-lambda_combine)*anc_scores[item[0]])*item[1] for item in blk_sent_info])
        blk_scores[blk] = score
    sort_blk_s = sorted(blk_scores.items(),key=lambda x:x[1],reverse=True)
    fast_blk_context = sort_blk_s[0][0]
    return fast_blk_context

bi_model_name = 'all-mpnet-base-v2'
bi_embedder = SentenceTransformer(bi_model_name)

from sentence_transformers import CrossEncoder
cross_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
cross_embedder = CrossEncoder(cross_model_name)

Anchor_Num =  10
strip_len = 20
block_len = 140
lambda_fast = 1 
pesudo_doc_len = 150

output_dir = "runs"
run_file_name = "_".join([str(Anchor_Num),str(strip_len),str(block_len),str(pesudo_doc_len),str(lambda_fast)])

for data_file in ['TRECDL_2019','TRECDL_2020','TRECDL_2021']:#
    output_run = "{}/{}_{}.json".format(output_dir,data_file,run_file_name)
    orig_data=read_json_file("Data/{}.json".format(data_file))
    #----statistics information--------------
    
    output_scores = {}
    allqids = []
    for qid in tqdm(orig_data.keys()):
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
        else:
            with open(sent_cache_path, "rb") as fIn:
                doc100_sents = pickle.load(fIn)
        
        #-------------------------------------------------------------------------------------
        all_doc_scores = {}
        for docid in doc100_sents.keys():
            if "TRECDL" in data_file:
                doc_title = top100doc_context[docid].split("\t")[2]
                doc_context = top100doc_context[docid].split("\t")[-1]
            else:
                doc_title = ""
                doc_context = top100doc_context[docid]
            doc_blocks = gen_slide_win(doc_context.split(" "),block_len,strip_len)
            sents = doc100_sents[docid]['sentences']
            if len(sents) ==0:
                continue
            sents_embed = doc100_sents[docid]['sent_embeddings']
            query_embed = doc100_sents[docid]['query_embedding']
            
            # -------------Anchor Detection-----------------------
            scores_sent_dot = util.dot_score(query_embed,sents_embed)[0].cpu().tolist()
            score_sent = list(zip(sents, scores_sent_dot))
            sorted_score_sent = sorted(score_sent, key=lambda x: x[1], reverse=True)
            Anchors = [tmp[0] for tmp in sorted_score_sent[0:min(Anchor_Num,len(sents))]]

            #-------------------Generate all candidate chunks------------
            selected_blk = []
            for anc_sent in Anchors:
                anc_around_blk = []
                for blk in doc_blocks:
                    if anc_sent in blk:
                        anc_around_blk.append(blk)
                
                if len(anc_around_blk)!=0:
                    #--------Pass Selection-----------------
                    fast_blk = fast_blk_selection(anc_around_blk,anc_sent,sents,sents_embed,query_embed,lambda_fast)
                    selected_blk.append(fast_blk)
            if len(selected_blk)!=0:
                query_blk_scores = cross_embedder.predict([(query,blk) for blk in selected_blk])
                sort_selected_blk = sorted(list(zip(selected_blk,query_blk_scores)), key=lambda x: x[1], reverse=True)
                pesudo_doc = " ".join(" ".join([tmp[0] for tmp in sort_selected_blk]).split(" ")[0:pesudo_doc_len])
            else:
                pesudo_doc = ""

            all_doc_scores[docid] = cross_embedder.predict([(query,doc_title+"\t"+pesudo_doc)]).tolist()[0]
        
        output_scores[qid] = all_doc_scores
    write_json_to_file(output_scores,output_run)


