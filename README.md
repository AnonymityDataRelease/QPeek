## Query-Peeking Document Re-ranking

We released the used Data and codes for the proposed method QPeek in the paper entitled "Query-Peeking Long Text Modeling for Neural Document Ranking".
### Data Preparation

 1. TREC-DL collection: the dataset is available on https://github.com/microsoft/MSMARCO-Document-Ranking. We provide already pre-processed training data for the main experiment.
		
		python trec_data_process.py

 3. Robust04 collection: it is not available for download (unless you sign an agreement). We followed the settings used in the paper: [Understanding Performance of Long-Document Ranking Models through Comprehensive Evaluation and Leaderboarding, 2022](https://arxiv.org/abs/2207.01262). Leonid Boytsov, Tianyi Lin, Fangwei Gao, Yutian Zhao, Jeffrey Huang, Eric Nyberg.

 
### Our QPeek Method 
Our  codes are based on the following packages:

	conda create -n py38ir python=3.8
    conda activate py38ir
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    
	pip install -U sentence-transformers 
	pip install ir-measures 
	pip install tqdm 
	pip install nltk

We employ [Sentence Transformer](https://www.sbert.net/#) to conduct anchor sentence detection and expansion. The embeddings of sentences and queries are encoded by [Bi-Encoder](https://www.sbert.net/examples/applications/retrieve_rerank/README.html). We employ the pretrained sentence embedding model ([link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)), It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search. For more details, please find [here](https://www.sbert.net/docs/pretrained_models.html#sentence-embedding-models). 
For the passage-level ranker, we employ the [Cross-Encoder](https://www.sbert.net/examples/applications/cross-encoder/README.html) pre-trained on MSMARCO passage re-ranking task.

The code for pre-computing the sentence embeddings:
	
	python gen_sent_cache.py

The code of our QPeek method:

	python QPeek_fast.py

For the first stage run files, we use the official files for TREC-DL collection.  For the Robust04 collection, we use the first-stage runs for fields   `title`  and  `description` released in [here](https://github.com/searchivarius/long_doc_rank_model_analysis/blob/main/trec_runs_cached/robust04). 

The used qrels and our results of document ranking are available in the Data folder.

For evaluation, we use [ir measures](https://pypi.org/project/ir-measures/) to compute nDCG@10, 20, and MAP. 

For human evaluation, the used data is from the training dataset from the MSMARCO document re-ranking task. The used queries are available on Data folder.
