from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import numpy as np
from textblob import TextBlob
import pickle
import argparse
tqdm.pandas()

import dgl
from dgl.data.utils import save_graphs, load_graphs

def main(args):
    # Read data
    df_train = pd.read_csv(args.train_file, delimiter='\t')
    df_test = pd.read_csv(args.test_file, delimiter='\t')
    df_dev = pd.read_csv(args.dev_file, delimiter='\t')
    df = pd.concat([df_train, df_test, df_dev])


    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli').cuda()
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

    def score(sequence1, sequence2):
        # run through model pre-trained on MNLI
        x = tokenizer.encode(sequence1, sequence2, return_tensors='pt',
                        truncation_strategy='only_first')
        logits = nli_model(x.cuda())[0]

        # we throw away "neutral" (dim 1) and take the probability of
        # "entailment" (2) as the probability of the label being true 
        entail_contradiction_logits = logits[:,[0,2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:,1]
        return prob_label_is_true.detach().item()

    df['score'] = df.progress_apply(lambda row : score(row.claim_text, row.evidence), axis=1)

    embedding_dict = pickle.load(open(args.embedding_file,'rb'))

    # Build node features
    claim_feats, claim_labels, claim_texts = [], [], []
    claim_ori_ids = []
    claim_ids = {}
    claim_labels = []
    src_support_claims = defaultdict(set)
    src_not_support_claims = defaultdict(set)

    for _, record in tqdm(df.iterrows(), desc='Read data'):
        if record.id_left not in claim_ids:
            claim_texts.append(record.claim_text)
            claim_feats.append(embedding_dict[record.id_left])
            claim_labels.append(int(record.cred_label))
            claim_ids[record.id_left] = len(claim_ids)
            claim_ori_ids.append(record.id_left)
        claim_id = claim_ids[record.id_left]

        if record.score < 0.5:
            src_not_support_claims[record.evidence_source].add(claim_id)
        else:
            src_support_claims[record.evidence_source].add(claim_id)
    claim_features = np.array(claim_feats)
    claim_labels = np.array(claim_labels)
    claim_ori_ids = np.array(claim_ori_ids)

    for src in list(src_support_claims):
        src_support_claims[src] = list(src_support_claims[src])
    for src in list(src_not_support_claims):
        src_not_support_claims[src] = list(src_support_claims[src])

    # Build edges
    edges = defaultdict(set)

    # Mutual-sentiment-evidence edges
    for src in src_support_claims:
        claims =  src_support_claims[src]
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                edges['1-1'].add((claims[i], claims[j]))
                edges['1-1'].add((claims[j], claims[i]))
    for src in src_not_support_claims:
        claims = src_not_support_claims[src]
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                edges['0-0'].add((claims[i], claims[j]))
                edges['0-0'].add((claims[j], claims[i]))
    for src in set(src_not_support_claims).intersection(src_support_claims):
        claims_1 = src_support_claims[src]
        claims_0 = src_not_support_claims[src]
        for claim_1 in claims_1:
            for claim_0 in claims_0:
                edges['1-0'].add((claim_1, claim_0))
                edges['0-1'].add((claim_0, claim_1))

    # Mutual-evidence edges
    for claims_ in src_support_claims.values():
        for i in range(len(claims_)):
            for j in range(i + 1, len(claims_)):
                edges['mutual_evd'].add((claims_[i], claims_[j]))
                edges['mutual_evd'].add((claims_[j], claims_[i]))

    # Mutual-NP edges
    term_to_claim_ids = defaultdict(set)
    for claim_id, claim_text in tqdm(enumerate(claim_texts), desc="Extract NPs"):
        text = TextBlob(claim_text)
        for term in text.noun_phrases:
            term_to_claim_ids[term].add(claim_id)

    for claim_ids_ in term_to_claim_ids.values():
        claim_ids_ = list(claim_ids_)
        for i in range(len(claim_ids_)):
            for j in range(i + 1, len(claim_ids_)):
                edges['np'].add((claim_ids_[i], claim_ids_[j]))
                edges['np'].add((claim_ids_[j], claim_ids_[i]))

    for etype in list(edges):
        edges[etype] = np.array([list(e) for e in edges[etype]])

    
    # Build train test masks
    train_ids, val_ids, test_ids = set(df_train.id_left), set(df_dev.id_left), set(df_test.id_left)
    train_mask, val_mask, test_mask = [], [], []
    for id_left in claim_ori_ids:
        is_train, is_val, is_test = 0, 0, 0
        if id_left in train_ids:
            is_train = 1
        elif id_left in test_ids:
            is_test = 1
        else:
            is_val = 1
        train_mask.append(is_train)
        val_mask.append(is_val)
        test_mask.append(is_test)
        
    graphs = []
    for ignored_rels in [['mutual_evd'], 
                        ['mutual_evd', 'np'],
                        ['1-1', '0-0', '1-0', '0-1'],
                        ['1-1', '0-0', '1-0', '0-1', 'np']]:
        g = dgl.heterograph({
            ('claim', etype, 'claim') : (merged_edges_[:, 0], merged_edges_[:, 1])
            for etype, merged_edges_ in edges.items() if etype not in ignored_rels
        }, num_nodes_dict = {'claim' : len(claim_labels)}).to('cuda')
        g.ndata['train_mask'] = torch.BoolTensor(train_mask).cuda()
        g.ndata['val_mask'] = torch.BoolTensor(val_mask).cuda()
        g.ndata['test_mask'] = torch.BoolTensor(test_mask).cuda()
        g.ndata['label'] = torch.LongTensor(claim_labels).cuda()
        g.ndata['feature'] = torch.FloatTensor(claim_features).cuda()
        graphs.append(g)
    save_graphs(args.outpath, graphs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        help="TSV train file in GET folder",
        default="data/formatted_data/declare/Snopes/mapped_data/5fold/train_0.tsv",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="TSV test file in GET folder",
        default="data/formatted_data/declare/Snopes/mapped_data/5fold/test_0.tsv",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        help="TSV dev file in GET folder",
        default="data/formatted_data/declare/Snopes/mapped_data/dev.tsv",
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        help="Path to GET embedding file",
        default="/home/hoangdzung/Downloads/glove.6B/glove.6B.100d.txt",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        help="Path of binary output file which saves the generated graphs",
    )
    args = parser.parse_args()
    main(args)
