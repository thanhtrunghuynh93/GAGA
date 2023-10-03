# %% [markdown]
# 

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GGNN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GGNN, self).__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.linearz0 = nn.Linear(out_features, out_features)
        self.linearz1 = nn.Linear(out_features, out_features)
        self.linearr0 = nn.Linear(out_features, out_features)
        self.linearr1 = nn.Linear(out_features, out_features)
        self.linearh0 = nn.Linear(out_features, out_features)
        self.linearh1 = nn.Linear(out_features, out_features)
        
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, adj, x):
        if hasattr(self, 'dropout'): 
            x = self.dropout(x)
        x = self.proj(x)
        a = adj.matmul(x)

        z0 = self.linearz0(a)
        z1 = self.linearz1(x)
        z = torch.sigmoid(z0 + z1)

        r0 = self.linearr0(a)
        r1 = self.linearr1(x)
        r = torch.sigmoid(r0 + r1)

        h0 = self.linearh0(a)
        h1 = self.linearh1(r*x)
        h = torch.tanh(h0 + h1)

        feat = h*z + x*(1-z)
    
        return feat

class GSL(nn.Module):
    def __init__(self, rate):
        super(GSL, self).__init__()
        self.rate = rate

    def forward(self, adj, score):
        N = adj.shape[-1]
        BATCH_SIZE = adj.shape[0]
        num_preserve_node = int(self.rate * N)
        _, indices = score.topk(num_preserve_node, 1)
        indices = torch.squeeze(indices, dim=-1)
        mask = torch.zeros([BATCH_SIZE, N, N]).to(adj.get_device())
        for i in range(BATCH_SIZE):
            mask[i].index_fill_(0, indices[i], 1)
            mask[i].index_fill_(1, indices[i], 1)
        adj = adj * mask
        # feat = torch.tanh(score) * feat
        return adj

class GGNN_with_GSL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rate=0.8, dropout=0.2):
        super(GGNN_with_GSL, self).__init__()

        self.feat_prop1 = GGNN(input_dim, hidden_dim, dropout)
        self.word_scorer1 = GGNN(hidden_dim, 1, dropout)
        self.gsl1 = GSL(rate)

        self.feat_prop2 = GGNN(hidden_dim, output_dim, dropout)
        # self.word_scorer2 = GGNN(output_dim, 1, dropout)
        # self.gsl2 = GSL(rate)
    
    def forward(self, adj, feat):
        feat = self.feat_prop1(adj, feat)
        score = self.word_scorer1(adj, feat)
        adj_refined = self.gsl1(adj, score)
        feat = self.feat_prop2(adj_refined, feat)
        # score = self.word_scorer2(adj_refined, feat)
        # adj_refined = self.gsl2(adj_refined, score)
        return feat

class ConcatNotEqualSelfAtt(nn.Module):
    def __init__(self, inp_dim: int, hid_dim: int, num_heads: int = 1):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, hid_dim, bias=False)
        self.linear2 = nn.Linear(hid_dim, num_heads, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor):
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        assert self.inp_dim == (left.size(-1) + right.size(-1))  # due to concat
        B, L, D = right.size()
        left_tmp = left.unsqueeze(1).expand(B, L, -1)  # (B, 1, X)
        tsr = torch.cat([left_tmp, right], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(B, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attended = torch.bmm(right.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        return attended, attention_weights


class GET_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding=None, n_classes=2,
                word_att_num_heads=5, evd_att_num_heads=5, 
                 max_doc_len=100, max_num_evd=30,
                 rate=0.6, gnn_dropout=0.2):
        """
        input_dim: The dimension of (Glove) word embeddings (300)
        hidden_dim: The common hidden size of several GNN layers
        embedding: Glove embedding matrix, if None, we initialize it ourselves
        n_classess: number of clasess
        word_att_num_heads: Number of attention heads 
            (attention of word embeddings in an evidence sentence according to the claim embeddings)
        evd_att_num_heads: Number of attention heads
            (attiention of evidence embeddings in a set of evidences according to claim embeddings)
        max_doc_len: Maximum length of evidences
        max_num_evd: Maximum number of evidences of a claim
        rate: the keeping rate in GSL module (to keep only important edges)
        gnn_dropout: Dropout used in GNN modules
        """
        super(GET_Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_doc_len = max_doc_len
        self.max_num_evd = max_num_evd
        self.word_att_num_heads = word_att_num_heads
        self.evd_att_num_heads = evd_att_num_heads
        self.n_classes = n_classes
        if embedding is None:
            self.embedding = nn.Embedding(50000, self.input_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            
        self.ggnn4claim = GGNN(self.input_dim, self.hidden_dim)
        self.ggnn_with_gsl = GGNN_with_GSL(input_dim=self.input_dim, 
                              hidden_dim=self.hidden_dim, 
                              output_dim=self.hidden_dim, 
                              rate=rate, dropout=gnn_dropout)
        # We concat word embeddings and claim embeddings so inp_dim = self.hidden_dim * 2
        self.self_att_word = ConcatNotEqualSelfAtt(inp_dim=self.hidden_dim * 2, 
                                      hid_dim=self.hidden_dim, 
                                      num_heads=self.word_att_num_heads)
        
        # After word attention, embeddings of all heads are concatenated so the embedding size of evidence embeddings
        # is self.word_att_num_heads * self.hidden_dim
        self.self_att_evd = ConcatNotEqualSelfAtt(inp_dim=self.word_att_num_heads * self.hidden_dim + self.hidden_dim, 
                                     hid_dim=self.hidden_dim, 
                                     num_heads=self.evd_att_num_heads)
        
        # After evd attention, embeddings of all heads are concatenated so the embedding size of evidence embeddings
        # is self.word_att_num_heads *  * self.evd_att_num_heads * self.hidden_dim        
        evd_input_size = self.word_att_num_heads * self.evd_att_num_heads * self.hidden_dim + self.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(evd_input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_classes)
        )

    def _query_embedding(self, query, adj, query_mask):
        """
        query: tensor of size (BATCH_SIZE, MAX_CLAIM_LEN) 
        adj: tensor of size (BATCH_SIZE, MAX_CLAIM_LEN, MAX_CLAIM_LEN)
        query_mask: tensor of size (BATCH_SIZE, MAX_CLAIM_LEN)
        """
        # Do the look up table
        embed_query = self.embedding(query)
        
        # Get the word embeddings of the claim
        query_gnn_hiddens = self.ggnn4claim(adj.float(), embed_query)
        
        # Basically, the claim embedding is the average of its word embeddings
        sum_query_repr = (query_gnn_hiddens * query_mask.unsqueeze(-1)).sum(1)
        query_repr = sum_query_repr / query_mask.sum(1, keepdim=True)
        assert query_repr.shape[-1] == self.hidden_dim
        return query_repr
    
    def _expand_query_embedding(self, query_repr, evd_count_per_query):
        """
        query_repr: tensor of (BATCH_SIZE, HIDDEN_DIM)
        evd_count_per_query: tensor (BATCH_SIZE, ): store the number of evidences for each claim
        -> output: tensor of size (sum(evd_count_per_query, HIDDEN_DIM))
        """
        expanded_query_repr = []
        for num_evd, tsr in zip(evd_count_per_query, query_repr):
            tmp = tsr.clone()
            tsr = tmp.expand(num_evd, self.hidden_dim)
            expanded_query_repr.append(tsr)
        expanded_query_repr = torch.cat(expanded_query_repr, dim = 0)
        return expanded_query_repr

    def _doc_embedding(self, doc, doc_adj):
        """
        doc: tensor size = (sum(evd_count_per_query), MAX_DOC_LEN)
        doc_adj : tensor size = (sum(evd_count_per_query), MAX_DOC_LEN, MAX_DOC_LEN)
        
        -> output: tensor of size = (sum(evd_count_per_query), HIDDEN_DIM)
        """
        embed_doc = self.embedding(doc)
        doc_out_ggnn = self.ggnn_with_gsl(doc_adj.float(), embed_doc)
        assert doc_out_ggnn.shape[1:] == (self.max_doc_len, self.hidden_dim)
        return doc_out_ggnn
    
    def _word_attention_doc_embedding(self, expanded_query_repr, doc_out_ggnn, doc_mask):
        """
        expanded_query_repr : size (sum(evd_count_per_query, HIDDEN_DIM))
        doc_out_ggnn: size (sum(evd_count_per_query), HIDDEN_DIM)
        doc_mask : (sum(evd_count_per_query), HIDDEN_DIM) 
        """
        avg, _ = self.self_att_word(expanded_query_repr, doc_out_ggnn, doc_mask)
        avg = torch.flatten(avg, start_dim=1)
        assert avg.shape == (doc_out_ggnn.shape[0], self.word_att_num_heads * self.hidden_dim)
        return avg
    
    def _pad_doc_embedding(self, attentioned_doc_repr, evd_count_per_query):
        last = 0
        padded_doc_repr = []
        for idx in range(evd_count_per_query.shape[0]):
            num_evd = evd_count_per_query[idx].item()
            hidden_vectors = attentioned_doc_repr[last: last + num_evd]  # (n1, H)
            padded = F.pad(hidden_vectors, (0, 0, 0, self.max_num_evd - num_evd), "constant", 0)
            padded_doc_repr.append(padded)
            last += num_evd
        padded_doc_repr = torch.stack(padded_doc_repr, dim=0)
        assert padded_doc_repr.shape == (evd_count_per_query.shape[0], 
                                         self.max_num_evd, 
                                         self.word_att_num_heads * self.hidden_dim)
        return padded_doc_repr

    def _evd_attention_doc_embedding(self, query_repr, padded_doc_repr, evd_count_per_query):
        batch_size = evd_count_per_query.shape[0]
        doc_mask = torch.arange(self.max_num_evd).repeat(batch_size, 1).to(query_repr.get_device())
        doc_mask = doc_mask < evd_count_per_query.unsqueeze(1)
        doc_mask = doc_mask.float()
        attended_avg, _ = self.self_att_evd(query_repr, padded_doc_repr, doc_mask)
        avg = torch.flatten(attended_avg, start_dim=1)
        assert avg.shape == (batch_size, self.word_att_num_heads * self.evd_att_num_heads * self.hidden_dim)
        
        return avg 
    
    def forward(self, query, query_adj, query_mask, doc, doc_adj, doc_mask, evd_count_per_query):
        """
        query: tensor of word ids, size (BATCH_SIZE, MAX_CLAIM_LEN)
        query_adj: tensor, size (BATCH_SIZE, MAX_CLAIM_LEN, MAX_CLAIM_LEN)
        query_mask: tensor, size (BATCH_SIZE, MAX_CLAIM_LEN)
        doc: tensor of word ids size (sum(evd_count_per_query), MAX_DOC_LEN)
        doc_adj: tensor of word ids size (sum(evd_count_per_query), MAX_DOC_LEN, MAX_DOC_LEN)
        doc_mask: tensor of word ids size (sum(evd_count_per_query), MAX_DOC_LEN)
        evd_count_per_query: (BATCH_SIZE, )
        """
        # (batch_size, hidden_dim)
        query_repr = self._query_embedding(query, query_adj, query_mask)
        
        #(sum(evd_count_per_query), hidden_dim)
        expanded_query_repr = self._expand_query_embedding(query_repr, evd_count_per_query)
        
        #(sum(evd_count_per_query), max_doc_len, hidden_dim) 
        doc_out_ggnn = self._doc_embedding(doc, doc_adj)
        
        #(sum(evd_count_per_query), word_att_num_heads * hidden_dim) 
        attentioned_doc_repr = self._word_attention_doc_embedding(expanded_query_repr, doc_out_ggnn, doc_mask)

        # (batch_size, max_num_evd, word_att_num_heads * hidden_dim) 
        padded_doc_repr = self._pad_doc_embedding(attentioned_doc_repr, evd_count_per_query)

        import pdb; pdb.set_trace()
        doc_repr = self._evd_attention_doc_embedding(query_repr, padded_doc_repr, evd_count_per_query)
        
        query_doc_repr = torch.cat([query_repr, doc_repr], dim=-1)
        logit = self.mlp(query_doc_repr)
        return logit


# %%
import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

import nltk
import re
import pandas as pd
import numpy as np
import scipy.sparse as sp


# %%
df_train = pd.read_csv('./formatted_data/declare/Snopes/mapped_data/5fold/train_0.tsv', delimiter='\t')
df_test = pd.read_csv('./formatted_data/declare/Snopes/mapped_data/5fold/test_0.tsv', delimiter='\t')
df_dev = pd.read_csv('./formatted_data/declare/Snopes/mapped_data/dev.tsv', delimiter='\t')

# %%
for df in [df_train, df_test, df_dev]:
    print("There are at most {} evidences per claim"\
          .format(df.groupby('id_left').count().cred_label.max()))

# %%
df_train.head()

# %%
class Data:
    def __init__(self, df, max_claim_len=30, max_evd_len=100):
        self.max_claim_len = max_claim_len
        self.max_evd_len = max_evd_len
        self._read_text(df)
        self.text_corpus = None
        self.src_corpus = None
    
    def _laplacian_normalize(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)).A

    def _text_transform(self, input_, n):
        _MATCH_PUNC = re.compile(r'[^\w\s]')
        x = nltk.word_tokenize(input_)
        x = [token.lower() for token in x]
        x = [token for token in x if not _MATCH_PUNC.search(token)]
        return x[: n]

    def _read_text(self, df):        
        self.claim_evidences = defaultdict(list)
        self.claim_ids = {}
        self.claim_texts, self.labels = [], []
        for i, (_, row) in tqdm(enumerate(df.iterrows()), desc='Read text'):
            if row.id_left not in self.claim_ids:
                self.claim_ids[row.id_left] = len(self.claim_ids)
                self.claim_texts.append(row.claim_text)
                self.labels.append(row.cred_label)
            self.claim_evidences[self.claim_ids[row.id_left]].append(i)
        self.evd_sources = df.evidence_source.apply(lambda x: x.strip()).values.tolist()
        self.evd_texts = df.evidence.values.tolist()
        
        self.tokenized_claim_texts = [self._text_transform(text, self.max_claim_len) for text in tqdm(self.claim_texts, desc='Transform claim')]
        self.tokenized_evd_texts = [self._text_transform(text, self.max_evd_len) for text in tqdm(self.evd_texts, desc='Transform evidences')]
    
    def get_text_corpus(self, rebuild=False):
        if self.text_corpus is not None and not rebuild:
            return self.text_corpus
        self.text_corpus = set()
        for tokenized_claim_text in self.tokenized_claim_texts:
            self.text_corpus.update(tokenized_claim_text)
        for tokenized_evd_text in self.tokenized_evd_texts:
            self.text_corpus.update(tokenized_evd_text)
            
        return self.text_corpus
    
    def get_src_corpus(self):
        return set(self.evd_sources)

    def encoding(self, vocab, evd_vocab=None):
        self.encoded_claim_texts = [vocab.transform(x) for x in tqdm(self.tokenized_claim_texts)]
        self.encoded_evd_texts = [vocab.transform(x) for x in tqdm(self.tokenized_evd_texts)]
        if evd_vocab is not None:
            self.encoded_evidence_source = [evd_vocab.transform(x) for x in self.evd_sources]
           
    def _convert_text(self, raw_text, fixed_length=30, window_size=5):
        words_list = list(set(raw_text))       # remove duplicate words in original order
        words_list.sort(key=raw_text.index)
        words2id = {word: id for id, word in enumerate(words_list)}

        length_, length = len(words2id), len(raw_text)
        neighbours = [set() for _ in range(length_)]
        # window_size = window_size if fixed_length == 30 else 300
        for i, word in enumerate(raw_text):
            for j in range(max(i-window_size+1, 0), min(i+window_size, length)):
                neighbours[words2id[word]].add(words2id[raw_text[j]])

        # gat graph
        adj = [[1 if (max(i, j) < length_) and (j in neighbours[i]) else 0 for j in range(fixed_length)]
               for i in range(fixed_length)]
        words_list.extend([0 for _ in range(fixed_length-length_)])
        adj = self._laplacian_normalize(np.array(adj))
        return words_list, adj
    
    def build_data(self, window_size=5): 
        self.claim_text, self.claim_adj = [], []
        for encoded_claim_text in tqdm(self.encoded_claim_texts):
            text_, adj = self._convert_text(encoded_claim_text, self.max_claim_len, window_size)
            self.claim_text.append(np.array(text_))
            self.claim_adj.append(np.array(adj))

        self.evd_text, self.evd_adj = [], []
        for encoded_evd_text in tqdm(self.encoded_evd_texts):
            text_, adj = self._convert_text(encoded_evd_text, self.max_evd_len, window_size)
            self.evd_text.append(np.array(text_))
            self.evd_adj.append(np.array(adj))


# %%
train_data = Data(df_train)
dev_data = Data(df_dev)
test_data = Data(df_test)

# %%
class Vocabulary:

    def __init__(self, pad_value: str = '<PAD>', oov_value: str = '<OOV>'):
        """Vocabulary unit initializer."""
        self._pad = pad_value
        self._oov = oov_value
        self._state = {}
        self._state['term_index'] = self.TermIndex()
        self._state['index_term'] = dict()

    class TermIndex(dict):
        """Map term to index."""

        def __missing__(self, key):
            """Map out-of-vocabulary terms to index 1."""
            return 1

    def fit(self, tokens: set):
        """Build a :class:`TermIndex` and a :class:`IndexTerm`."""
        self._state['term_index'][self._pad] = 0
        self._state['term_index'][self._oov] = 1
        self._state['index_term'][0] = self._pad
        self._state['index_term'][1] = self._oov
        for index, term in enumerate(sorted(tokens)):
            self._state['term_index'][term] = index + 2
            self._state['index_term'][index + 2] = term

    def transform(self, input_: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._state['term_index'][token] for token in input_]


# %%
vocab = Vocabulary()
vocab.fit(train_data.get_text_corpus())
evd_vocab = Vocabulary()
evd_vocab.fit(train_data.get_src_corpus())

# %%
term_index = vocab._state['term_index']
embedding_data = {}
output_dim = 0
count_word_hit = 0
file_path = '/home/datht/glove.6B.300d.txt'
with open(file_path, 'r', encoding = "utf-8") as f:
    output_dim = len(f.readline().rstrip().split(' ')) - 1
    f.seek(0)
    for line in tqdm(f):
        current_line = line.rstrip().split(' ')
        if current_line[0] not in term_index: continue
        embedding_data[current_line[0]] = current_line[1:]
        count_word_hit += 1

    print("Word hit: " + str((count_word_hit, len(term_index))) + " " + str(count_word_hit / len(term_index) * 100))


# %%
input_dim = len(term_index)
matrix = np.empty((input_dim, output_dim))
valid_keys = embedding_data.keys()
for term, index in sorted(term_index.items(), key = lambda x: x[1]):  # Starting the smallest index to the largest
    if term in valid_keys:
        matrix[index] = embedding_data[term]
    else:
        matrix[index] = np.random.uniform(-0.2, 0.2)


# %%
train_data.encoding(vocab, evd_vocab)
dev_data.encoding(vocab, evd_vocab)
test_data.encoding(vocab, evd_vocab)

# %%
train_data.build_data()
dev_data.build_data()
test_data.build_data()


# %%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_evd_per_claim=30):
        self.claim_text = data.claim_text
        self.claim_adj = data.claim_adj
        self.evd_text = data.evd_text
        self.evd_adj = data.evd_adj
        self.labels = data.labels
        self.claim_evidences = data.claim_evidences
        self.n_evd_per_claim = n_evd_per_claim

    def _text_to_mask(self, text):
        return (text > 0).astype(int)
    
    def __getitem__(self, idx):
        query = self.claim_text[idx]
        query_adj = self.claim_adj[idx]
        query_mask = self._text_to_mask(query)
        label = self.labels[idx]
        n_evds = len(self.claim_evidences[idx])
        evds, evd_adjs, evd_masks = [], [], []
        for evd_idx in self.claim_evidences[idx]:
            evds.append(self.evd_text[evd_idx])
            evd_adjs.append(self.evd_adj[evd_idx])
            evd_masks.append(self._text_to_mask(evds[-1]))
        evds = np.stack(evds)
        evd_adjs = np.stack(evd_adjs)
        evd_masks = np.stack(evd_masks)

        return query, query_adj, query_mask, evds, evd_adjs, evd_masks, n_evds, label

    def __len__(self):
        return len(self.labels)

# %%
train_dataset = MyDataset(train_data)
dev_dataset = MyDataset(dev_data)
test_dataset = MyDataset(test_data)

# %%
def collate_batch(batch):
    query_list, query_adj_list, query_mask_list, evd_list, evd_adj_list, evd_mask_list, n_evd_list, label_list \
    = [], [], [], [], [], [], [], []

    for query, query_adj, query_mask, evds, evd_adjs, evd_masks, n_evds, label in batch:
        query_list.append(query)
        query_adj_list.append(query_adj)
        query_mask_list.append(query_mask)
        evd_list.append(evds)
        evd_adj_list.append(evd_adjs)
        evd_mask_list.append(evd_masks)
        n_evd_list.append(n_evds)
        label_list.append(label)
    
    query_list = torch.LongTensor(np.stack(query_list))
    query_adj_list = torch.FloatTensor(np.stack(query_adj_list))
    query_mask_list = torch.LongTensor(np.stack(query_mask_list))

    evd_list = torch.LongTensor(np.vstack(evd_list))
    evd_adj_list = torch.FloatTensor(np.vstack(evd_adj_list))
    evd_mask_list = torch.LongTensor(np.vstack(evd_mask_list))
    
    n_evd_list = torch.LongTensor(n_evd_list)
    label_list = torch.LongTensor(label_list)
    return (query_list, query_adj_list, query_mask_list, evd_list, evd_adj_list, evd_mask_list, n_evd_list), label_list
train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_batch, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=8, collate_fn=collate_batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_batch, shuffle=False)

# %%
get_model = GET_Model(input_dim=300, hidden_dim=300,\
                      embedding=torch.FloatTensor(matrix),\
                      max_num_evd=30, word_att_num_heads=5,
                     evd_att_num_heads=2).cuda()

# %%
optimizer = torch.optim.Adam(get_model.parameters(), lr=1e-4, weight_decay=1e-3)
loss_fn = nn.CrossEntropyLoss()

# %%
for _ in range(2):
    total_loss = 0
    for x, y in tqdm(train_loader):
        x = tuple([i.cuda() for i in x])
        y = y.cuda()
        logits = get_model(*x)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(get_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        # break
    print(total_loss)

# %%
get_model.eval()

with torch.no_grad():
    all_logits = []
    all_trues = []
    for x, y in tqdm(test_loader):
        x = tuple([i.cuda() for i in x])
        y = y.cuda()
        logits = get_model(*x)
        all_logits.append(logits)
        all_trues.append(y)
all_trues = torch.cat(all_trues).detach().cpu().numpy()
all_logits = torch.cat(all_logits, dim=0)
predicts = all_logits.argmax(dim=1).detach().cpu().numpy()


# %%
from sklearn.metrics import f1_score

print(f1_score(all_trues, predicts, average="micro"))
print(f1_score(all_trues, predicts, average="macro"))


import pdb; 

print('11')
