import nltk
import re
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import torch
from tqdm import tqdm


class QueryDataset:
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
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return (adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)).A

    def _text_transform(self, input_, n):
        _MATCH_PUNC = re.compile(r"[^\w\s]")
        x = nltk.word_tokenize(input_)
        x = [token.lower() for token in x]
        x = [token for token in x if not _MATCH_PUNC.search(token)]
        return x[:n]

    def _read_text(self, df):
        self.claim_evidences = defaultdict(list)
        self.claim_ids = {}
        self.claim_texts, self.labels = [], []
        for i, (_, row) in tqdm(enumerate(df.iterrows()), desc="Read text"):
            if row.id_left not in self.claim_ids:
                self.claim_ids[row.id_left] = len(self.claim_ids)
                self.claim_texts.append(row.claim_text)
                self.labels.append(row.cred_label)
            self.claim_evidences[self.claim_ids[row.id_left]].append(i)
        self.evd_sources = df.evidence_source.apply(lambda x: x.strip()).values.tolist()
        self.evd_texts = df.evidence.values.tolist()

        self.tokenized_claim_texts = [
            self._text_transform(text, self.max_claim_len)
            for text in tqdm(self.claim_texts, desc="Transform claim")
        ]
        self.tokenized_evd_texts = [
            self._text_transform(text, self.max_evd_len)
            for text in tqdm(self.evd_texts, desc="Transform evidences")
        ]

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
        self.encoded_claim_texts = [
            vocab.transform(x)
            for x in tqdm(self.tokenized_claim_texts, desc="Encoding claim")
        ]
        self.encoded_evd_texts = [
            vocab.transform(x)
            for x in tqdm(self.tokenized_evd_texts, desc="Encoding evidence")
        ]
        if evd_vocab is not None:
            self.encoded_evidence_source = [
                evd_vocab.transform(x) for x in self.evd_sources
            ]

    def _convert_text(self, raw_text, fixed_length=30, window_size=5):
        # remove duplicate words in original order
        words_list = list(set(raw_text))
        words_list.sort(key=raw_text.index)
        words2id = {word: id for id, word in enumerate(words_list)}

        length_, length = len(words2id), len(raw_text)
        neighbours = [set() for _ in range(length_)]
        # window_size = window_size if fixed_length == 30 else 300
        for i, word in enumerate(raw_text):
            for j in range(max(i - window_size + 1, 0), min(i + window_size, length)):
                neighbours[words2id[word]].add(words2id[raw_text[j]])

        # gat graph
        adj = [
            [
                1 if (max(i, j) < length_) and (j in neighbours[i]) else 0
                for j in range(fixed_length)
            ]
            for i in range(fixed_length)
        ]
        words_list.extend([0 for _ in range(fixed_length - length_)])
        adj = self._laplacian_normalize(np.array(adj))
        return words_list, adj

    def build_data(self, window_size=5):
        self.claim_text, self.claim_adj = [], []
        for encoded_claim_text in tqdm(
            self.encoded_claim_texts, desc="Build claim graph"
        ):
            text_, adj = self._convert_text(
                encoded_claim_text, self.max_claim_len, window_size
            )
            self.claim_text.append(np.array(text_))
            self.claim_adj.append(np.array(adj))

        self.evd_text, self.evd_adj = [], []
        for encoded_evd_text in tqdm(
            self.encoded_evd_texts, desc="Build evidence graph"
        ):
            text_, adj = self._convert_text(
                encoded_evd_text, self.max_evd_len, window_size
            )
            self.evd_text.append(np.array(text_))
            self.evd_adj.append(np.array(adj))


class Vocabulary:
    def __init__(self, pad_value: str = "<PAD>", oov_value: str = "<OOV>"):
        """Vocabulary unit initializer."""
        self._pad = pad_value
        self._oov = oov_value
        self._state = {}
        self._state["term_index"] = self.TermIndex()
        self._state["index_term"] = dict()

    class TermIndex(dict):
        """Map term to index."""

        def __missing__(self, key):
            """Map out-of-vocabulary terms to index 1."""
            return 1

    def fit(self, tokens: set):
        """Build a :class:`TermIndex` and a :class:`IndexTerm`."""
        self._state["term_index"][self._pad] = 0
        self._state["term_index"][self._oov] = 1
        self._state["index_term"][0] = self._pad
        self._state["index_term"][1] = self._oov
        for index, term in enumerate(sorted(tokens)):
            self._state["term_index"][term] = index + 2
            self._state["index_term"][index + 2] = term

    def transform(self, input_: list) -> list:
        """Transform a list of tokens to corresponding indices."""
        return [self._state["term_index"][token] for token in input_]


class QueryTorchDataset(torch.utils.data.Dataset):
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
