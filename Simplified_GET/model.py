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
        mask = torch.zeros([BATCH_SIZE, N, N])
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
    def __init__(self, input_dim, hidden_dim, n_classes=2,
                 word_att_hidden_dim=32, word_att_num_heads=5, 
                 evd_att_hidden_dim=32, evd_att_num_heads=5, 
                 max_doc_len=100, max_num_evd=30,
                 rate=0.5, dropout=0.5):
        super(GET_Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_doc_len = max_doc_len
        self.max_num_evd = max_num_evd
        self.word_att_hidden_dim = word_att_hidden_dim
        self.word_att_num_heads = word_att_num_heads
        self.evd_att_hidden_dim = evd_att_hidden_dim
        self.evd_att_num_heads = evd_att_num_heads
        self.n_classes = n_classes
        
        self.embedding = nn.Embedding(1000, self.input_dim)
        
        self.ggnn4claim = GGNN(self.input_dim, self.hidden_dim)
        self.ggnn_with_gsl = GGNN_with_GSL(input_dim=INPUT_DIM, 
                              hidden_dim=DOC_GNN_HIDDEN_DIM, 
                              output_dim=HIDDEN_DIM, 
                              rate=rate, dropout=dropout)
        
        self.self_att_word = ConcatNotEqualSelfAtt(inp_dim=self.hidden_dim * 2, 
                                      hid_dim=self.word_att_hidden_dim, 
                                      num_heads=self.word_att_num_heads)
        self.self_att_evd = ConcatNotEqualSelfAtt(inp_dim=self.word_att_num_heads * self.hidden_dim + self.hidden_dim, 
                                     hid_dim=self.evd_att_hidden_dim, 
                                     num_heads=self.evd_att_num_heads)
        
        evd_input_size = self.word_att_num_heads * self.evd_att_num_heads * self.hidden_dim + self.hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(evd_input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_classes)
        )

    def _query_embedding(self, query, adj, query_mask):
        embed_query = self.embedding(query)
        query_gnn_hiddens = self.ggnn4claim(adj.float(), embed_query)
        sum_query_repr = (query_gnn_hiddens * query_mask.unsqueeze(-1)).sum(1)
        query_repr = sum_query_repr / query_mask.sum(1, keepdim=True)
        assert query_repr.shape[-1] == self.hidden_dim
        return query_repr
    
    def _expand_query_embedding(self, query_repr, evd_count_per_query):
        expanded_query_repr = []
        for num_evd, tsr in zip(evd_count_per_query, query_repr):
            tmp = tsr.clone()
            tsr = tmp.expand(num_evd, self.hidden_dim)
            expanded_query_repr.append(tsr)
        expanded_query_repr = torch.cat(expanded_query_repr, dim = 0)
        return expanded_query_repr

    def _doc_embedding(self, doc, doc_adj):
        embed_doc = self.embedding(doc)
        doc_out_ggnn = self.ggnn_with_gsl(doc_adj.float(), embed_doc)
        assert doc_out_ggnn.shape[1:] == (self.max_doc_len, self.hidden_dim)
        return doc_out_ggnn
    
    def _word_attention_doc_embedding(self, expanded_query_repr, doc_out_ggnn, doc_mask):
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
        doc_mask = torch.arange(self.max_num_evd).repeat(batch_size, 1)
        doc_mask = doc_mask < evd_count_per_query.unsqueeze(1)
        doc_mask = doc_mask.float()
        attended_avg, _ = self.self_att_evd(query_repr, padded_doc_repr, doc_mask)
        avg = torch.flatten(attended_avg, start_dim=1)
        assert avg.shape == (batch_size, self.word_att_num_heads * self.evd_att_num_heads * self.hidden_dim)
        
        return avg 
    
    def forward(self, query, query_adj, query_mask, doc, doc_adj, doc_mask, evd_count_per_query):
        query_repr = self._query_embedding(query, query_adj, query_mask)
        expanded_query_repr = self._expand_query_embedding(query_repr, evd_count_per_query)
        doc_out_ggnn = self._doc_embedding(doc, doc_adj)
        attentioned_doc_repr = self._word_attention_doc_embedding(expanded_query_repr, doc_out_ggnn, doc_mask)

        padded_doc_repr = self._pad_doc_embedding(attentioned_doc_repr, evd_count_per_query)
        doc_repr = self._evd_attention_doc_embedding(query_repr, padded_doc_repr, evd_count_per_query)
        query_doc_repr = torch.cat([query_repr, doc_repr], dim=-1)
        logit = self.mlp(query_doc_repr)
        
        return logit