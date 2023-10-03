import torch
from torch import nn 

class GAGA(nn.Module):
    def __init__(self, indim, outdim, nheads, nlayers, nclasses, K, P, R):
        super(GAGA, self).__init__()
        self.indim = indim
        self.outdim = outdim 
        self.nheads = nheads
        self.nlayers = nlayers 
        self.K = K
        self.P = P
        self.R = R
        self._create_aux_input_params()
        self.linear = nn.Linear(indim, outdim) 
        self.hop_embedding = nn.Embedding(K + 1, outdim)
        self.rela_embedding = nn.Embedding(R, outdim)
        self.group_embedding = nn.Embedding(P, outdim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=outdim, nhead=nheads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.linear_out = nn.Linear(R * outdim, nclasses)

    def _create_aux_input_params(self):
        hop_input = [0]
        for hop in range(1, self.K + 1):
            hop_input += [hop] * self.P
        hop_input = hop_input * self.R

        rela_input = []
        for i in range(self.R):
            rela_input += [i] * (1 + self.K * self.P)

        group_input = []
        for _ in range(self.R):
            group_input.append(0)
            group_input += list(range(self.P)) * self.K
            
        self.hop_input_param = nn.Parameter(torch.LongTensor([hop_input]), requires_grad=False)
        self.rela_input_param = nn.Parameter(torch.LongTensor([rela_input]), requires_grad=False)
        self.group_input_param = nn.Parameter(torch.LongTensor([group_input]), requires_grad=False)
        
    def forward(self, x):
        Xs = self.linear(x)
        batch_size = x.shape[0]
        Xh = self.hop_embedding(self.hop_input_param.repeat(batch_size, 1))
        Xr = self.rela_embedding(self.rela_input_param.repeat(batch_size, 1))
        Xg = self.group_embedding(self.group_input_param.repeat(batch_size, 1))
        X = Xs + Xh + Xr + Xg
        
        X = self.transformer_encoder(X)
        logits = X[:, :: self.P * self.K + 1].reshape(batch_size, -1)
        output = self.linear_out(logits)
        
        return output
        