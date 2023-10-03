import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import sys 
from tqdm import tqdm

from model import GAGA
from dataset import CustomDataset

amz = loadmat(sys.argv[1])
train_nodes, test_nodes = train_test_split(np.arange(len(amz['label'][0])), test_size=0.4)
train_nodes, val_nodes = train_test_split(train_nodes, test_size=1./3)

train_dataset = CustomDataset(len(train_nodes), amz['features'][train_nodes], amz['label'][0][train_nodes], alpha=1.5,\
                        edges_list=[amz['net_upu'][train_nodes, :][:, train_nodes],\
                                    amz['net_usu'][train_nodes, :][:, train_nodes], \
                                    amz['net_uvu'][train_nodes, :][:, train_nodes]], \
                        n_samples = [5, 5])

train_val_nodes = np.concatenate([val_nodes, train_nodes])
val_dataset = CustomDataset(len(val_nodes), amz['features'][train_val_nodes], amz['label'][0][train_val_nodes], alpha=1.5,\
                        edges_list=[amz['net_upu'][train_val_nodes, :][:, train_val_nodes],\
                                    amz['net_usu'][train_val_nodes, :][:, train_val_nodes],\
                                    amz['net_uvu'][train_val_nodes, :][:, train_val_nodes]], \
#                       adjlist=adj_list,
                        n_samples = [5, 5], is_train=False)

train_test_nodes = np.concatenate([test_nodes, train_nodes])
test_dataset = CustomDataset(len(test_nodes), amz['features'][train_test_nodes], amz['label'][0][train_test_nodes], alpha=1.5,\
                        edges_list=[amz['net_upu'][train_test_nodes, :][:, train_test_nodes],\
                                    amz['net_usu'][train_test_nodes, :][:, train_test_nodes],\
                                    amz['net_uvu'][train_test_nodes, :][:, train_test_nodes]], \
#                       adjlist=adj_list,
                        n_samples = [5, 5], is_train=False)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = GAGA(indim=25, outdim=16, nheads=4, nlayers=2, nclasses=2, K=2, P=3, R=3)
optimizer = torch.optim.Adam(model.parameters())
loss_funct = nn.CrossEntropyLoss()

for _ in range(10):
    preds = []
    labels = []
    for x, y in tqdm(train_dataloader):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = loss_funct(out, y)
        loss.backward()
        optimizer.step()
        pred = F.softmax(out, 1)[:, 1]
        preds += pred.detach().numpy().tolist()
        labels += y.numpy().tolist()

    train_score = roc_auc_score(labels, preds)
    
    preds = []
    labels = []
    for x, y in tqdm(val_dataloader):
        model.eval()
        optimizer.zero_grad()
        out = model(x)
        pred = F.softmax(out, 1)[:, 1]
        preds += pred.detach().numpy().tolist()
        labels += y.numpy().tolist()

    val_score = roc_auc_score(labels, preds)
    print(train_score, val_score)