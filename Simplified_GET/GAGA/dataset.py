from torch.utils.data import Dataset
from collections import deque
from tqdm import tqdm
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, n_nodes, features, labels, alpha=None, adj_lists = None, edges_list = None, n_samples = [5, 5], is_train=True):
        self.n_nodes = n_nodes
        self.features = features
        self.labels = labels
        self.label_set = sorted(set(self.labels))
        if adj_lists is not None:
            self.adj_lists = adjadj_listslists
        else:
            self.adj_lists = self._process_edges(edges_list)
        self.n_relas = len(self.adj_lists[0]) 
        self.n_samples = n_samples
        if alpha is None:
            alpha = 1
        self.alpha = alpha 
        self.is_train = is_train
        
    def _process_edges(self, edges_list):
        adj_lists = []
        for target_id in tqdm(range(edges_list[0].shape[0])):
            neighs = []
            for edges in edges_list:
                neighs.append(edges.getrow(target_id).nonzero()[1])
            adj_lists.append(neighs)
        return adj_lists 
    
    def __len__(self):
        return self.n_nodes
    
    def _sample(self, target_node, n_samples, rela):

        queue = deque([target_node])
        res = []
        for n_sample in n_samples:
            curr_level = []
            len_queue = len(queue)
            for _ in range(len_queue):
                curr = queue.popleft()
                neighs = self.adj_lists[curr][rela]
                if len(neighs) == 0:
                    continue 
                neighs = np.random.choice(neighs, n_sample, replace = n_sample > len(neighs)).tolist()
                for neigh in neighs:
                    queue.append(neigh)
                    curr_level.append(neigh)
            res.append(curr_level)
        return res

    def __getitem__(self, target_node):
        target_label = self.labels[target_node]
        
        feats = []
        target_feat = self.features.getrow(target_node).todense().reshape(-1,)
        for rela in range(self.n_relas):
            khop_neighbors = self._sample(target_node, self.n_samples, rela)
            feats.append(target_feat)
            for hop, neighbors in enumerate(khop_neighbors):
                neighbors = np.array(neighbors)
                if len(neighbors) == 0:
                    for label in [-1] + self.label_set:
                        feats.append(np.zeros_like(target_feat)) # add 0s
                    continue 
                    
                labels = self.labels[neighbors] # get all labels
                labels[neighbors == target_node] = -1 # mask target node
                if not self.is_train:
                    for i, neighbor in enumerate(neighbors):
                        if neighbor in range(self.n_nodes):
                            labels[i] = -1

                for label in [-1] + self.label_set:
                    mask = labels == label
                    if mask.all() == False:
                        feats.append(np.zeros_like(target_feat)) # add 0s
                    else:
                        sub_neighbors = neighbors[mask]
                        feats.append(self.features[sub_neighbors].sum(axis=0).reshape(-1,) / len(sub_neighbors)**(self.alpha))

        feats = np.concatenate(feats)
        return torch.tensor(np.squeeze(np.asarray(feats)), dtype=torch.float32), int(target_label)