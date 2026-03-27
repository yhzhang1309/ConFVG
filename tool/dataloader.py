import torch
import pickle
import numpy as np
import random
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

data_route="./Dataset/"
data_route01="./Dataset/0.1/"

def online_dataloader(clear_dataset):
    online_dataset = []
    for element, edges, y, y_reduced in clear_dataset:
        valid_mask = y_reduced != -1
            
        filtered_element = element[valid_mask]
        filtered_y = y[valid_mask]
        filtered_y_reduced = y_reduced[valid_mask]
        
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(valid_mask)[0])}
        
        valid_edges_mask = np.logical_and(
            np.isin(edges[0], np.where(valid_mask)[0]),
            np.isin(edges[1], np.where(valid_mask)[0])
        )
        filtered_edges = edges[:, valid_edges_mask]
        
        filtered_edges = torch.LongTensor([
            [old_to_new.get(int(edges[0, i]), -1) for i in range(filtered_edges.shape[1])],
            [old_to_new.get(int(edges[1, i]), -1) for i in range(filtered_edges.shape[1])]
        ])
        
        valid_edge_indices = (filtered_edges != -1).all(dim=0)
        filtered_edges = filtered_edges[:, valid_edge_indices]
        
        if filtered_element.shape[0] > 0:
            online_dataset.append([filtered_element, filtered_edges, filtered_y, filtered_y_reduced])
    
    return online_dataset

def get_online():
    with open(data_route + 'dataset.dat', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def get_online_1():
    with open(data_route + 'dataset1.dat', 'rb') as f:
        dataset = pickle.load(f)
    return dataset
def get_online_001():
    with open(data_route + 'dataset0.01.dat', 'rb') as f:
        dataset = pickle.load(f)
    return dataset
def get_online_01():
    with open(data_route01 + 'dataset.dat', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def get_contrast():
    with open(data_route + 'datasettwo.dat', 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def get_online_dataset():
    with open(data_route + 'datasetonline.dat', 'rb') as f:
        dataset = pickle.load(f)
    clear_dataset = []
    for i in dataset:
        element = i[0]
        edges = i[1]
        y = element[:, 5]
        element = np.delete(element, 5, axis=1)
        element = torch.tensor(element, dtype=torch.float)
        edges = torch.LongTensor(edges).long()
        y = torch.tensor(y.T, dtype=torch.long)
        y = np.array(y, dtype=bool)
        y_reduced = np.full_like(y, fill_value=-1, dtype=np.int64)  
        mask = np.random.choice([True, False], size=y.shape, p=[1,0])  
        y_reduced[mask] = y[mask]  
        clear_dataset.append([element, edges, y, y_reduced])
        with open(data_route+"dataset.dat", "wb") as file:
            pickle.dump(clear_dataset, file)

    return clear_dataset

def get_online_dataset_label_noise():
    with open(data_route + 'datasetonline.dat', 'rb') as f:
        dataset = pickle.load(f)
    clear_dataset = []
    for i in dataset:
        element = i[0]
        edges = i[1]
        y = element[:, 5]
        element = np.delete(element, 5, axis=1)
        element = torch.tensor(element, dtype=torch.float)
        edges = torch.LongTensor(edges).long()
        y = torch.tensor(y.T, dtype=torch.long)
        y = np.array(y, dtype=bool)
        y_reduced = y.copy().astype(np.int64)
        flip_mask = np.random.choice([True, False], size=y.shape, p=[0.4, 0.6])
        y_reduced[flip_mask] = 1 - y_reduced[flip_mask]
        clear_dataset.append([element, edges, y, y_reduced])
        with open(data_route+"dataset.dat", "wb") as file:
            pickle.dump(clear_dataset, file)

    return clear_dataset

def create_contrast_dataset(dataset, num_pairs=1):
    contrast_dataset = []
    for graph_idx, (x,edge_index,y,y_reduced) in enumerate(dataset):
        num_nodes=x.size(0)
        data=Data(x=x,edge_index=edge_index,y=torch.tensor(y_reduced,dtype=torch.long))
        pos_mask = data.y == 1  
        neg_mask = data.y == 0  
        pos_nodes = torch.where(pos_mask)[0]
        neg_nodes = torch.where(neg_mask)[0]
    
        for _ in range(num_pairs):
            if len(pos_nodes) > 0:
                    pos_edge_index, _ = subgraph(pos_nodes, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
                    pos_x = data.x[pos_nodes]
                    contrast_dataset.append([pos_x, pos_edge_index])
        
            if len(neg_nodes) > 0:
                    neg_edge_index, _ = subgraph(neg_nodes, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
                    neg_x = data.x[neg_nodes]
                    contrast_dataset.append([neg_x, neg_edge_index])
    with open(data_route+"datasettwo.dat", "wb") as file:
        pickle.dump(contrast_dataset, file)
    
    return contrast_dataset



def get_online_dataset_old():
    with open(data_route+'datasetonline.dat','rb') as f:
        dataset=pickle.load(f)
    clear_dataset=[]
    for i in dataset:
        element=i[0]
        edges=i[1]
        y = element[:, 5]
        element = np.delete(element, 5, axis=1)
        element = torch.tensor(element, dtype=torch.float)
        y = torch.tensor(y.T, dtype=torch.long)
        edges = torch.LongTensor(edges).long()
        y= np.array(y,dtype=bool)
        clear_dataset.append([element,edges,y])
    return clear_dataset

def get_contrast_dataset_old():
    with open(data_route+'datasettwo.dat','rb') as f:
        datasettwo=pickle.load(f)
    clear_datasettwo=[]
    for i in datasettwo:
        element=i[0]
        edges=i[1]
        y = element[:, 5]
        element = np.delete(element, 5, axis=1)
        element = torch.tensor(element, dtype=torch.float)
        y = torch.tensor(y.T, dtype=torch.bool)
        edges = torch.LongTensor(edges).long()
        clear_datasettwo.append([element,edges,y])
    return clear_datasettwo

def get_contrast_dataset():
    with open(data_route+'datasettwo.dat','rb') as f:
        datasettwo=pickle.load(f)
    clear_datasettwo=[]
    for i in datasettwo:
        element=i[0]
        edges=i[1]
        y = element[:, 5]
        element = np.delete(element, 5, axis=1)
        element = torch.tensor(element, dtype=torch.float)
        edges = torch.LongTensor(edges).long()
        y = torch.tensor(y.T, dtype=torch.bool)
        y=np.array(y,dtype=bool)
        y_reduced=np.full_like(y, fill_value=-1, dtype=np.int64)  
        mask = np.random.choice([True, False], size=y.shape, p=[0.5, 0.95])  
        y_reduced[mask]=y[mask]
        clear_datasettwo.append([element,edges,y,y_reduced])
    return clear_datasettwo

def get_new_dataset():
    online_dataset = get_online_dataset()
    contrast_dataset = create_contrast_dataset(online_dataset)

