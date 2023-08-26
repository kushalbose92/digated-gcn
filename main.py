import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import os 
import dgl
import time 

import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from cora_loader import Cora
from citeseer_loader import CiteSeer
from pubmed_loader import PubMed

from webkb_loader import WebKBData
from wikipedianetwork_loader import WikipediaData
from actor_loader import ActorData

from train import *
from test import *
from utils import *

from model import *
import argparse


def argument_parser():

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help = 'enter name of dataset in smallcase', default = 'cora', type = str)
    parser.add_argument('--lr', help = 'learning rate', default = 0.2, type = float)
    parser.add_argument('--seed', help = 'Random seed', default = 100, type = int)
    parser.add_argument('--num_layers', help = 'number of hidden layers', default = 2, type = int)
    parser.add_argument('--hidden_dim', help = 'hidden dimension for node features', default = 16, type = int)
    parser.add_argument('--train_iter', help = 'number of training iteration', default = 100, type = int)
    parser.add_argument('--test_iter', help = 'number of test iterations', default = 1, type = int)
    parser.add_argument('--dropout', help = 'Dropoout in the layers', default = 0.60, type = float)
    parser.add_argument('--w_decay', help = 'Weight decay for the optimizer', default = 0.0005, type = float)
    parser.add_argument('--device', help = 'cpu or gpu device to be used', default = 'cpu', type = None)

    return parser

parsed_args = argument_parser().parse_args()

dataset = parsed_args.dataset
lr = parsed_args.lr
seed = parsed_args.seed
num_layers = parsed_args.num_layers
hidden_dim = parsed_args.hidden_dim
train_iter = parsed_args.train_iter
test_iter = parsed_args.test_iter
dropout = parsed_args.dropout
weight_decay = parsed_args.w_decay
device = parsed_args.device

print("Device: ", device)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == 'cuda:0':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
            

if dataset == 'Cora':
    data_obj = Cora()
elif dataset == 'Citeseer':
    data_obj = CiteSeer()
elif dataset == 'Pubmed':
    data_obj = PubMed()
elif dataset == 'Chameleon':
    data_obj = WikipediaData(dataset)
elif dataset == 'Wisconsin':
    data_obj = WebKBData(dataset)
elif dataset == 'Cornell':
    data_obj = WebKBData(dataset)
elif dataset == 'Texas':
    data_obj = WebKBData(dataset)
elif dataset == 'Film':
    data_obj = ActorData()
elif dataset == 'Squirrel':
    data_obj = WikipediaData(dataset)
elif dataset == 'Crocodile':
    data_obj = WikipediaData(dataset.lower())
else:
    print("Incorrect name of dataset")
 
print("Model Name: Gated Heterophily")
print("number of hidden layers:", num_layers)

print("------------------------------------------------")
print("Dataset:", dataset.upper())
print("number of features ", data_obj.num_features)
print("number of nodes ", data_obj.num_nodes)
print("number of edges ", data_obj.num_edges)
print("number of classes ", data_obj.num_classes)
# print("node labels ", data_obj.node_labels)
print("---------------------------------------------------")

def mask_generation(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype = torch.bool)
    mask[index] = 1
    return mask

def create_graph(data, train_idx, val_idx, test_idx):
    feat = data.node_features
    train_mask = mask_generation(train_idx, data.num_nodes)
    val_mask = mask_generation(val_idx, data.num_nodes)
    test_mask = mask_generation(test_idx, data.num_nodes)
    label = data.node_labels.to(device)
    src = data.edge_index[0]
    dst = data.edge_index[1]
    g = dgl.graph((src, dst)).to(device)
    g.ndata['feat'] = feat.to(device)
    g.ndata['train_mask'] = train_mask.to(device)
    g.ndata['val_mask'] = val_mask.to(device)
    g.ndata['test_mask'] = test_mask.to(device)
    g.ndata['label'] = label
    # print(label)
    return g

# def create_graph(G, features, labels, train_mask, val_mask, test_mask):
#     g = dgl.from_networkx(G).to(device)
#     # g = dgl.graph((src, dst)).to(device)
#     g.ndata['feat'] = features.to(device)
#     g.ndata['train_mask'] = train_mask.to(device)
#     g.ndata['val_mask'] = val_mask.to(device)
#     g.ndata['test_mask'] = test_mask.to(device)
#     g.ndata['label'] = labels.to(device)
#     print(train_mask.sum(), "  ", val_mask.sum(), "  ", test_mask.sum())
#     return g


test_acc_list = []
for fid in range(10):
    print(f"Training on file: {fid:01d}")
    # splitstr = 'splits/'+ dataset.lower() +'_split_0.6_0.2_'+str(fid)+'.npz'
    file_path = os.getcwd() + "/splits/" + dataset.lower() + '_split_0.6_0.2_'+str(fid)+'.npz'
    indices = np.load(file_path)
    train_idx, val_idx, test_idx = indices['train_mask'], indices['val_mask'], indices['test_mask']
    graph = create_graph(data_obj, train_idx, val_idx, test_idx)

    # G, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = full_load_data(dataset, splitstr)
    # graph = create_graph(G, features, labels, train_mask, val_mask, test_mask)

    model = DiGatedGCN(num_layers, data_obj.num_features, data_obj.num_classes, hidden_dim, dropout)
    model.to(device)
    opti = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    # training of the model
    print("Optimization started....")
    model_path = train(model, graph, dataset, train_iter, opti, device)
    print("Training finished....")

    # evaluation
    print("\nEvaluating on Test set")

    chkp = torch.load(model_path)
    model.load_state_dict(chkp['model_state_dict'])
    test_acc = test(model, graph)
    test_acc_list.append(test_acc)
    
    model.eval()
    src = graph.edges()[0]
    tgt = graph.edges()[1]
    edge_index = torch.zeros(2, src.shape[0], dtype=torch.long).to(device)
    edge_index[0], edge_index[1] = src, tgt
    out = model(graph.ndata['feat'], edge_index)
    visualize(out, graph.ndata['label'].detach().cpu().numpy(), dataset, num_layers, fid)

    print(f"Test Accuracy: {test_acc:.4f}")
    print()

print("Evaluation on all splits completed....")
print(test_acc_list)
print(f'Final Test Statistics: {np.mean(test_acc_list):.4f} || {np.std(test_acc_list):.4f}')



