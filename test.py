import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import time
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def test(model, graph):

    correct = 0
    model.eval()
    src = graph.edges()[0]
    tgt = graph.edges()[1]
    edge_index = torch.zeros(2, src.shape[0], dtype=torch.long).to(device)
    edge_index[0], edge_index[1] = src, tgt
    out = model(graph.ndata['feat'], edge_index)
    pred = out.argmax(dim = 1)
    pred = pred[graph.ndata['test_mask']]
    label = graph.ndata['label'][graph.ndata['test_mask']]
    correct = (pred == label).sum().item()
    accuracy = correct / int(graph.ndata['test_mask'].sum())

    return accuracy