import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch_geometric.utils import degree, remove_self_loops, add_self_loops

class DiGatedGCNLayer(nn.Module):
  def __init__(self, in_features, out_features, dropout):
    super(DiGatedGCNLayer, self).__init__()

    self.W = nn.Linear(in_features, out_features)
    self.U = nn.Linear(in_features, out_features)
    self.V = nn.Linear(2 * out_features, out_features)
    self.D = nn.Linear(out_features, out_features)
    self.E = nn.Linear(out_features, out_features)
    self.dropout = dropout

    nn.init.xavier_uniform_(self.W.weight)
    nn.init.xavier_uniform_(self.U.weight)
    nn.init.xavier_uniform_(self.V.weight)
    nn.init.xavier_uniform_(self.D.weight)
    nn.init.xavier_uniform_(self.E.weight)

  # directional edge gates 
  def edge_gates(self, h, edge_index):
    h_tilde = self.U(h)
    
    src = edge_index[0]
    dst = edge_index[1]
    
    h_src = h_tilde[src]
    h_dst = h_tilde[dst]
    
    h_i = self.D(h_src)
    h_j = self.E(h_dst)
    
    e = torch.cat([h_src, h_dst], dim=1)
    e = F.relu(self.V(e))
    e = e + (h_i + h_j)
    return e

  def forward(self, x, edge_index):

    edge_index = remove_self_loops(edge_index)[0]
    edge_index = add_self_loops(edge_index)[0]

    digates = self.edge_gates(x, edge_index)

    x = F.dropout(x, p=self.dropout, training=True)
    x = self.W(x)
    
    src_feats = x[edge_index[0]]
    tgt_feats = x[edge_index[1]]

    degrees = degree(edge_index[0])
    src_degrees = 1 / torch.sqrt(degrees[edge_index[0]])
    tgt_degrees = 1 / torch.sqrt(degrees[edge_index[1]])
    degree_scaler = src_degrees * tgt_degrees
    degree_scaler = degree_scaler.unsqueeze(1)
    degree_scaler = degree_scaler.tile((x.shape[1],))

    messages_from_neighbors = (tgt_feats * degree_scaler)
    messages_from_neighbors *= digates

    t1 = edge_index[0].view(edge_index[0].shape[0], 1).expand(-1, messages_from_neighbors.shape[1])
    unique_labels, labels_count = t1.unique(dim=0, return_counts=True)
    aggr_feats = torch.zeros_like(unique_labels, dtype=torch.float)
    x_final = aggr_feats.scatter_add_(0, t1, messages_from_neighbors)
    
    return x_final
    




