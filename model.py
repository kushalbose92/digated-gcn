import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import MiniGCDataset
import dgl

from layer import DiGatedGCNLayer

   
class DiGatedGCN(nn.Module):
  def __init__(self, num_layers, input_dim, output_dim, hidden_dim, dropout):
    super().__init__()
    self.input = input_dim
    self.output = output_dim
    self.hidden = hidden_dim
    self.num_layers = num_layers
    self.dropout = dropout
    self.gated_gcn_layers = nn.ModuleList()
    
    for i in range(self.num_layers):
      if i == 0 and self.num_layers == 1:
        model = DiGatedGCNLayer(self.input, self.output, dropout)
      elif i == 0 and self.num_layers > 1:
        model = DiGatedGCNLayer(self.input, self.hidden, dropout)
      elif i > 0 and i < self.num_layers - 1:
        model = DiGatedGCNLayer(self.hidden, self.hidden, dropout)
      else:
        model = DiGatedGCNLayer(self.hidden, self.output, dropout)
    
      self.gated_gcn_layers.append(model)
  
  def forward(self, h, edge_index):

    for i in range(self.num_layers):
        h = self.gated_gcn_layers[i](h, edge_index)
        if i != self.num_layers-1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=True)
   
    return F.log_softmax(h, dim = 1)






# from torch_geometric.nn import DiGatedGCN, GCN2Conv
    
# class GCN(nn.Module):
#     def __init__(self, data_obj, num_layers, hidden_dim, dropout, device):
#         super(GCN, self).__init__()
#         self.data_obj = data_obj
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.num_layers = num_layers
#         self.device = device
        
#         self.gcn_convs = nn.ModuleList()
#         self.lin_layers = nn.ModuleList()
        
#         for i in range(self.num_layers):
#             self.gcn_convs.append(DiGatedGCN(self.hidden_dim, self.hidden_dim, catched=True))
#             self.lin_layers.append(nn.Linear(2 * hidden_dim, self.hidden_dim))

#         self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
#         self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)

#     def forward(self, x_h, edge_index):
    
#         x_h = self.init_w(x_h)

#         for i in range(self.num_layers):
#             x_h = self.gcn_convs[i](x_h, edge_index)

#             if i != self.num_layers-1:
                
#                 x_h = F.dropout(x_h, p=self.dropout, training=self.training)
#                 x_h = F.relu(x_h)
    
#         x_h = self.last_w(x_h)
                
#         embedding = x_h
#         x_h = F.log_softmax(x_h, dim=1)

#         return embedding, x_h


# class GCNII(nn.Module):
#     def __init__(self, data_obj, num_layers, hidden_dim, dropout, device):
#         super(GCNII, self).__init__()
#         self.num_layers = num_layers
#         self.data_obj = data_obj
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.theta = 0.1
#         self.beta = 0.1
#         self.device = device
        
#         self.gcn_convs = nn.ModuleList()
#         self.lin_layers = nn.ModuleList()
    
#         for i in range(self.num_layers):
#             self.gcn_convs.append(GCN2Conv(self.hidden_dim, self.theta, self.beta, i+1))
#             self.lin_layers.append(nn.Linear(2 * hidden_dim, self.hidden_dim))

#         self.init_w = nn.Linear(self.data_obj.num_features, self.hidden_dim)
#         self.last_w = nn.Linear(self.hidden_dim, self.data_obj.num_classes)

#     def forward(self, x_h, edge_index):

#         x_h = self.init_w(x_h)
#         x_0 = x_h

#         for i in range(self.num_layers):
#             x_h = self.gcn_convs[i](x_h, x_0, edge_index)

#             if i != self.num_layers-1:
                
#                 x_h = F.dropout(x_h, p=self.dropout, training=self.training)
#                 x_h = F.relu(x_h)
    
#         x_h = self.last_w(x_h)
                
#         embedding = x_h
#         x_h = F.log_softmax(x_h, dim=1)

#         return embedding, x_h


# import torch
# import torch.nn as nn 
# import torch.nn.functional as F

# import torch_geometric.nn as pyg_nn 
# from torch_geometric.nn import DiGatedGCN
# from torch_geometric.utils import dropout_edge, add_random_edge


# class BiGDC(nn.Module):
#     def __init__(self, feat_dim, hidden_dim, num_classes, num_layers, p_remove, p_add, num_steps, dropout, device):
#         super(BiGDC, self).__init__()
#         self.feat_dim = feat_dim
#         self.hidden_dim = hidden_dim
#         self.p_remove = p_remove
#         self.p_add = p_add
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.num_steps = num_steps
#         self.device = device
#         self.dropout = dropout
        
#         self.gcn_1 = DiGatedGCN(self.feat_dim, self.hidden_dim).to(self.device)
#         self.gcn_2 = DiGatedGCN(self.hidden_dim, self.num_classes).to(self.device)
        
#     def forward(self, x, edge_index):
#         x = x.to(self.device)
#         edge_index = edge_index.to(self.device)
#         added_edge_index = edge_index
#         for _ in range(self.num_steps):
#             removed_edge_index, removed_edge_mask = dropout_edge(added_edge_index, p=self.p_remove, force_undirected=True)
#             added_edge_index, added_edges = add_random_edge(removed_edge_index, p=self.p_add, force_undirected=True)
#         rewired_edge_index = added_edge_index
#         x = self.gcn_1(x, rewired_edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, self.dropout, training = self.training)
#         x = self.gcn_2(x, rewired_edge_index)
#         embedding = x 
#         x = F.log_softmax(x, dim = 1)
#         return embedding, x
        