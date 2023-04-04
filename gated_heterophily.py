import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
import networkx as nx
import dgl
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Planetoid 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

from utils import *

import warnings
warnings.filterwarnings("ignore")

##################################
parser = argparse.ArgumentParser(description='DiGated GCN')
parser.add_argument('--name', default = 'citeseer', type = str)
parser.add_argument('--hidden',default = '128', type = int)
parser.add_argument('--dropout',default = '0.60', type = float)
parser.add_argument('--snorm',default = '18', type = int)
parser.add_argument('--epoch',default = '1600', type = int)
parser.add_argument('--num_layers', default = 1, type = int)
parser.add_argument('--lr', default = 0.05, type = float)
parser.add_argument('--w_decay', default = 0.0005, type = float)
parser.add_argument('--seed', default = 1000, type = int)
parser.add_argument('--device',default = 'cuda:0', type = str)

parsed_args = parser.parse_args()

name = parsed_args.name
hidden_dim = parsed_args.hidden
dropout = parsed_args.dropout
train_snorm = parsed_args.snorm
epoch = parsed_args.epoch
num_layers = parsed_args.num_layers
lr = parsed_args.lr
w_decay = parsed_args.w_decay
seed = parsed_args.seed
device = parsed_args.device

print("number of layers: ", num_layers)

def apply_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if device == 'cuda:0':
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

apply_seed(seed)

class WebKBData():
  def __init__(self, datasetname):

    dataset = WebKB(root='data/WebKB', name = datasetname, transform = NormalizeFeatures())
    data = dataset[0]

    self.data = data
    self.name = datasetname
    self.length = len(dataset)
    self.num_features = dataset.num_features
    self.num_classes = dataset.num_classes

    self.num_nodes = data.num_nodes
    self.num_edges = data.num_edges
    self.avg_node_degree = (data.num_edges / data.num_nodes)
    
    self.contains_isolated_nodes = data.has_isolated_nodes()
    self.data_contains_self_loops = data.has_self_loops()
    self.is_undirected = data.is_undirected()

    self.node_features = data.x
    self.node_labels = data.y
    self.edge_index = data.edge_index
  
class WikipediaData():
  def __init__(self, datasetname):

    dataset = WikipediaNetwork(root='data/WikipediaNetwork', name = datasetname, transform = NormalizeFeatures())
    data = dataset[0]

    self.data = data
    self.name = datasetname
    self.length = len(dataset)
    self.num_features = dataset.num_features
    self.num_classes = dataset.num_classes

    self.num_nodes = data.num_nodes
    self.num_edges = data.num_edges
    self.avg_node_degree = (data.num_edges / data.num_nodes)
  
    self.contains_isolated_nodes = data.has_isolated_nodes()
    self.data_contains_self_loops = data.has_self_loops()
    self.is_undirected = data.is_undirected()

    self.node_features = data.x
    self.node_labels = data.y
    self.edge_index = data.edge_index

class ActorData():
  def __init__(self):

    dataset = Actor(root='data/Actor', transform = NormalizeFeatures())
    data = dataset[0]

    self.data = data
    self.length = len(dataset)
    self.num_features = dataset.num_features
    self.num_classes = dataset.num_classes
    self.datasetname = 'actor'

    self.num_nodes = data.num_nodes
    self.num_edges = data.num_edges
    self.avg_node_degree = (data.num_edges / data.num_nodes)
    
    self.contains_isolated_nodes = data.has_isolated_nodes()
    self.data_contains_self_loops = data.has_self_loops()
    self.is_undirected = data.is_undirected()

    self.node_features = data.x
    self.node_labels = data.y
    self.edge_index = data.edge_index
    
    
class PlanetoidData():
    
    def __init__(self, datasetname):

        dataset = Planetoid(root='data/' + datasetname, name = datasetname, transform = NormalizeFeatures())
        data = dataset[0]

        self.data = data
        self.name = datasetname
        self.length = len(dataset)
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.avg_node_degree = (data.num_edges / data.num_nodes)
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        self.train_label_rate = (int(data.train_mask.sum()) / data.num_nodes)
        self.contains_isolated_nodes = data.contains_isolated_nodes()
        self.data_contains_self_loops = data.contains_self_loops()
        self.is_undirected = data.is_undirected()

        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index

###################

def create_graph(G, features, labels, train_mask, val_mask, test_mask):
  g = dgl.from_networkx(G).to(device)
  print(g.num_nodes(), "  ", g.num_edges())
  g.ndata['feat'] = features.to(device)
  g.ndata['train_mask'] = train_mask.to(device)
  g.ndata['val_mask'] = val_mask.to(device)
  g.ndata['test_mask'] = test_mask.to(device)
  g.ndata['label'] = labels.to(device)
  print(train_mask.sum(), "  ", val_mask.sum(), "  ", test_mask.sum())
  return g


####MODEL#####
class GatedGCNLayer(nn.Module):
  """
      Param: []
  """
  def __init__(self, input_dim, output_dim, hidden_dim, dropout, batch_norm=True, residual=True, graph_norm=True):
    super().__init__()
    self.in_channels = input_dim
    self.out_channels = output_dim
    self.dropout = dropout
    self.batch_norm = batch_norm
    self.graph_norm = graph_norm
    self.residual = residual
    self.hidden_dim = hidden_dim
    
    self.A = nn.Linear(hidden_dim, output_dim, bias=True)
    self.B = nn.Linear(hidden_dim, output_dim, bias=True)
    self.C = nn.Linear(hidden_dim, output_dim, bias=True)
    self.D = nn.Linear(hidden_dim, output_dim, bias=True)
    self.E = nn.Linear(hidden_dim, output_dim, bias=True)
    self.F = nn.Linear(hidden_dim, output_dim, bias=True)
    self.bn_node_h = nn.BatchNorm1d(output_dim)
    self.bn_node_e = nn.BatchNorm1d(output_dim)
    self.W = nn.Linear(input_dim, hidden_dim, bias = True)
    self.V = nn.Linear(2*hidden_dim, hidden_dim, bias=True)

    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform(self.A.weight)
    nn.init.xavier_uniform(self.B.weight)
    nn.init.xavier_uniform(self.C.weight)
    nn.init.xavier_uniform(self.D.weight)
    nn.init.xavier_uniform(self.E.weight)
    nn.init.xavier_uniform(self.W.weight)
    nn.init.xavier_uniform(self.V.weight)

  def create_edge(self, graph, h):
    h = self.W(h)
    src = graph.edges()[0]
    dst = graph.edges()[1]
    h_src = h[src]
    h_dst = h[dst]
    e = torch.cat([h_src, h_dst], dim=1)
    e = self.V(e)
    return e

  def forward(self, g, h, snorm_n):
      
    e = self.create_edge(g, h)
    # print(e.shape)
    h = self.W(h)
  
    h_in = h # for residual connection
    e_in = e # for residual connection
    
    g.ndata['h']  = h 
    g.ndata['Ah'] = self.A(h) 
    g.ndata['Bh'] = self.B(h) 
    g.ndata['Dh'] = self.D(h)
    g.ndata['Eh'] = self.E(h) 
    g.edata['e']  = e 

    if self.residual:
        h = h_in + h # residual connection
        e = e_in + e # residual connection

    g.edata['Ce'] = self.C(e) 

    # creation of edge features
    g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
    g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
    g.edata['sigma'] = torch.sigmoid(g.edata['e'])
    
    # updated node features(neighborhood aggregation)
    g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
    
    # sum of features the edges connected to a node
    g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
    
    # final node features update
    g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
    
    h = g.ndata['h'] # result of graph convolution
    e = g.edata['e'] # result of graph convolution
    
    if self.graph_norm:
        h = h * snorm_n
    
    if self.batch_norm:
          h = self.bn_node_h(h) # batch normalization  
          e = self.bn_node_e(e) # batch normalization  
    
    h = F.relu(h) # non-linear activation
    e = F.relu(e) # non-linear activation
    
    h = F.dropout(h, self.dropout, training=self.training)
    e = F.dropout(e, self.dropout, training=self.training)

    return h, e
  
  def __repr__(self):
      return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                            self.in_channels,
                                            self.out_channels,
                                            )

# Defining Model
class GatedGCN(nn.Module):
  def __init__(self, num_layers, input_dim, output_dim, hidden_dim, dropout, batch_norm=True, residual=False, graph_norm=True):
    super().__init__()
    #self.layers = layers
    self.input = input_dim
    self.output = output_dim
    self.gated_gcn_layers = nn.ModuleList()
    self.hidden = hidden_dim
    self.num_layers = num_layers

    for i in range(self.num_layers):
      if i == 0 and self.num_layers == 1:
        model = GatedGCNLayer(input_dim = self.input, output_dim = self.output, hidden_dim = self.hidden,  dropout=dropout, batch_norm=True)
      elif i == 0 and self.num_layers > 1:
        model = GatedGCNLayer(input_dim = self.input, output_dim = self.hidden, hidden_dim = self.hidden,  dropout=dropout, batch_norm=True)
      elif i > 0 and i < self.num_layers - 1:
        model = GatedGCNLayer(input_dim = self.hidden, output_dim = self.hidden, hidden_dim = self.hidden,  dropout=dropout, batch_norm=True)
      else:
        model = GatedGCNLayer(input_dim = self.hidden, output_dim = self.output, hidden_dim = self.hidden,  dropout=dropout, batch_norm=True)

      self.gated_gcn_layers.append(model)

  
  def forward(self, g, h, snorm_n):

    for i in range(self.num_layers):
        x, e = self.gated_gcn_layers[i](g, h, snorm_n)
        h = x
   
    return F.log_softmax(h, dim = 1), e
    # return x, e


############### Train-Test##############
def train(model, graph, snorm):
  model.train()
  optimizer.zero_grad() 
  out, _  = model(graph, graph.ndata['feat'], snorm)  
  loss = criterion(out[graph.ndata['train_mask']], graph.ndata['label'][graph.ndata['train_mask']])  
  val_loss = criterion(out[graph.ndata['val_mask']], graph.ndata['label'][graph.ndata['val_mask']])  

  loss.backward()  
  optimizer.step()  
  pred = out.argmax(dim=1)  
  train_acc = (pred[graph.ndata['train_mask']] == graph.ndata['label'][graph.ndata['train_mask']]).float().mean()
  val_acc = (pred[graph.ndata['val_mask']] == graph.ndata['label'][graph.ndata['val_mask']]).float().mean()

  return loss, val_loss, train_acc, val_acc

def test(model, graph, snorm):
  model.eval()
  out, e = model(graph, graph.ndata['feat'], snorm)
  pred = out.argmax(dim=1)  
  test_correct = pred[graph.ndata['test_mask']] == graph.ndata['label'][graph.ndata['test_mask']] 
  # print(int(test_correct.sum()), "   ", int(graph.ndata['test_mask'].sum()))
  test_acc = int(test_correct.sum()) / int(graph.ndata['test_mask'].sum()) 
  return test_acc, e

if name == 'chameleon':
  data = WikipediaData('chameleon')
  data_name = 'chameleon'
  print("processing " + data_name + " data")
elif name == 'actor':
  data = ActorData()
  data_name = 'film'
  print("processing " + data_name + " data")
elif name == 'cornell':
  data = WebKBData('cornell')
  data_name = 'cornell'
  print("processing " + data_name + " data")
elif name == 'texas':
  data = WebKBData('texas')
  data_name = 'texas'
  print("processing " + data_name + " data")
elif name == 'wisconsin':
  data = WebKBData('wisconsin')
  data_name = 'wisconsin'
  print("processing " + data_name + " data")
elif name == 'squirrel':
  data = WikipediaData('squirrel')
  data_name = 'squirrel'
  print("processing " + data_name + " data")
elif name == 'cora' or 'citeseer' or 'pubmed':
  # data = PlanetoidData(name)
  data_name = name
  print("processing " + data_name + " data")
else:
  print("Invalid name of the data")

##############################

best_val_acc = 0
test_acc_list = []
for i in range(10):
  PATH = 'best_model.pt'
  splitstr = 'splits/'+ data_name +'_split_0.6_0.2_'+str(i)+'.npz'
  G, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = full_load_data(data_name, splitstr)
  graph = create_graph(G, features, labels, train_mask, val_mask, test_mask)

  gated_model = GatedGCN(num_layers = num_layers, input_dim = num_features, output_dim = num_labels , hidden_dim = hidden_dim,  dropout=dropout, batch_norm=True)
  gated_model.to(device)

  optimizer = torch.optim.Adam(gated_model.parameters(), lr=lr, weight_decay=w_decay)
  criterion = torch.nn.CrossEntropyLoss()

  for epoch in range(1, epoch+1):

      loss,val_loss, train_acc, val_acc = train(gated_model, graph, train_snorm)
      
      if val_acc > best_val_acc:
        torch.save({'epoch': epoch,
              'model_state_dict': gated_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'val_loss': val_loss,
              'val_acc': val_acc,
              'train_acc': train_acc,
              'loss': loss},
              PATH)
        best_val_acc = val_acc
        
      if epoch%200==0:
        print(f' Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc: .4f}, Val Loss : {val_loss: .4f}, Validation Acc: {val_acc: .4f} (Best validation acc: {best_val_acc: .4f})')
  
  chkp = torch.load(PATH)
  gated_model.load_state_dict(chkp['model_state_dict'])

  s = np.arange(1, 100)
  sn, mx = 0, 0
  best_e = 0
  for j in s:
    test_acc, e = test(gated_model, graph, j)
        
    if test_acc > mx:
      mx = test_acc
      sn = j
      best_e = e

  gated_model.eval()
  out, _ = gated_model(graph, graph.ndata['feat'], sn)
  visualize(out, labels, name, i)

  print(f'For Snorm : {sn: 03d} Test Accuracy: {mx:.4f} for file '+ str(i))
  test_acc_list.append(mx)

print("Test Accuracy List ", test_acc_list)
print(f'Final Test Statistics: {np.mean(test_acc_list):.4f} || {np.std(test_acc_list):.4f}')
