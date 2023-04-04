import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data
import torch_geometric.utils as pyg_utils

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mtick
import matplotlib.cm as cm

from sklearn.manifold import TSNE
import math
import numpy as np
import torch
import os
import torch.nn.functional as F
import random 
import seaborn as sns
import networkx as nx
import scipy.sparse as sp
import pickle as pkl
import sys
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

#adapted from geom-gcn

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def full_load_citation(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def full_load_data(dataset_name, splits_file_path=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = full_load_citation(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}


        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        # adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)

    # g = adj
  
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    
    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    # g = sys_normalized_adjacency(g)
    # g = sparse_mx_to_torch_sparse_tensor(g)

    return G, features, labels, train_mask, val_mask, test_mask, num_features, num_labels

# visuals of embedding
def visualize(feat_map, color, name, hidden_layers):
    z = TSNE(n_components=2).fit_transform(feat_map.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig(os.getcwd() + "/visuals/" + name + "_" + str(hidden_layers) + "_layers_embedding.png")
    plt.clf()
    pl

# loss function
def loss_fn(pred, label):

    # return F.nll_loss(pred, label)
    return F.cross_entropy(pred, label)

def visualize(h, color, data_name, idx):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=40, c=color, cmap="Set1")
    plt.savefig(os.getcwd() + "/visuals/"+ data_name +  "_node_emb_" + str(idx) + ".png")
    plt.clf()

def edge_gate_plot(gate_val, edge_types, idx):
    num_edges = edge_types.shape[0]
    edge_id = [i for i in range(num_edges)]
    c = ['red' if edge_types[i] == 0 else 'green' for i in range(num_edges)]
    plt.bar(edge_id, gate_val, color=c)
    plt.savefig(os.getcwd() + "/edge_gates/" + "_" + str(idx) + "_.png")
    plt.clf()

def circular_bar_plot(gate_val, edge_types, idx):
    # print(edge_types)
    # print()
    # print(gate_val)
    num_edges = edge_types.shape[0]
    
    plt.figure(figsize=(15,10))

    # plot polar axis
    ax = plt.subplot(111, polar=True)

    # remove grid
    plt.axis('off')

    # Set the coordinates limits
    upperLimit = 100
    lowerLimit = 30

    # Compute max and min in the dataset
    max = gate_val.max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * gate_val + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / num_edges

    c = ['blue' if edge_types[i] == 0 else 'cyan' for i in range(num_edges)]    
    # Compute the angle each bar is centered on:
    indexes = list(range(1, num_edges+1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=1, 
        color=c,
        edgecolor='white')
    ax.legend(['Intra class edges', 'Inter class edges'])
    
    plt.savefig(os.getcwd() + "/edge_gates/" + "_" + str(idx) + "_.png")
    plt.clf()

def heat_map_generation(adj, edge_types, idx):
    # print(adj[:75, :75])
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    hm1 = sns.heatmap(data = adj , annot = True, annot_kws={'size': 2}, linewidths = 0.1, cmap="Blues", cbar_kws={"shrink": .8})
    # ax[0].set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [r'$0$', r'$1$', r'$2$', r'$3$', r'$4$', r'$5$', r'$6$', r'$7$', r'$8$', r'$9$'], fontsize = 12)
    # ax[0].set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [r'$0$', r'$1$', r'$2$', r'$3$', r'$4$', r'$5$', r'$6$', r'$7$', r'$8$', r'$9$'], fontsize = 12)
    # ax[0].set_title('Heat Map', fontdict = {'fontsize': 20})
    plt.savefig(os.getcwd() + "/heat_map/" + str(idx) + "_.png")
    # plt.show()
    plt.clf()




 # gate_val = e.norm(dim = 1, p = 2).cpu().detach().numpy()
  # print(gate_val)
  # src = graph.edges()[0]
  # dst = graph.edges()[1]
  # src_labels = graph.ndata['label'][src]
  # dst_labels = graph.ndata['label'][dst]
  # edge_types = (src_labels == dst_labels).cpu().int()
  # edge_types = edge_types.detach().numpy()
  # print(edge_types)
  # intra_class_edges = []
  # inter_class_edges = []
  # for k in range(graph.num_edges()):
  #   if edge_types[k] == 1:
  #     intra_class_edges.append(gate_val[k])
  #   else:
  #     inter_class_edges.append(gate_val[k]) 
  # print("intra ", np.mean(np.array(intra_class_edges)), "   ", np.std(np.array(intra_class_edges)))
  # print("inter ", np.mean(np.array(inter_class_edges)), "   ", np.std(np.array(inter_class_edges)))
  # edge_gate_plot(gate_val, edge_types, i)
  # circular_bar_plot(gate_val, edge_types, i)
  # heat_map_generation(adj, edge_types, i)