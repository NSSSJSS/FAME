import numpy as np
import scipy
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from normalization import fetch_normalization, row_normalize
from time import perf_counter
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from fastrp import adj_matrix_weight_merge, fastrp_projection, fastrp_merge


def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = edge_key.split('_')[0]
        y = edge_key.split('_')[1]
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G


def load_training_data(f_name):
    # print('We are loading training data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            # print(words)
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total training nodes: ' + str(len(all_nodes)))
    # print('Finish loading training data')
    return edge_data_by_type


def load_testing_data(f_name):
    # print('We are loading testing data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    # print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type


def load_node_type(f_name):
    # print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_link_prediction_data(dataset_str="Aminer_1_13", cuda=True):
    """
    Load our Networks Datasets
    Avoid awful code
    """
    mat = loadmat('data/LP/' + dataset_str)

    try:
        train = mat['train']
    except:
        try:
            train = mat['train_full']
        except:
            train = mat['edges']
    ###########
    try:
        feature = mat['full_feature']
    except:
        try:
            feature = mat['feature']
        except:
            try:
                feature = mat['features']
            except:
                feature = mat['attribute']
    try:
        feature = csr_matrix(feature)
    except:
        pass

    # edges to adj
    # for i in range(4):
    #     a = train[i][0]
    #     a = a.A
    #     a = a[:4025, :4025]
    #     a = np.mat(a)
    #     train[i][0] = a

    row = train[0][0].shape[0]
    # row = 4025
    adj = csr_matrix((row, row))
    for t in train:
        # adj += t[0]
        adj += t[0] + t[0].T
    # adj += mat['IUI_buy'] + mat['IUI_buy'].T

    print('{} node number: {}'.format(dataset_str, row))
    try:
        feature = feature.astype(np.int16)
    except:
        pass
    try:
        features = torch.FloatTensor(np.array(feature.todense())).float()
    except:
        features = torch.FloatTensor(np.array(feature)).float()
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    if cuda:
        features = features.cuda()
        adj = adj.cuda()

    return adj, features, row


def load_our_data(dataset_str="IMDB", cuda=True):
    """
    Load our Networks Datasets
    Avoid awful code
    """
    data = loadmat('data/' + dataset_str + '.mat')
    # label
    try:
        labels = data['label']  # 某一类节点的label
    except:
        labels = data['labelmat']  # 某一类节点的label
    N = labels.shape[0]
    try:
        labels = labels.todense()
    except:
        pass

    # idx train valid test
    idx_train = data['train_idx'].ravel()
    try:
        idx_val = data['valid_idx'].ravel()
    except:
        idx_val = data['val_idx'].ravel()
    # idx_test = data['train_idx'].ravel()
    idx_test = data['test_idx'].ravel()
    # idx_train = np.concatenate((idx_train, idx_test))
    # node features
    try:
        node_features = data['feature']
    except:
        try:
            node_features = data['full_feature'].toarray()
        except:
            node_features = data['features']
    features = csr_matrix(node_features)

    # edges to adj
    if dataset_str == 'small_alibaba_1_10':
        num_nodes = data['IUI_buy'].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        adj = data['IUI_buy'] + data['IUI_cart'] + data["IUI_clk"] + data['IUI_collect']
    elif dataset_str == 'Aminer_10k_4class':
        num_nodes = data['PAP'].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        adj = data['PAP'] + data['PCP'] + data["PTP"]
        idx_test = idx_test - 1
        idx_train = idx_train - 1
        idx_val = idx_val - 1
        features = features[0:10000]

    else:
        edges = data['edges'][0].tolist()
        num_nodes = edges[0].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        for edge in edges:
            adj += edge

    print('{} node number: {}'.format(dataset_str, num_nodes))
    try:
        features = features.astype(np.int16)
    except:
        pass
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train.astype(np.int16))
    idx_val = torch.LongTensor(idx_val.astype(np.int16))
    idx_test = torch.LongTensor(idx_test.astype(np.int16))

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
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
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def sgc_precompute(features, adj, degree):
    # sgc 预计算，先乘以节点特征矩阵可以降低矩阵维度，降低运算时间复杂度
    # features
    t = perf_counter()  # 记录时间
    for i in range(degree):
        # 做degree次乘积
        features = torch.spmm(adj, features)  # 稀疏矩阵乘积
    precompute_time = perf_counter() - t
    return features, precompute_time

def FAME_precompute(feature, A, weight_a, weight_b, conf):
# def FAME_precompute(feature, A, weight_a1, weight_a2, weight_a3, weight_b1, weight_b2, weight_b3, conf):
    t = perf_counter()
    # final_adj_matrix = adj_matrix_weight_merge(A, weight_b1, weight_b2, weight_b3)
    final_adj_matrix = adj_matrix_weight_merge(A, weight_b)
    U_list = fastrp_projection(A,
                               feature,
                               final_adj_matrix,
                               q=conf['q'],
                               dim=conf['dim'],
                               projection_method=conf['projection_method'],
                               input_matrix=conf['input_matrix'],
                               feature_similarity=conf['feature_similarity']
                               )
    # U = fastrp_merge(U_list, weight_a1, weight_a2, weight_a3, normalization=False, q=3)
    U = fastrp_merge(U_list, weight_a, normalization=False, q=3)
    precompute_time = perf_counter() - t
    return U, precompute_time


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']


def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features - features.mean(dim=0)) / features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index


def load_training_data(f_name):
    # print('We are loading training data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            # print(words)
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total training nodes: ' + str(len(all_nodes)))
    # print('Finish loading training data')
    return edge_data_by_type


def load_testing_data(f_name):
    # print('We are loading testing data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            # words = line[:-1].split('\t')
            words = line[:-1].split()
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    # print('Finish loading testing data')
    return true_edge_data_by_type, false_edge_data_by_type

def ACC(a,b):
    t=0
    for i in range(a.size):
        if(a[i]==b[i]):
            t+=1
    return t/a.size

def my_Kmeans(x, y, k, time=10, return_NMI=False):

    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    estimator = KMeans(n_clusters=k)
    ACC_list = []
    SIL_list = []
    if time:
        for i in range(time):
            estimator.fit(x)
            y_pred = estimator.predict(x, y)
            label = estimator.labels_
            score = ACC(y_pred, y)
            ACC_list.append(score)
            s2 = silhouette_score(x,y_pred)
            SIL_list.append(s2)
        score = sum(ACC_list) / len(ACC_list)
        s2 = sum(SIL_list) / len(SIL_list)
        print('ACC (10 avg): {:.4f} {:.4f} , SIL (10avg): {:.4f} {:.4f}'.format(score, np.std(ACC_list,ddof=1), s2, np.std(SIL_list,ddof=1)))

    else:
        estimator.fit(x, y)
        y_pred = estimator.predict(x)
        score = normalized_mutual_info_score(y, y_pred)
        print("NMI on all label data: {:.5f}".format(score))
    if return_NMI:
        return score, s2

def Getyy(f_name):
    mat_file = open(f_name, 'rb')
    file = scipy.io.loadmat(mat_file)
    try:
        label = file['label'].toarray()
    except:
        label = file['label']
    # yy = np.zeros(shape=(len(label), 1))
    # for i in range(len(label)):
    #     for j in range(len(label[0])):
    #         if (label[i][j] == 1):
    #             yy[i] = j + 1
    # yy = np.squeeze(yy)
    # if (f_name == 'data/small_alibaba_1_10/small_alibaba_1_10.mat'):
    #     return label
    # else:
    #     return yy
    return label
