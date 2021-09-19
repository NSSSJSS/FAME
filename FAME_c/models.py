import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

# from fastrp import adj_matrix_weight_merge, fastrp_projection, fastrp_merge
from fastrp import adj_matrix_weight_merge, fastrp_projection, fastrp_merge


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        # 线性神经网络 可以训练的权重矩阵 W 大小为 nfeat * nclass
        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class FAME(nn.Module):
    def __init__(self, nfeat, nclass, num_line):
        super(FAME, self).__init__()

        # Alibaba
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(4, 1), requires_grad=True)
        self.weight_a = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)
        torch.nn.init.uniform_(self.weight_a,a = 0,b = 0.1)

        # Aminer
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(3, 1), requires_grad=True)
        # self.weight_a = torch.nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        # torch.nn.init.uniform_(self.weight_a, a=0, b=0.1)

        # IMDB
        # self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # self.weight_a = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        # torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)
        # torch.nn.init.uniform_(self.weight_a, a=0, b=0.1)

    def forward(self, feature, A, conf):
        final_adj_matrix = adj_matrix_weight_merge(A, self.weight_b)
        U_list = fastrp_projection(A,
                                   feature,
                                   final_adj_matrix,
                                   q=conf['q'],
                                   dim=conf['dim'],
                                   projection_method=conf['projection_method'],
                                   input_matrix=conf['input_matrix'],
                                   feature_similarity=conf['feature_similarity']
                                   )
        U = fastrp_merge(U_list, self.weight_a, normalization=False)
        return U


def get_model(model_opt, nfeat, nclass, A, nhid=0, dropout=0, cuda=True):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass)
    elif model_opt == "FAME":
        num = A[0][0].tocoo().shape[0]
        model = FAME(nfeat=nfeat,
                     nclass=nclass, num_line=A[0][0].tocoo().shape[0])
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model
