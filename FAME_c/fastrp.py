import csv
import itertools
import math
import time
import logging
import sys
import os
import random
import warnings
import pandas as pd
import numpy as np
import scipy
import optuna
import sklearn.preprocessing as pp
import torch


from tqdm import tqdm_notebook as tqdm
from collections import Counter, defaultdict

from pathlib import Path
from sklearn import random_projection
from sklearn.preprocessing import normalize, scale, MultiLabelBinarizer
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, diags, spdiags, vstack, hstack

# from utils import sparse_mx_to_torch_sparse_tensor


# projection method: choose from Gaussian and Sparse
# input matrix: choose from adjacency and transition matrix
# alpha adjusts the weighting of nodes according to their degree
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def Tocoo(A):
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def adj_matrix_weight_merge(A, adj_weight):

    N = A[0][0].shape[0]  # 矩阵的行数
    temp = coo_matrix((N, N))  # 初始化稀疏矩阵
    temp = Tocoo(temp) # 转化为稀疏张量
    # 将邻接矩阵数据转化为稀疏张量

    # Alibaba
    a = Tocoo(A[0][0].tocoo())
    b = Tocoo(A[1][0].tocoo())
    c = Tocoo(A[2][0].tocoo())
    d = Tocoo(A[3][0].tocoo())
    A_t = torch.stack([a, b, c, d], dim=2).to_dense()

    # Aminer
    # a = Tocoo(A[0][0].tocoo())
    # b = Tocoo(A[0][1].tocoo())
    # c = Tocoo(A[0][2].tocoo())
    # A_t = torch.stack([a, b, c], dim=2).to_dense()

    # IMDB
    # a = Tocoo(A[0][0].tocoo())
    # b = Tocoo(A[0][2].tocoo())
    # A_t = torch.stack([a, b], dim=2).to_dense()

    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)
    temp = temp.to_sparse()  # 最终结果转换回稀疏张量

    return temp + temp.transpose(0, 1)




def torch_sparse_matrix_to_scipy_sparse(M):
    indices = M._indices()
    value = M._values()
    shape = M.size()

    return coo_matrix((value, indices),shape).tocsr()


def fastrp_projection(train, feature, final_adj_matrix, q=1, dim=128, projection_method='gaussian',
                      input_matrix='adj', alpha=None, s=1, threshold=0.95, gama=1, feature_similarity=True):  # 随机投影
    assert input_matrix == 'adj' or input_matrix == 'trans'
    assert projection_method == 'gaussian' or projection_method == 'sparse'
    M = final_adj_matrix  # 空
    if feature_similarity == True:
        feature = pp.normalize(feature, axis=1).T
    # Gaussian projection matrix
    if projection_method == 'gaussian':
        transformer = random_projection.GaussianRandomProjection(n_components=dim, random_state=7)
    # Sparse projection matrix
    else:
        transformer = random_projection.SparseRandomProjection(n_components=dim, random_state=7)
    Y = transformer.fit(feature)
    # Construct the inverse of the degree matrix
    if input_matrix != 'adj':
        # @liuzhijun 为了计算新邻接矩阵的度矩阵，需要将新邻接矩阵M转换回scipy.sparse 矩阵
        temp_M = torch_sparse_matrix_to_scipy_sparse(M)
        rowsum = temp_M.sum(axis=1)
        colsum = temp_M.sum(axis=0).T
        rowsum = np.squeeze(np.asarray(rowsum + colsum)) ** -1
        rowsum[np.isinf(rowsum)] = 1
        D_inv = diags(rowsum)  # 度矩阵
    cur_U = transformer.transform(feature)
    if feature_similarity == True:
        cur_U = feature.T @ cur_U  # 属性相似性矩阵
    D_inv = sparse_mx_to_torch_sparse_tensor(D_inv)
    cur_U = sparse_mx_to_torch_sparse_tensor(csr_matrix(cur_U))
    cur_U = torch.sparse.mm(M, cur_U.to_dense())
    if input_matrix != 'adj':
        # normalization
        torch.sparse.mm(D_inv, cur_U)
    U_list = [cur_U]
    for j in range(1, q):
        torch.sparse.mm(M, cur_U)
        if input_matrix != 'adj':
            # normalization
            torch.sparse.mm(D_inv, cur_U)

        U_list.append(cur_U)  # AXR,AAXR,AAAXR

    return U_list


# When weights is None, concatenate instead of linearly combines the embeddings from different powers of A
def fastrp_merge(U_list, weight, normalization=False):

    U = torch.zeros_like(U_list[0])

    # Alibaba
    a = U_list[0]
    b = U_list[1]
    c = U_list[2]
    U_t = torch.stack([a, b, c], dim=2)

    # Aminer
    # a = U_list[0]
    # U_t = torch.stack([a], dim=2)

    # IMDB
    # a = U_list[0]
    # b = U_list[1]
    # U_t = torch.stack([a, b], dim=2)

    U_merged = torch.matmul(U_t, weight)
    U_merged = torch.squeeze(U_merged)

    return U_merged
