import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed, load_our_data
from models import get_model
from metrics import accuracy, f1
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from sklearn.metrics import f1_score

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

# adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)
# args.dataset = 'imdb_1_10'
args.dataset = 'Aminer_10k_4class'
# args.dataset = 'small_alibaba_1_10'
adj, features, labels, idx_train, idx_val, idx_test = load_our_data(args.dataset, args.cuda)

model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)
# model 返回的是模型，如果是SGC的话，就是一个简单的线性权重矩阵的分类器

# degree 表示SGC的层数
if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
print("{:.4f}s".format(precompute_time))

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    # 这是一个简单的分类器函数

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)

def test_f1(model,test_features, test_labels):
    model.eval()
    # f1_ma = f1_score(model(test_features).detach().numpy(), test_labels.detach().numpy(), average='macro')
    # f1_mi = f1_score(model(test_features), test_labels, average='micro')
    f1_mi, f1_ma = f1(model(test_features), test_labels)
    return f1_ma, f1_mi

if args.model == "SGC":
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                     args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])
    f1_ma, f1_mi = test_f1(model, features[idx_test], labels[idx_test])
print('Test F1-ma: {:.3f}, F1-mi: {:.3f}'.format(f1_ma,f1_mi))
print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
