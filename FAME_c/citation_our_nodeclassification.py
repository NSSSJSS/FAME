import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from scipy.io import loadmat
from utils import load_citation, sgc_precompute, set_seed, load_our_data, FAME_precompute, Getyy, my_Kmeans
from models import get_model
from metrics import accuracy, f1
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from sklearn.metrics import f1_score
from node_classification_evaluate import *
from scipy.sparse import csc_matrix
torch.autograd.set_detect_anomaly(True)

conf = {
        'projection_method': 'sparse',  # sparse gaussian
        'input_matrix': 'trans',
        'normalization': True,
        'q': 3,
        'dim': 128,
        'edge_type': [0, 1, 2, 3],
        's': 1,
        'trials': 100,
        'feature_similarity': True
    }


# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "FAME":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented
# setting random seeds
set_seed(args.seed, args.cuda)


# IMDB
# args.dataset = 'imdb_1_10_4'
# eval_name = r'data/imdb_1_10_4'
# net_path = r"data/imdb_1_10_4.mat"
# savepath = r'data/imdb_embedding_1_10'
# eval_name = r'imdb_1_10_4'

# Aminer
# args.dataset = 'Aminer_10k_4class'
# eval_name = r'Aminer_10k_4class'
# net_path = r'data/Aminer_1_13/Aminer_10k_4class.mat'
# savepath = r'embedding/Aminer_10k_4class_aminer_embedding_'
# eval_name = r'Aminer_10k_4class'

# alibaba
args.dataset = 'small_alibaba_1_10'
eval_name = r'small_alibaba_1_10'
net_path = r'data/small_alibaba_1_10/small_alibaba_1_10.mat'
savepath = r'data/alibaba_embedding_'
eval_name = r'small_alibaba_1_10'


order_range = 3
number_edge_type = 4
feature_similarity = False

mat = loadmat(net_path)

try:
    train = mat['train']+mat['valid']+mat['test']
except:
    try:
        train = mat['train_full']+mat['valid_full']+mat['test_full']
    except:
        train = mat['edges']

try:
    feature = mat['full_feature']
except:
    try:
        feature = mat['node_feature']
    except:
        feature = mat['features']
feature = csc_matrix(feature) if type(feature) != csc_matrix else feature

if net_path == r'data/amazon':
    A = train
elif args.dataset == 'Aminer_10k_4class':
    A = [[mat['PAP'], mat['PCP'], mat['PTP'] ]]
elif net_path == 'imdb_1_10.mat':
    A = train[0]
else:
    A = train


adj, features, labels, idx_train, idx_val, idx_test = load_our_data(args.dataset, args.cuda)

model = get_model(args.model, features.size(1), labels.max().item()+1, A, args.hidden, args.dropout, args.cuda)

starttime=time.time()

f1_ma, f1_mi = node_classification_evaluate(model, feature, A, conf, eval_name, file_type='mat', device=torch.device('cpu'))
endtime=time.time()

embeds = model(feature, A, conf)
embeds = embeds.data.numpy()

# IMDB
xx = embeds[:4658]
yy = Getyy('data/IMDB/imdb_1_10.mat')
my_Kmeans(xx, yy, 4)

# Aminer
# xx = embeds[:10000]
# yy = Getyy('data/Aminer_10k_4class.mat')
# my_Kmeans(xx, yy, 5)

# Alibaba
# xx = embeds[:4025]
# yy = Getyy('data/small_alibaba_1_10.mat')
# my_Kmeans(xx, yy, 3)



print("{:.4f}s".format(endtime-starttime))
def train_regression(model, A, conf,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features, A, conf)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features, A, conf)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

def test_regression(model, test_features, test_labels, A, conf):
    model.eval()
    return accuracy(model(test_features, A, conf), test_labels)

def test_f1(model,test_features, test_labels, A, conf):
    model.eval()
    # f1_ma = f1_score(model(test_features).detach().numpy(), test_labels.detach().numpy(), average='macro')
    # f1_mi = f1_score(model(test_features), test_labels, average='micro')
    f1_mi, f1_ma = f1(model(test_features, A, conf), test_labels)
    return f1_ma, f1_mi

print('Test F1-ma: {:.10f}, F1-mi: {:.10f}'.format(f1_ma,f1_mi))