import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from models_nc import LogReg
import torch.nn as nn
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
from sklearn.metrics import classification_report
import scipy.io as sio
import pickle as pkl
import itertools
import warnings

warnings.filterwarnings('ignore')


def load_data(dataset, datasetfile_type, device):

    if datasetfile_type == 'mat':
        data = sio.loadmat('data/{}.mat'.format(dataset))
    else:
        data = pkl.load(open('data/{}.pkl'.format(dataset), "rb"))
    try:
        labels = data['label'] # 某一类节点的label
    except:
        labels = data['labelmat'] # 某一类节点的label
    N = labels.shape[0]

    idx_train = data['train_idx'].ravel()
    try:
        idx_val = data['valid_idx'].ravel()
    except:
        idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    # return labels, idx_train.astype(np.int32), idx_val.astype(np.int32), idx_test.astype(np.int32)
    return labels, idx_train.astype(np.int32)-1, idx_val.astype(np.int32)-1, idx_test.astype(np.int32)-1

def node_classification_evaluate(model, feature, A, conf, file_name, file_type, device, isTest=True):
    embeds = model(feature, A, conf)

    labels, idx_train, idx_val, idx_test = load_data(file_name, file_type, device)
    
    try:
        labels = labels.todense()
    except:
        pass
    labels = labels.astype(np.int16)
    embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    
    
    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]
    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = [] ##

    t0 = []
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    t6 = []
    t7 = []
    t8 = []
    t9 = []
    t10 = []

    for _ in range(1):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.05}, {'params': log.parameters()}], lr=0.005, weight_decay=0.0)
        log.to(device)

        val_accs = []; test_accs = []
        val_micro_f1s = []; test_micro_f1s = []
        val_macro_f1s = []; test_macro_f1s = []

        # IMDB 73 Aminer 300 Alibaba 60
        for iter_ in range(60):
            embeds = model(feature, A, conf)
            embeds = torch.FloatTensor(embeds[np.newaxis]).to(device)
            train_embs = embeds[0, idx_train]
            val_embs = embeds[0, idx_val]
            test_embs = embeds[0, idx_test]

            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            logits_val = log(val_embs)
            preds = torch.argmax(logits_val, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

            print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_+1, loss.item(), val_acc, val_f1_macro, val_f1_micro))

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits_test = log(test_embs)
            preds = torch.argmax(logits_test, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter]) ###

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        if _ == 9:
            ma_f1s = np.mean(macro_f1s)
            sa_f1s = np.std(macro_f1s)
            mi_f1s = np.mean(micro_f1s)
            si_f1s = np.std(micro_f1s)
            breakpoint_i = 1
            print("ma_f1s: {:.4f}".format(ma_f1s))

    if isTest:
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                np.std(macro_f1s),
                                                                                                np.mean(micro_f1s),
                                                                                                np.std(micro_f1s)))
    else:
        return np.mean(macro_f1s), np.mean(micro_f1s)

    test_embs = np.array(test_embs.cpu().detach().numpy())
    test_lbls = np.array(test_lbls.cpu().detach().numpy())


    return np.mean(macro_f1s), np.mean(micro_f1s)

def run_similarity_search(test_embs, test_lbls):
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4)))

    st = ','.join(st)
    print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))


def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    for i in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)

        s1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)

    s1 = sum(NMI_list) / len(NMI_list)

    print('\t[Clustering] NMI: {:.4f}'.format(s1))