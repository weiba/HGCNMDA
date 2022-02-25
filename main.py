import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch import nn, optim
from model import Model
from trainData import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
from numpy import *
from utils import *
import copy
import torch


class Config(object):
    def __init__(self):
        #self.validation = 5
        self.epoch = 100
        #self.epoch = 300
        #self.alpha = 0.1
        self.alpha = 0.4

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
    def forward(self, one_index, zero_index, target,target_gd, target_gm, input,input_gd,input_gm):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        loss_gd = loss(input_gd, target_gd)
        loss_gm = loss(input_gm, target_gm)

        return (1 - opt.alpha) * loss_sum[one_index].sum() + opt.alpha * loss_sum[zero_index].sum()+loss_gd.sum()+loss_gm.sum()

class Sizes(object):
    def __init__(self, dataset):
        self.mdg = dataset['dd']['data'].size(0) + dataset['mm']['data'].size(0) + dataset['gg']['data'].size(0)
        self.d = dataset['dd']['data'].size(0)
        self.m = dataset['mm']['data'].size(0)
        self.g = dataset['gg']['data'].size(0)
        self.fg = 256
        self.fd = 128
        self.k = 64

def train(model, train_data, optimizer, opt):

    model.train()
    regression_crit = Myloss()
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000, 3000], gamma=0.5)
    one_index = train_data[5][0].cuda().t().tolist()
    zero_index = train_data[5][1].cuda().t().tolist()

    def train_epoch():
        model.zero_grad()
        score, score_gd, score_gm = model(train_data)
        loss = regression_crit(one_index, zero_index, train_data[7].cuda(), train_data[3]['data'].cuda(), train_data[4]['data'].cuda(), score, score_gd, score_gm)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        return loss, score

    for epoch in range(1, opt.epoch + 1):
        train_reg_loss, predict = train_epoch()
    return predict

if __name__ == "__main__":
    
    opt = Config()

    # D = np.genfromtxt(r"/home/chezicheng/program/data/md_delete.txt")
    # R11 = np.genfromtxt(r"/home/chezicheng/program/data/mm_delete.txt")
    # R22 = np.genfromtxt(r"/home/chezicheng/program/data/dd_delete.txt")
    # gg = np.genfromtxt(r"/home/chezicheng/program/data/gg.txt")
    # dg = np.genfromtxt(r"/home/chezicheng/program/data/dg_delete.txt")
    # mg = np.genfromtxt(r"/home/chezicheng/program/data/mg_delete.txt")

    D = np.genfromtxt(r"./data/md_delete.txt")
    R11 = np.genfromtxt(r"./data/mm_delete.txt")
    R22 = np.genfromtxt(r"./data/dd_delete.txt")
    gg = np.genfromtxt(r"./data/gg.txt")
    dg = np.genfromtxt(r"./data/dg_delete.txt")
    mg = np.genfromtxt(r"./data/mg_delete.txt")

    # dg = gd.T
    # mg = gm.T
    [row, col] = np.shape(D)

    indexn = np.argwhere(D == 0)
    Index_zeroRow = indexn[:, 0]
    Index_zeroCol = indexn[:, 1]
    indexp = np.argwhere(D == 1)
    Index_PositiveRow = indexp[:, 0]
    Index_PositiveCol = indexp[:, 1]
    totalassociation = np.size(Index_PositiveRow)
    fold = int(totalassociation / 5)
    zero_length = np.size(Index_zeroRow)
    cv_num = 5
    n = 10

    varauc = []
    AAuc_list1 = []

    varf1_score = []
    f1_score_list1 = []
    varprecision = []
    precision_list1 = []
    varrecall = []
    recall_list1 = []
    varaupr = []
    aupr_list1 = []

    for time in range(1, n + 1):
        Auc_per = []

        f1_score_per = []
        precision_per = []
        recall_per = []
        aupr_per = []
        p = np.random.permutation(totalassociation)
        for f in range(1, cv_num + 1):
            print("cross_validation:", '%01d' % (f))
            if f == cv_num:
                testset = p[((f - 1) * fold): totalassociation + 1]
            else:
                testset = p[((f - 1) * fold): f * fold]

            ######train pos and neg
            # all_f = np.random.permutation(np.size(Index_zeroRow))
            # train_p = list(set(p).difference(set(testset)))
            # train_f = all_f[0:len(train_p)]
            # test_p = list(testset)
            # difference_set_f = list(set(all_f).difference(set(train_f)))
            # # test_f = difference_set_f[0:len(test_p)]
            # test_f = difference_set_f

            ######test pos and neg
            all_f = np.random.permutation(np.size(Index_zeroRow))
            test_p = list(testset)
            # test_f = difference_set_f[0:len(test_p)]
            test_f = all_f[0:len(test_p)]
            difference_set_f = list(set(all_f).difference(set(test_f)))
            train_p = list(set(p).difference(set(testset)))
            # train_f = all_f[0:len(train_p)]
            train_f = difference_set_f
            # test_f = difference_set_f

            X = copy.deepcopy(D)
            Xn = copy.deepcopy(X)
            #test_length = len(test_p)
            zero_index = []
            for ii in range(len(train_f)):
                zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])
            true_list = zeros((len(test_p) + len(test_f), 1))
            for ii in range(len(test_p)):
                Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
                true_list[ii, 0] = 1
            D1 = copy.deepcopy(Xn)
            dataset = prepare_data(opt, D, D1, R11, R22, gg, dg, mg, zero_index)
            sizes = Sizes(dataset)
            train_data = Dataset(opt, dataset)

            model = Model(sizes)
            model = model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            gc.collect()
            t.cuda.empty_cache()
            predict = train(model, train_data[f], optimizer, opt)
            predict = predict.data.cpu().numpy()
            test_predict = create_resultlist(predict, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,
                                             Index_zeroCol, len(test_p), zero_length, test_f)
            label = true_list
            test_auc = roc_auc_score(label, test_predict)
            Auc_per.append(test_auc)
            print("//////////每一次auc: " + str(test_auc))
            varauc.append(test_auc)

            ####
            max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),torch.from_numpy(test_predict).float())
            f1_score_per.append(max_f1_score)
            print("//////////max_f1_score:", max_f1_score)
            # acc = accuracy_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),threshold)
            # print("acc:", acc)
            precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),threshold)
            precision_per.append(precision)
            print("//////////precision:", precision)
            recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),threshold)
            recall_per.append(recall)
            print("//////////recall:", recall)
            # mcc_score = mcc_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),threshold)
            # print("mcc_score:", mcc_score)
            pr, re, thresholds = precision_recall_curve(label, test_predict)
            aupr = auc(re, pr)
            aupr_per.append(aupr)
            print("//////////aupr", aupr)

            varf1_score.append(max_f1_score)
            varprecision.append(precision)
            varrecall.append(recall)
            varaupr.append(aupr)


            # average_precision = average_precision_score(label, test_predict)
            # print('Average precision-recall score: {0:0.2f}'.format(average_precision))
            #
            # test_predict[test_predict >= 0.5] = 1
            # test_predict[test_predict < 0.5] = 0
            # precision = metrics.precision_score(label, test_predict)
            # print("precision", precision)
            #
            # r = metrics.recall_score(label, test_predict)
            # print("recall", r)
            #
            # print("f1-score", metrics.f1_score(label, test_predict))


        AAuc_list1.append(np.mean(Auc_per))

        f1_score_list1.append(np.mean(f1_score_per))
        precision_list1.append(np.mean(precision_per))
        recall_list1.append(np.mean(recall_per))
        aupr_list1.append(np.mean(aupr_per))

        print("//////////Aucaverage: " + str(AAuc_list1))
        print("//////////f1_scoreaverage: " + str(f1_score_list1))
        print("//////////precisionaverage: " + str(precision_list1))
        print("//////////recallaverage: " + str(recall_list1))
        print("//////////aupraverage: " + str(aupr_list1))

    vauc = np.var(varauc)

    vf1_score = np.var(varf1_score)
    vprecision = np.var(varprecision)
    vrecall = np.var(varrecall)
    vaupr = np.var(varaupr)

    print("sumauc = %f±%f\n" % (float(np.mean(AAuc_list1)), vauc))

    print("sumf1_score = %f±%f\n" % (float(np.mean(f1_score_list1)), vf1_score))
    print("sumprecision = %f±%f\n" % (float(np.mean(precision_list1)), vprecision))
    print("sumrecall = %f±%f\n" % (float(np.mean(recall_list1)), vrecall))
    print("sumaupr = %f±%f\n" % (float(np.mean(aupr_list1)), vaupr))

