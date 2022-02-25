import os
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from torch import nn, optim
from model import Model
from trainData import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import *
import copy
from sklearn.metrics import precision_recall_curve, auc


class Config(object):
    def __init__(self):
        #self.validation = 5
        self.epoch = 100
        #self.epoch = 100
        #self.alpha = 0.1
        self.alpha = 0.5


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
    def forward(self, one_index, zero_index, target, target_gd, target_gm, input, input_gd, input_gm):
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
    one_index = train_data[5][0].cuda().t().tolist()
    zero_index = train_data[5][1].cuda().t().tolist()

    def train_epoch():
        model.zero_grad()
        score, score_gd, score_gm = model(train_data)
        loss = regression_crit(one_index, zero_index, train_data[7].cuda(), train_data[3]['data'].cuda(),train_data[4]['data'].cuda(), score, score_gd, score_gm)
        loss.backward()
        optimizer.step()
        return loss, score

    for epoch in range(1, opt.epoch + 1):
        train_reg_loss, predict = train_epoch()
    return predict


if __name__ == "__main__":

    opt = Config()

    D = np.genfromtxt(r"/home/chezicheng/program/data/md_delete.txt")
    R11 = np.genfromtxt(r"/home/chezicheng/program/data/mm_delete.txt")
    R22 = np.genfromtxt(r"/home/chezicheng/program/data/dd_delete.txt")
    gg = np.genfromtxt(r"/home/chezicheng/program/data/gg.txt")
    dg = np.genfromtxt(r"/home/chezicheng/program/data/dg_delete.txt")
    mg = np.genfromtxt(r"/home/chezicheng/program/data/mg_delete.txt")


    # dg = gd.T
    # mg = gm.T
    [row, col] = np.shape(D)

    ######
    prolist = np.array(list(range(col)))

    indexn = np.argwhere(D == 0)
    Index_zeroRow = indexn[:, 0]
    Index_zeroCol = indexn[:, 1]

    indexp = np.argwhere(D == 1)
    Index_PositiveRow = indexp[:, 0]
    Index_PositiveCol = indexp[:, 1]

    ####number of line
    totalassociation = np.size(prolist)
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
        ###生成打乱的列索引
        p = np.random.permutation(totalassociation)
        for f in range(1, cv_num + 1):
            print("multi zeroingcol:", '%01d' % (f))

            if f == cv_num:
                testset = p[((f - 1) * fold): totalassociation + 1]
            else:
                testset = p[((f - 1) * fold): f * fold]

            # test_arr = list(testset)
            # train_arr = list(set(p).difference(set(testset)))
            X = copy.deepcopy(D)
            W = copy.deepcopy(X)
            Xn = copy.deepcopy(X)
            test_length = len(testset)

            #五分之一的列清零
            for ii in range(test_length):
                Xn[:, prolist[testset[ii]]] = 0
                W[:, prolist[testset[ii]]] = 1

            D1 = copy.deepcopy(Xn)
            dataset = prepare_data(opt, W, D1, R11, R22, gg, dg, mg)
            sizes = Sizes(dataset)
            train_data = Dataset(opt, dataset)

            model = Model(sizes)
            model = model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            #######cuda
            gc.collect()
            t.cuda.empty_cache()

            predict = train(model, train_data[f], optimizer, opt)
            predict = predict.data.cpu().numpy()
            result_matrix = create_resultmatrix(predict, testset, prolist)
            result_matrix = result_matrix.reshape(-1, 1)

            ####label
            true_label = create_resultmatrix(D, testset, prolist)
            true_label = true_label.reshape(-1, 1)

            test_auc = roc_auc_score(true_label, result_matrix)
            Auc_per.append(test_auc)
            print("//////////每一次auc: " + str(test_auc))
            varauc.append(test_auc)

            ####
            max_f1_score, threshold = f1_score_binary(torch.from_numpy(true_label).float(),
                                                      torch.from_numpy(result_matrix).float())
            f1_score_per.append(max_f1_score)
            print("//////////max_f1_score:", max_f1_score)
            # acc = accuracy_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),threshold)
            # print("acc:", acc)
            precision = precision_binary(torch.from_numpy(true_label).float(), torch.from_numpy(result_matrix).float(),
                                         threshold)
            precision_per.append(precision)
            print("//////////precision:", precision)
            recall = recall_binary(torch.from_numpy(true_label).float(), torch.from_numpy(result_matrix).float(),
                                   threshold)
            recall_per.append(recall)
            print("//////////recall:", recall)
            # mcc_score = mcc_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(),threshold)
            # print("mcc_score:", mcc_score)
            pr, re, thresholds = precision_recall_curve(true_label, result_matrix)
            aupr = auc(re, pr)
            aupr_per.append(aupr)
            print("//////////aupr", aupr)

            varf1_score.append(max_f1_score)
            varprecision.append(precision)
            varrecall.append(recall)
            varaupr.append(aupr)

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
