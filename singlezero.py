

#import os
import gc
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from torch import nn, optim
from model import Model
from trainData import Dataset
from sklearn.metrics import roc_curve, auc
import numpy as np
import copy
from utils import *


class Config(object):
    def __init__(self):
        self.validation = 5
        self.epoch = 100
        self.alpha = 0.1


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):

        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return (1 - opt.alpha) * loss_sum[one_index].sum() + opt.alpha * loss_sum[zero_index].sum()


class Sizes(object):
    def __init__(self, dataset):
        self.mdg = dataset['dd'].size(0) + dataset['mm'].size(0) + dataset['gg'].size(0)
        # self.m = dataset['mm']['data'].size(0)
        # self.d = dataset['dd']['data'].size(0)
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
        score = model(train_data)
        loss = regression_crit(one_index, zero_index, train_data[7].cuda(), score)
        loss.backward()
        optimizer.step()
        return loss, score

    for epoch in range(1, opt.epoch + 1):

        train_reg_loss, perdict = train_epoch()
    return perdict


if __name__ == "__main__":

    opt = Config()
    index1 = 0
    while index1 < 3:

        if index1 == 0:
            datasetname = 'HMDD2.0-Lan'
            print("%s: " % datasetname)

        if index1 == 1:
            datasetname = 'HMDD2.0-Yan'
            print("%s: " % datasetname)

        if index1 == 2:
            datasetname = 'HMDD3.0new'
            print("%s: " % datasetname)

        # D = np.genfromtxt(r"/home/chezicheng/program/data/" + datasetname + "-miRNA-disease.txt")
        # R22 = np.genfromtxt(r"/home/chezicheng/program/data/" + datasetname + "-diseasefunsim.txt")
        # R12 = np.genfromtxt(r"/home/chezicheng/program/data/" + datasetname + "-diseasesemsim.txt")

        D = np.genfromtxt(r"./data/" + datasetname + "-miRNA-disease.txt")
        R22 = np.genfromtxt(r"./data/" + datasetname + "-diseasefunsim.txt")
        R12 = np.genfromtxt(r"./data/" + datasetname + "-diseasesemsim.txt")

        
        gg = np.genfromtxt(r".\data\\" + datasetname + "-gg.txt")
        #gg = np.genfromtxt(r"/home/chezicheng/program/data/" + datasetname + "-gg.txt")
        dg = np.genfromtxt(r".\data\\" + datasetname + "-dg.txt")
        #dg = np.genfromtxt(r"/home/chezicheng/program/data/" + datasetname + "-dg.txt")
        mg = np.genfromtxt(r".\data\\" + datasetname + "-mg.txt")
        #mg = np.genfromtxt(r"/home/chezicheng/program/data/" + datasetname + "-mg.txt")

        [row, col] = np.shape(D)
        for i in range(col):
            if R22[i, i] != 1:
                R22[i, i] = 1

        ######
        prolist = np.array(list(range(col)))

        indexn = np.argwhere(D == 0)
        Index_zeroRow = indexn[:, 0]
        Index_zeroCol = indexn[:, 1]

        indexp = np.argwhere(D == 1)
        Index_PositiveRow = indexp[:, 0]
        Index_PositiveCol = indexp[:, 1]

        ####
        totalassociation = col
        fold = int(totalassociation / col)
        zero_length = np.size(Index_zeroRow)
        #cv_num = 5
        cv_num = col
        n = 10
        varauc = []
        AAuc_list1 = []
        test_aucs = []

        for time in range(1, n + 1):
            Auc_per = []
            p = np.random.permutation(totalassociation)
            for f in range(cv_num):
                print("single zeroing:", '%01d' % (f))
                testset = p[f]
                #if i == cv_num:
                    #testset = p[((i - 1) * fold): totalassociation + 1]
                #else:
                    #testset = p[((i - 1) * fold): i * fold]

                # test_arr = list(testset)
                # train_arr = list(set(p).difference(set(testset)))
                X = copy.deepcopy(D)
                W = copy.deepcopy(X)
                Xn = copy.deepcopy(X)
                #test_length = len(testset)

                Xn[:, prolist[testset]] = 0
                W[:, prolist[testset]] = 1

                D1 = copy.deepcopy(Xn)
                #
                R11 = np.eye(row)
                for i in range(row):
                    for j in range(i + 1, row):
                        setm1 = np.where(D1[i, :] == 1)
                        setm2 = np.where(D1[j, :] == 1)
                        sumnumm1 = sum(D1[i, :])
                        sumnumm2 = sum(D1[j, :])
                        summ1 = 0
                        if len(setm1[0]) > 0 and len(setm2[0]) > 0:
                            for k in range(len(setm1[0])):
                                summ1 = summ1 + R12[setm1[0][k], setm2[0]].max()
                        else:
                            summ1 = 0
                        summ2 = 0
                        if len(setm2[0]) > 0 and len(setm1[0]) > 0:
                            for k in range(len(setm2[0])):
                                summ2 = summ2 + R12[setm2[0][k], setm1[0]].max()
                        else:
                            summ2 = 0
                        if (sumnumm1 + sumnumm2) > 0:
                            R11[i, j] = (summ1 + summ2) / (sumnumm1 + sumnumm2)
                            R11[j, i] = R11[i, j]

                Km, Kd = computer_GIP(D1, 1, 1)
                for i in range(row):
                    for j in range(row):
                        if R11[i][j] == 0:
                            R11[i][j] = Km[i][j]
                for i in range(col):
                    for j in range(col):
                        if R22[i][j] == 0:
                            R22[i][j] = Kd[i][j]

                dataset = prepare_data(opt, W, D1, R11, R22, gg, dg, mg)
                sizes = Sizes(dataset)
                train_data = Dataset(opt, dataset)

                model = Model(sizes)
                model = model.cuda()

                optimizer = optim.Adam(model.parameters(), lr=0.0001)
                #######cuda
                gc.collect()
                t.cuda.empty_cache()

                perdict = train(model, train_data[f], optimizer, opt)
                perdict = perdict.data.cpu().numpy()

                result_matrix = create_resultmatrix(perdict, testset, prolist)
                ####label
                true_matrix = create_resultmatrix(D, testset, prolist)

                false_positive_rate, recall, thresholds = roc_curve(true_matrix, result_matrix)
                test_auc = auc(false_positive_rate, recall)

                Auc_per.append(test_auc)
                print("//////////每一次auc: " + str(test_auc))
                varauc.append(test_auc)

            AAuc_list1.append(np.mean(Auc_per))
            print("//////////average: " + str(AAuc_list1))
        index1 += 1
        vauc = np.var(varauc)
        print("sumauc = %f±%f\n" % (float(np.mean(AAuc_list1)), vauc))
