import math
import numpy as np
import pickle
import random

class dataLoad():
    def __init__(self, cv_fold_num=5):
        # load data
        # [circmir, cirdis, mircirc, mirdis, discir, dismir] circ:1313, mir:638, dis: 152
        with open('data/posSamples_all.pkl', 'rb') as file1:
            pos_all = pickle.load(file1)

        with open('data/posSamples_circdis.pkl', 'rb') as file2:
            pos_circdis = pickle.load(file2)

        with open('data/posSamples_others.pkl', 'rb') as file3:
            pos_others = pickle.load(file3)

        with open('data/neg_cirdis_tmp.pkl', 'rb') as file6:
            neg_cirdis_tmp = pickle.load(file6)

        with open('data/neg_others_tmp.pkl', 'rb') as file7:
            neg_others_tmp = pickle.load(file7)

        self.pos_all = pos_all
        self.pos_circdis = pos_circdis
        self.pos_others = pos_others

        # self.pos_others = [i for i in self.pos_all if i not in self.pos_circdis]
        # with open('data/posSamples_others.pkl', 'wb') as file3:
        #     pickle.dump(self.pos_others, file3)

        # self.neg_set_cirdis = []
        # self.neg_set_others = []

        self.neg_set_cirdis = neg_cirdis_tmp
        self.neg_set_others = neg_others_tmp

        self.num_nodes = 2102
        self.cv_fold_num = cv_fold_num
        self.train_num_ratio = float(cv_fold_num-1) / cv_fold_num
        self.val_num_ratio = (1 - self.train_num_ratio) / 2

        self.test_cirdis_pos_size = 0
        self.test_others_pos_size = 0

        self.trainVal_cirdis_pos = []
        self.trainVal_cirdis_neg = []
        self.trainVal_others_pos = []
        self.trainVal_others_neg = []

    def generateNegSamples(self):
        # generate negative samples circdis --- keep cirdis samples ratio
        neg_set_cirdis = []
        while len(neg_set_cirdis) < 4026:
            rowInd1 = random.randint(0, 1312) # restrict range
            colInd1 = random.randint(1951, 2102)
            tmpSample = (rowInd1, colInd1)
            if tmpSample not in self.pos_circdis:
                neg_set_cirdis.append(tmpSample)

        self.neg_set_cirdis = neg_set_cirdis

        neg_set_others = []
        while len(neg_set_others) < len(self.pos_others):
            # circ-dis negative
            rowInd2 = random.randint(0, self.num_nodes)
            colInd2 = random.randint(0, self.num_nodes)
            tmpSample = (rowInd2, colInd2)
            if tmpSample not in self.pos_all:
                if tmpSample not in self.neg_set_cirdis:
                    neg_set_others.append(tmpSample)

        self.neg_set_others = neg_set_others

        # with open('data/neg_cirdis_tmp.pkl', 'wb') as file4:
        #     pickle.dump(neg_set_cirdis, file4)
        #
        # with open('data/neg_others_tmp.pkl', 'wb') as file5:
        #     pickle.dump(neg_set_others, file5)

    def generateTestSamples(self):
        pos_cirdis_tmp = self.pos_circdis
        pos_others_tmp = self.pos_others
        neg_cirdis_tmp = self.neg_set_cirdis
        neg_others_tmp = self.neg_set_others

        pos_cirdis_num = len(self.pos_circdis)
        pos_others_num = len(self.pos_others)
        neg_cirdis_num = len(self.neg_set_cirdis)
        neg_others_num = len(self.neg_set_others)

        random.shuffle(pos_cirdis_tmp)
        random.shuffle(pos_others_tmp)
        random.shuffle(neg_cirdis_tmp)
        random.shuffle(neg_others_tmp)

        # test samples and test labels
        test_circdis = pos_cirdis_tmp[:math.ceil(0.1 * pos_cirdis_num)] \
                       + neg_cirdis_tmp[:math.ceil(0.1 * neg_cirdis_num)]

        test_others = pos_others_tmp[:math.ceil(0.1 * pos_others_num)] \
                      + neg_others_tmp[:math.ceil(0.1 * neg_others_num)]

        test_cirdis_label_num = len(pos_cirdis_tmp[:math.ceil(0.1 * pos_cirdis_num)])
        test_others_label_num = len(pos_others_tmp[:math.ceil(0.1 * pos_others_num)])

        test_label_cirdis = [1] * test_cirdis_label_num + [0] * test_cirdis_label_num
        test_label_others = [1] * test_others_label_num + [0] * test_others_label_num

        self.test_cirdis_pos_size = test_cirdis_label_num
        self.test_others_pos_size = test_others_label_num
        self.trainVal_cirdis_pos = pos_cirdis_tmp[math.ceil(0.1 * pos_cirdis_num):]
        self.trainVal_cirdis_neg = neg_cirdis_tmp[math.ceil(0.1 * neg_cirdis_num):]
        self.trainVal_others_pos = pos_others_tmp[math.ceil(0.1 * pos_others_num):]
        self.trainVal_others_neg = neg_others_tmp[math.ceil(0.1 * neg_others_num):]

        test_samples = test_circdis + test_others
        test_labels = test_label_cirdis + test_label_others

        return test_samples, test_labels

    def shuffleMakeTraining(self):
        pos_cirdis_tmp = self.trainVal_cirdis_pos
        pos_others_tmp = self.trainVal_others_pos
        neg_cirdis_tmp = self.trainVal_cirdis_neg
        neg_others_tmp = self.trainVal_others_neg

        pos_cirdis_num = len(self.pos_circdis)
        pos_others_num = len(self.pos_others)
        neg_cirdis_num = len(self.neg_set_cirdis)
        neg_others_num = len(self.neg_set_others)

        random.shuffle(pos_cirdis_tmp)
        random.shuffle(pos_others_tmp)
        random.shuffle(neg_cirdis_tmp)
        random.shuffle(neg_others_tmp)

        # train samples and train labels
        train_circdis = pos_cirdis_tmp[:math.ceil(0.8*pos_cirdis_num)] \
                 + neg_cirdis_tmp[:math.ceil(0.8*neg_cirdis_num)]

        train_others = pos_others_tmp[:math.ceil(0.8*pos_others_num)] \
                 + neg_others_tmp[:math.ceil(0.8*neg_others_num)]

        train_cirdis_label_num = len(pos_cirdis_tmp[:math.ceil(0.8*pos_cirdis_num)])
        train_others_label_num = len(pos_others_tmp[:math.ceil(0.8*pos_others_num)])

        train_label_cirdis = [1] * train_cirdis_label_num + [0] * train_cirdis_label_num
        train_label_others = [1] * train_others_label_num + [0] * train_others_label_num

        train_samples = train_circdis + train_others
        train_labels = train_label_cirdis + train_label_others

        # valid samples and valid labels
        val_circdis = pos_cirdis_tmp[math.ceil(0.8*pos_cirdis_num):] \
                 + neg_cirdis_tmp[math.ceil(0.8*neg_cirdis_num):]

        val_others = pos_others_tmp[math.ceil(0.8*pos_others_num):] \
                 + neg_others_tmp[math.ceil(0.8*neg_others_num):]

        val_cirdis_label_num = len(pos_cirdis_tmp[math.ceil(0.8*pos_cirdis_num):])
        val_others_label_num = len(pos_others_tmp[math.ceil(0.8*pos_others_num):])

        val_label_cirdis = [1] * val_cirdis_label_num + [0] * val_cirdis_label_num
        val_label_others = [1] * val_others_label_num + [0] * val_others_label_num

        val_samples = val_circdis + val_others
        val_labels = val_label_cirdis + val_label_others

        return train_samples, train_labels, val_samples, val_labels

