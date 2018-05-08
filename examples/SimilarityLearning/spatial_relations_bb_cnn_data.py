#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: embedding_data.py
# Author: tensorpack contributors

import numpy as np
import sys
sys.path.append('utils/')
from amt import Dataset
import random

num_relations = 12

def get_test_data(pathFile, data_type='test', batch=128):
    ds = Dataset(pathFile, data_type, shuffle=True)
    ds = BatchData(ds, batch)
    return ds


def get_digits_by_label(features, labels, bb):
    global num_relations
    data_dict = []
    for clazz in range(0, num_relations):
        #clazz_filter = np.where(labels == clazz)
        #data_dict.append(list(images[clazz_filter].reshape((-1, 28, 28))))
        clazz_filter = [i for i, j in enumerate(labels) if j == clazz]
        
        #features_clazz = [features[i] for i in clazz_filter]
        #bb_clazz = [bb[i] for i in clazz_filter]
        data_clazz = [np.concatenate((features[i], bb[i]),axis=0) for i in clazz_filter]
        data_dict.append(data_clazz)
    #return data_dict
    # TODO: combine feat and bb into one vector 

    return data_dict


class DatasetPairs(Dataset):
    """We could also write

    .. code::

        ds = dataset.Mnist('train')
        ds = JoinData([ds, ds])
        ds = MapData(ds, lambda dp: [dp[0], dp[2], dp[1] == dp[3]])
        ds = BatchData(ds, 128 // 2)

    but then the positives pairs would be really rare (p=0.1).
    """
    global num_relations
    def __init__(self, pathFile, train_or_test):
        super(DatasetPairs, self).__init__(pathFile, train_or_test, shuffle=False)
        # now categorize these digits
        self.data_dict = get_digits_by_label(self.features, self.labels, self.bb)

    def pick(self, label):
        idx = self.rng.randint(len(self.data_dict[label]))
        return self.data_dict[label][idx].astype(np.float32)

    def pick2(self,label):
        idxs = random.sample(range(0,len(self.data_dict[label])-1), 2)
        idx1 = idxs[0]
        idx2 = idxs[1]
        return self.data_dict[label][idx1].astype(np.float32), self.data_dict[label][idx2].astype(np.float32)

    def get_data(self):
        while True:
            y = self.rng.randint(2)
            if y == 0:
                pick_label, pick_other = self.rng.choice(num_relations, size=2, replace=False)
            else:
                pick_label = self.rng.randint(num_relations)
                pick_other = pick_label

            a = self.pick(pick_label)
            b = self.pick(pick_other)

            yield [a, b, y]


class DatasetTriplets(DatasetPairs):
    def get_data(self):
        while True:
            pick_label, pick_other = self.rng.choice(10, size=2, replace=False)
            #yield [self.pick(pick_label), self.pick(pick_label), self.pick(pick_other)]
            ab = self.pick2(pick_label)
            c = self.pick(pick_other)

            yield [ab[0], ab[1], c]
