# -*- coding: utf-8 -*-
'''
Created on Feb 25, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import sys
sys.path.append('..')
sys.path.append('../../')
import time
from utils import Timeit
from dataset import *
from models.stacked_autoencoder import StackedAutoEncoder


class Trainer(object):
    def __init__(self, pk_data_ph):
        #self.dataset = Dataset(pk_data_ph = pk_data_ph)
        self._init()

    def _init(self):
        #self.dataset.fromfile()
        #self.trainset, self.validset = self.dataset.trans_data_type()
        self.trainset  = load_dataset('data/sample-3000.pk')
        #print 'trainset:'
        #print self.trainset
        # train the model
        records, labels = self.trainset
        #records = numpy.array(records).astype(theano.config.floatX)
        #labels = numpy.array(labels)

        n_records, n_features = records.shape

        self.sA = StackedAutoEncoder(
            n_visible = n_features,
            hidden_struct = [500, 200],
            n_output = 10,
            corrupt_levels = [0.03, 0.03],
            learning_rate = 0.02,
            )

    def __call__(self):
        records, labels = self.trainset
        timeit = Timeit(time.time())
        for i in range(20):
            self.sA.pretrain(records, n_iters=1000, batch_size=400)
            timeit.print_time()
            self.sA.finetune(records, labels, n_iters=800, batch_size=400)
        timeit.print_time()






if __name__ == '__main__':
    #dataset = Dataset('./trainset.csv', './norm_float_dataset.pk')
    #dataset.load_ori_dataset()
    #dataset.load_dataset_to_norm_float()
    #dataset.tofile()
    #dataset.fromfile()
    #trainset, validset = dataset.trans_data_type()
    #print trainset.shape, validset.shape
    trainer = Trainer(
        pk_data_ph = './norm_float_dataset.pk'
        )
    trainer()
