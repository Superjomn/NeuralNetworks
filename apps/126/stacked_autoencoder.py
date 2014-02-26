# -*- coding: utf-8 -*-
'''
Created on Feb 25, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import csv
import cPickle as pickle
import time
import numpy
import theano
from theano import tensor as T

class Dataset(object):
    def __init__(self, ori_dataset_ph=None, pk_data_ph=None):
        '''
        :parameters:
            data_ph: string
                path to train dataset

            train_prob: float
                the proportion of train set
        '''
        self.data_ph = ori_dataset_ph
        self.pk_data_ph = pk_data_ph
        self.records = []
        self.labels = []

    def load_ori_dataset(self):
        print 'load data ...'
        start_time = time.time()
        with open(self.data_ph) as f:
            reader = csv.reader(f)
            for i,ls in enumerate(reader):
                if i == 0:
                    continue
                if i % 1000 == 0:
                    print '> load\t%d\trecords' % i
                label = int(ls[0])
                record = [int(r) for r in ls[1:]]
                self.records.append(record)
                self.labels.append(label)
        end_time = time.time()
        print '> used time: %d seconds' % int(start_time - end_time)

    def tofile(self):
        print '... save data in pickle format'
        dataset = (self.labels, self.records,)
        with open(self.pk_data_ph, 'wb') as f:
            pickle.dump(dataset, f)

    def fromfile(self):
        print '... load dataset from :\t %s' % self.pk_data_ph
        start_time = time.time()
        with open(self.pk_data_ph, 'rb') as a:
            self.labels, self.records = pickle.load(a)
        end_time = time.time()
        print '>used %d seconds' % int(end_time - start_time)
        print '... done'
        
    def trans_data_type(self, train_prob=0.8):
        print 'trains data ...'
        self.train_prob = train_prob
        n_records = len(self.labels)
        n_trainset = int(n_records * self.train_prob)

        self.trainset = (self.labels[:n_trainset], self.records[:n_trainset],)
        # validation set
        self.validset = (self.labels[n_trainset:], self.records[n_trainset:],)
        print 'load records', len(self.labels)
        # train set
        labels, records = self.trainset
        labels = numpy.array(labels).astype(theano.config.floatX)
        records = numpy.array(records).astype(theano.config.floatX)
        self.trainset = (labels, records)
        # validate set
        labels, records = self.validset
        labels = numpy.array(labels).astype(theano.config.floatX)
        records = numpy.array(records).astype(theano.config.floatX)
        self.validset = (labels, records)
        return self.trainset, self.validset



if __name__ == '__main__':
    dataset = Dataset('./trainset.csv', './dataset.pk')
    dataset.load_ori_dataset()
    dataset.tofile()
    #dataset.fromfile()
    #trainset, validset = dataset.trans_data_type()
    #print trainset.shape, validset.shape
