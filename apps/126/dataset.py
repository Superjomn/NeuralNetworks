# -*- coding: utf-8 -*-
'''
Created on March 2, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division

N_PIXEL_VALUES = 256
MAX_PIXEL_VALUE = 255

import sys
sys.path.append('..')
sys.path.append('../../')

import csv
import cPickle as pickle
import os
import numpy
import random
import theano
from utils import Timeit

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
        self.trainset = None

    def load_ori_dataset(self):
        print 'load data ...'
        timeit = Timeit()
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
        timeit.print_time()

    def load_dataset_to_norm_float(self):
        '''
        load original dataset and transform
        value of each pixel to 0-1 float
        '''
        print 'load data ...'
        timeit = Timeit()
        with open(self.data_ph) as f:
            reader = csv.reader(f)
            for i,ls in enumerate(reader):
                if i == 0:
                    continue
                if i % 1000 == 0:
                    print '> load\t%d\trecords' % i
                label = int(ls[0])
                record = [int(r)/N_PIXEL_VALUES for r in ls[1:]]
                self.records.append(record)
                self.labels.append(label)

        self.labels = numpy.array(self.labels)
        self.records = numpy.array(self.records).astype(theano.config.floatX)
        timeit.print_time()

    def load_records_to_norm_float(self):
        print 'load data ...'
        timeit = Timeit()
        with open(self.data_ph) as f:
            reader = csv.reader(f)
            for i,ls in enumerate(reader):
                if i == 0:
                    continue
                if i % 1000 == 0:
                    print '> load\t%d\trecords' % i
                record = [int(r)/N_PIXEL_VALUES for r in ls]
                self.records.append(record)

        self.records = numpy.array(self.records).astype(theano.config.floatX)
        timeit.print_time()
        return self.records



    def tofile(self):
        print '... save data in pickle format'
        timeit = Timeit()
        dataset = (self.labels, self.records,)
        with open(self.pk_data_ph, 'wb') as f:
            pickle.dump(dataset, f)
        timeit.print_time()


    def fromfile(self):
        print '... load dataset from :\t %s' % self.pk_data_ph
        timeit = Timeit()
        with open(self.pk_data_ph, 'rb') as a:
            self.labels, self.records = pickle.load(a)
        timeit.print_time()
        print '... done'

        
    def trans_data_type(self, train_prob=0.8):
        if self.trainset: return
        print 'trains data ...'
        timeit = Timeit()
        self.train_prob = train_prob
        n_records = len(self.labels)
        n_trainset = int(n_records * self.train_prob)

        self.trainset = (self.labels[:n_trainset], self.records[:n_trainset],)
        # validation set
        self.validset = (self.labels[n_trainset:], self.records[n_trainset:],)
        print 'load records', len(self.labels)
        # train set
        labels, records = self.trainset
        labels = numpy.array(labels)
        records = numpy.array(records).astype(theano.config.floatX)
        records /= MAX_PIXEL_VALUE
        self.trainset = (labels, records)
        # validate set
        labels, records = self.validset
        labels = numpy.array(labels)
        records = numpy.array(records).astype(theano.config.floatX)
        records /= MAX_PIXEL_VALUE
        self.validset = (labels, records)
        timeit.print_time()
        return self.trainset, self.validset

    def sample(self, n_records=100):
        '''
        get a part of dataset
        '''
        self.trans_data_type()
        labels, records = self.trainset
        return labels[:n_records], records[:n_records]


# operations of dataset
def save_ori_dataset_to_norm_pk(ori_dataset_ph, pk_data_ph):
    print 'save original dataset to norm pickle file ... '
    timeit = Timeit()
    d = Dataset(ori_dataset_ph, pk_data_ph)
    d.load_dataset_to_norm_float()
    data = (d.records, d.labels)
    with open(pk_data_ph, 'wb') as f:
        pickle.dump(data, f)
    timeit.print_time()

def sample_norm_dataset_to_file(pk_data_ph, data_root="data", n_samples=1000):
    print 'sample norm dataset to file ...'
    timeit = Timeit()
    name = os.path.join(data_root, "sample-%d.pk" % n_samples)
    print 'load pickle data ...'
    with open(pk_data_ph, 'rb') as f:
        records, labels = pickle.load(f)
    # generate random index
    n_records = records.shape[0]
    assert n_records > n_samples
    print 'generate random indexs ...'
    indexs = set()
    while len(indexs) < n_samples:
        r = random.randint(0, n_records-1)
        indexs.add(r)
    print 'sample the dataset ...'
    indexs = list(indexs)
    sample_records = records[indexs]
    sample_labels = labels[indexs]
    data = (sample_records, sample_labels)
    print 'dump data to \t', name
    with open(name, 'wb') as f:
        pickle.dump(data, f)
    timeit.print_time()


def load_dataset(path):
    '''
    load (records, labels) from pickle file
    '''
    t = Timeit()
    print 'load dataset from:\t', path
    with open(path, 'rb') as f:
        data = pickle.load(f)
    records, labels = data
    #print 'r:', records[0]
    #print 'l:', labels[0]
    t.print_time()
    return data


def split_dataset_to_train_validate(dataset_ph, data_root="data", ratio=0.8):
    timeit = Timeit()
    # paths
    train_ph = os.path.join(data_root, 'train-%f.pk'%ratio)
    valid_ph = os.path.join(data_root, 'valid-%f.pk'%ratio)

    records, labels  = load_dataset(dataset_ph)
    #assert labels
    n_records = records.shape[0]
    n_train = n_records * ratio
    train_ids = set()
    while len(train_ids) < n_train:
        r = random.randint(0, n_records-1)
        train_ids.add(r)
    # get trainset
    print 'generate trainset ...'
    train_ids = list(train_ids)
    t_records = records[train_ids]
    t_labels = labels[train_ids]
    # get validset 
    print 'generate validset ...'
    total_ids = set([i for i in xrange(n_records)])
    [total_ids.discard(i) for i in train_ids]
    v_records = records[list(total_ids)]
    v_labels = labels[list(total_ids)]

    trainset = (t_records, t_labels)
    validset = (v_records, v_labels)

    with open(train_ph, 'wb') as f:
        pickle.dump(trainset, f)

    with open(valid_ph, 'wb') as f:
        pickle.dump(validset, f)
    timeit.print_time()





if __name__ == '__main__':
    #save_ori_dataset_to_norm_pk('./trainset.csv', './data/norm_float_dataset.pk')
    #sample_norm_dataset_to_file('./data/norm_float_dataset.pk', n_samples=6000)
    #sample_norm_dataset_to_file('./data/norm_float_dataset.pk', n_samples=8000)
    #sample_norm_dataset_to_file('./data/norm_float_dataset.pk', n_samples=3000)
    #load_dataset('data/sample-3000.pk')
    split_dataset_to_train_validate('./data/train-0.800000.pk', ratio=0.78)
