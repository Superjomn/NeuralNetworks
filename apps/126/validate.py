#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 3, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import sys
import theano
import numpy
from theano import scalar as T
import cPickle as pickle
import argparse
sys.path.append('../..')
from models.stacked_autoencoder import StackedAutoEncoder
from dataset import Dataset as DenoDataset


def load_dataset(dataset_ph):
    '''
    test if the file in pickle format
    predict if the file in csv format
    '''
    dataset_ph = dataset_ph
    if dataset_ph.endswith('.pk'):
        with open(dataset_ph) as f:
            dataset = pickle.load(f)
    else:
        print '.. dataset is in csv format'
        print '.. will ignore the first line'
        deno_dataset = DenoDataset(dataset_ph)
        records = deno_dataset.load_records_to_norm_float()
        dataset = (records, None)
    return dataset


def load_model(path):
    '''
    load pretrained StackedAutoencoder object from a file
    '''
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


class Validator(object):
    '''
    given some records and predict label
    '''
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self._init()

    def _init(self):
        train_fn, self.predict_fn = self.model.compile_finetune_funcs()

    def predict(self):
        res = []
        records,labels = self.dataset
        n_records = records.shape[0]
        for i in range(n_records):
            x = records[i:i+1]
            #print 'x:', x
            y = self.predict_fn(x)[0]
            print 'y:', y, labels[i]
            res.append(y)
        return res

    def validate(self):
        records,labels = self.dataset
        labels = list(labels)
        n_records = records.shape[0]
        res = self.predict()
        num = 0
        #print 'labels', labels
        print 'len res labels', len(res), len(labels)
        for i in xrange(n_records):
            if res[i] == labels[i]:
                num += 1.0
        #num = len(filter(lambda x:x, res == labels))
        print 'num', num
        c_rate = num/n_records
        print 'Error rate:', 1 - c_rate
        return c_rate




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "predict and validate")
    parser.add_argument('-d', action='store',
        dest='dataset_ph', help='path to dataset'
        )
    parser.add_argument('-t', action='store',
        dest='task', help='task: validate or predict', 
        )
    parser.add_argument('-m', action='store',
        dest='model_ph', help='path of model file', 
        )
    parser.add_argument('-f', action='store',
        dest='topath', help='path of output file'
        )

    if len(sys.argv) == 1:
        parser.print_help()
        exit(-1)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_ph)
    model = load_model(args.model_ph)
    validator = Validator(
        dataset = dataset,
        model = model,
        )

    # task
    if args.task == 'predict':
        pass
    elif args.task == 'validate':
        validator.validate()
    else:
        print 'unrecognized task: "%s"' % args.task

    # TODO to file?
