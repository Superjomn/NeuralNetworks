#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 7, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import cPickle as pickle

def obj_from_file(path):
    '''
    load pickle file to memory
    '''
    with open(path, 'rb') as f:
        obj = pickle.load(f)
        return obj
