#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    '''
    derivation of sigmoid
    '''
    return - np.exp(x) \
        (np.exp(x) + 1) ** 2

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / \
            (np.exp(z) + np.exp(-z))

def tanh_der(z):
    return 4 * np.exp(2 * z) / \
            (np.exp(2*z) + 1) ** 2



if __name__ == "__main__":
    pass

