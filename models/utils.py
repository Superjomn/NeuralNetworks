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

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / \
            (np.exp(z) + np.exp(-z))



if __name__ == "__main__":
    pass

