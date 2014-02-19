#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
from __future__ import division

class Layer(object):
    def __init__(self, f=None, W=None):
        '''
        a layer of neural network

        :parameters:
            f: function
                activation function like sigmoid or tanh
            W: np.matrix
                weight matrix
            b: bias
        '''
        self.f = f
        self.W = W

    def step(self, 


    def _init(self):
        if None in (self.f, self.W,):
            raise Exception("f, W is None")

        


if __name__ == "__main__":
    pass

