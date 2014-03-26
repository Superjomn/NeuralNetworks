#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 7, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import numpy as np

class BaseNode(object):
    '''
    base model of tree's node
    '''
    def __init__(self, lchild=None, rchild=None):
        self.lchild = lchild
        self.rchild = rchild

    def is_leaf(self):
        return not (self.lchild or self.rchild)


class BinaryNode(BaseNode):
    def __init__(self, lchild=None, rchild=None, vector=None):
        # index to determine wheather to update vectors
        BaseNode.__init__(self, 
            lchild, rchild)
        self.pred_index = 0
        # count of children
        self.n_children = np.float32(1.0)
        self.vector = vector
        self.delta_out1 = None
        self.delta_out2 = None
        self.parent_delta = None
        self.Y1C1 = None
        self.Y2C2 = None



if __name__ == "__main__":
    pass

