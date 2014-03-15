#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 7, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division

class BaseNode(object):
    '''
    base model of tree's node
    '''
    def __init__(self):
        self.lchild = None
        self.rchild = None

    def is_leaf(self):
        return not (self.lchild or self.rchild)



if __name__ == "__main__":
    pass

