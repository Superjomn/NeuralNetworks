#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import numpy as np

class Loss():
    def __init__(self):
        self.label = "Uninitialized Loss Function"
        
    def MSE(self, x, y):
        self.dE_dY = -(y - x)
        self.E = np.sum(self.dE_dY**2)
        self.label = "MSE: "
        return self
        
    def __repr__(self):
        return self.label+ str(self.E)
        
    def Cross_Entropy(self, x, y):
        def loss(x, o):
            return 0.5*(x - o)**2
        zeros = np.where(y==0)[0]
        ones = np.where(y==1)[0]
        other = np.where(np.sin(y*np.pi)>0.01)[0]
        self.E = np.zeros(y.shape)
        self.E[zeros] = - np.log(1 - x[zeros]) 
        self.E[ones] = - np.log(x[ones]) 
        self.E[other] = loss(y[other], x[other])
        self.E = np.sum(self.E**2)
        
        # compute gradients
        self.dE_dY = - (y - x)
        self.label = "Log Loss: "
        return self
