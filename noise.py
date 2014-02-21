#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
from __future__ import division

import np as np

class Noise():
    def SaltAndPepper(self, X, rate=0.3):
        # Salt and pepper noise
        
        drop = np.arange(X.shape[1])
        np.random.shuffle(drop)
        sep = int(len(drop)*rate)
        drop = drop[:sep]
        X[:, drop[:sep/2]]=0
        X[:, drop[sep/2:]]=1
        return X
        
    def GaussianNoise(self, X, sd=0.5):
        # Injecting small gaussian noise
        X += np.random.normal(0, sd, X.shape)
        return X
        
    def MaskingNoise(self, X, rate=0.5):
        mask = (np.random.uniform(0,1, X.shape)<rate).astype("i4")
        X = mask*X
        return X
        
def SaltAndPepper(rate=0.3):
    # Salt and pepper noise
    def func(X):
        drop = np.random.uniform(0,1, X.shape)
        z = np.where(drop < 0.5*rate)
        o = np.where(np.abs(drop - 0.75*rate) < 0.25*rate)
        X[z]=0
        X[o]=1   
        return X
    return func
    
def GaussianNoise(self, sd=0.5):
    # Injecting small gaussian noise
    def func(X):
        X += np.random.normal(0, sd, X.shape)
        return X
    return func
