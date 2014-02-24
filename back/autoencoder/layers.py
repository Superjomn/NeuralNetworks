#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
from __future__ import division

import numpy as np

null = lambda x: 0
identity = lambda x: x

class BaseLayer():
    def backwardPass(self, y):
        gradient = -(y - self.output) + self.w_constraint(self.W)
        #sys.stdout.write( "\rError: %4.4f"%numpy.mean((y - self.output)**2) )
        print "%s %d: %4.4f"%(self.name, self.epoch, np.mean((y - self.output)**2))
        return gradient
        
    def update(self, transfer):
        # theta
        gradient =  transfer * self.gradOutput
        transfer = np.dot(gradient, self.W[1:].T)
        self.W -= self.alpha * np.dot(self.input.reshape(self.n_input, 1), gradient.reshape(1, self.n_output))
        self.W = self.W * self.weight_decay
        self.epoch += 1
        return transfer

class SigmoidLayer(BaseLayer):
    def __init__(self, n_input, n_output, bias=True, alpha=0.01, weight_decay=1, w_constraint=null, noise=identity):
        self.W = np.random.uniform(-0.1,0.1, (n_input + int(bias),n_output))
        self.bias = bias
        self.alpha = alpha
        self.n_input = n_input + int(bias)
        self.n_output = n_output
        self.w_constraint = w_constraint
        self.noise_regularizer = noise
        self.weight_decay = weight_decay
        self.epoch = 0
        self.name = str(n_output) + "x" + str(n_input)
        
    def activate(self, X):
        X = self.noise_regularizer(X)
        if self.bias:
            X = np.column_stack([np.ones((X.shape[0],1)), X])
        print 'noise, X', X
        print 'self.W'
        print self.W
        self.input = X
        self.output = np.tanh(np.dot(self.input, self.W)) 
        self.gradOutput = 1 - self.output**2
        return self.output
        
     
class SoftmaxLayer(BaseLayer):
    def __init__(self, n_input, n_output, bias=True, alpha=0.01, weight_decay=1, w_constraint=null, noise=identity):
        self.W = np.random.uniform(-0.1,0.1, (n_input + int(bias),n_output))
        self.bias = bias
        self.alpha = alpha
        self.n_input = n_input + int(bias)
        self.n_output = n_output
        self.w_constraint = w_constraint
        self.noise_regularizer = noise
        self.weight_decay = weight_decay
        self.epoch = 0
        self.name = str(n_output) + "x" + str(n_input)
        
    def activate(self, X):
        X = self.noise_regularizer(X)
        if self.bias:
            X = np.column_stack([np.ones((X.shape[0],1)), X])
        self.input = X
        v = np.exp(np.dot(self.input, self.W))
        self.output = v/np.sum(v)
        self.gradOutput = self.output*(1 - self.output)
        return self.output

if __name__ == '__main__':
    layer = SigmoidLayer(
        n_input = 8,
        n_output = 4)
    X = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    Y = np.array([1, 0, 0, 1])
    a = layer.activate(X)
    transfer = layer.backwardPass(Y)
    print 'a'
    print a
    print 'transfer',
    print transfer
        
