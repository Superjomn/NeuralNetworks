#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import numpy as np
import utils

class BaseLayer(object):
    def __init__(self):
        self.upper_layer = None
        self.lower_layer = None

    def set_lower_layer(self, layer):
        self.lower_layer = layer
        if not layer.upper_layer:
            layer.set_upper_layer(self)

    def set_upper_layer(self, layer):
        self.upper_layer = layer
        if not layer.lower_layer:
            layer.set_lower_layer(self)

    def forward(self, X):
        '''
        forward algorithm
        :parameters: 
            X: vector
            the input
        '''
        raise NotImplemented

    def backward(self, upl_cost):
        '''
        backward algorithm
        :parameters:
            upl_cost: vector
            uppder cost
        '''
        raise NotImplemented


class HiddenLayer(BaseLayer):
    '''
    a layer of neural network
    input activations of the previous layer
    and output activations of the current layer
    '''
    def __init__(self, W=None, f=None, par_f=None, b=None, n_neurons=None, n_features=None):
        '''
        :parameters:
            f:  function
                activation function
            par_f: function
                Partial differential function
            W: matrix(array)
                n_neurons*n_features weight matrix
            b: vector
                n_neurons*1 bias
            n_neurons: integer
                number of current layer's neurons
            n_features: integer
                number of next layer's neurons
        '''
        BaseLayer.__init__(self)
        self.W = W
        self.f = f
        self.par_f = par_f
        self.b = b
        self.n_neurons = n_neurons
        self.n_features = n_features
        self.upper_layer = None
        self._init()

    def forward(self, X=None):
        X = X if X is not None else self.lower_layer.a
        self.z = self.lower_layer.W * \
            (np.ones((self.n_neurons, 1)) * \
            X.reshape((1, self.lower_layer.n_neurons)))
        self.z = np.sum(self.z, axis=1) + self.lower_layer.b
        self.a = self.f(self.z)
        return self.a

    def backward(self, up_cost=None):
        # cost's deviration is (1, n_neurons)
        if up_cost is None:
            up_cost = self.upper_layer.cost
        self.cost = []
        for i in range(self.n_neurons):
            theta_i = sum(
                self.W[j][i] * up_cost[j] \
                for j in range(len(up_cost))) \
                    * self.par_f(self.z[i])
            self.cost.append(theta_i)
        self.cost = np.array(self.cost)
        par_W = up_cost.reshape(self.n_features, 1) * self.a.reshape(1, self.n_neurons)
        #print 'self.W.shape', self.W.shape
        #print 'up_cost', up_cost.shape
        par_b = up_cost
        self.W += par_W
        self.b += par_b
        return self.cost

    def _init(self):
        '''
        init parameters
        '''
        # W, b serves for the upper layer
        self.W = self.W if self.W else np.random.random((self.n_features, self.n_neurons))
        self.b = self.b if self.b is not None else np.random.random( self.n_features)
        self.f = self.f if self.f else utils.sigmoid
        self.par_f = self.par_f if self.par_f else utils.sigmoid_der

    def show(self):
        print '<Layer:'
        print 'n_neurons', self.n_neurons
        print 'W'
        print self.W
        print 'b'
        print self.b
        print 'f'
        print self.f
        print '>'


class OutputLayer(BaseLayer):
    def __init__(self, f=None, par_f=None, n_neurons=None):
        self.f = f
        self.par_f = f
        self.n_neurons = n_neurons

    def forward(self, X=None):
        X = X if X else self.lower_layer.a
        self.z = self.lower_layer.W * \
            (np.ones(self.n_neurons, 1) * \
            X.reshape(1, self.n_neurons))
        self.z = np.sum(self.z, axis=1) + self.lower_layer.b
        self.a = self.f(self.z)
        return self.a

    def backward(self, Y):
        self.cost = []
        for i in range(self.n_neurons):
            theta_i = \
                -(Y[i] - self.a[i]) * self.par_f(self.z[i])
            self.cost.append(theta_i)
        self.cost = np.array(self.cost)
        #self.cost = -(Y - self.a) * self.par_f(self.z)
        return self.cost



