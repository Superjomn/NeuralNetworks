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

ALPHA = 0.01

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
    def __init__(self, W=None, f=None, par_f=None, b=None, n_neurons=None, n_features=None, alpha=ALPHA):
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
            alpha: float
                study radio
        '''
        BaseLayer.__init__(self)
        self.W = W
        self.f = f
        self.par_f = par_f
        self.b = b
        self.alpha = alpha
        self.n_neurons = n_neurons
        self.n_features = n_features
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
        self.W -= self.alpha * par_W
        self.b -= self.alpha * par_b
        return self.cost

    def reconst(self):
        '''
        print "*" * 40
        print 'self.W'
        print self.W
        print 'self.upper_layer.a'
        print self.upper_layer.a
        print 'self.b'
        print self.b
        print 'self.n_neurons', self.n_neurons
        print 'upper_layer.n_neurons', self.upper_layer.n_neurons
        '''

        tmp = np.dot(
            self.W.reshape(
                (self.n_neurons, self.upper_layer.n_neurons)),
            self.upper_layer.a.reshape( (self.upper_layer.n_neurons, 1))
            )
        #print 'tmp'
        #print tmp
        #print 'self.b'
        #print self.b
        tmp = tmp.reshape(1, self.n_neurons) + self.b.reshape(self.upper_layer.n_neurons, 1)
        self.reconstruction = self.f(tmp)
        #print 'reconstruction'
        #print self.reconstruction
        tmp = (self.reconstruction - self.lower_layer.a)
        cost = np.sum(tmp * tmp)
        print 'cost:', cost
        return self.reconstruction


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


class InputLayer(BaseLayer):
    def __init__(self, X = None):
        BaseLayer.__init__(self)
        self.X = X
        if X:
            self.set_input(X)

    def set_input(self, X):
        self.X = X
        self.n_neurons = len(X)
        self.reconstruction = None
        self._init()
        self.a = X


    def _init(self):
        n = len(self.X)
        self.W = np.identity(n)
        self.b = np.zeros(n)

    def forward(self):
        return self.X

    def error(self):
        return np.sum((self.X - self.reconstruction) ** 2)


class OutputLayer(BaseLayer):
    def __init__(self, f=None, par_f=None, n_neurons=None):
        BaseLayer.__init__(self)
        self.f = f
        self.par_f = f
        self.n_neurons = n_neurons
        self._init()

    def forward(self, X=None):
        X = X if X is not None else self.lower_layer.a
        self.z = self.lower_layer.W * \
            (np.ones((self.n_neurons, 1)) * \
            X.reshape((1, self.lower_layer.n_neurons)))
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

    def _init(self):
        self.f = self.f if self.f else utils.sigmoid
        self.par_f = self.par_f if self.par_f else utils.sigmoid_der


class AutoEncoder(object):

    InputLayer = InputLayer
    HiddenLayer = HiddenLayer
    OutputLayer = OutputLayer

    def __init__(self, input_struc, n_neurons):
        '''
        :parameters:
            input_struc: integer
                length of input
            n_neurons: integer
                num of neuron of the hidden encode
        '''
        self.input_struc = input_struc
        self.n_neurons = n_neurons
        self._init()

    def _init(self):
        # init layers
        self.input_layer = self.InputLayer()
        self.layer1 = self.HiddenLayer(
            n_neurons = self.input_struc, 
            n_features=self.n_neurons, 
            f=utils.tanh, par_f=utils.tanh_der)
        self.layer2 = self.HiddenLayer(
            n_neurons = self.n_neurons, 
            n_features=self.input_struc, 
            f=utils.tanh, par_f=utils.tanh_der)
        self.layer3 = self.OutputLayer(
            n_neurons = self.input_struc,
            f=utils.tanh, 
            par_f=utils.tanh_der)
        self.layer1.set_lower_layer(self.input_layer)
        self.layer1.set_upper_layer(self.layer2)
        self.layer2.set_upper_layer(self.layer3)

    def train(self, inputs, n_iters=100):
        for i in range(n_iters):
            for X in inputs:
                self.input_layer.set_input(X)
                self.layer1.forward(X)
                a2 = self.layer2.forward(X)
                self.layer3.forward(a2)
                grad3 = self.layer3.backward(X)
                grad2 = self.layer2.backward(grad3)
                self.layer1.backward(grad2)
                self.layer1.reconst()
        return self.layer1
