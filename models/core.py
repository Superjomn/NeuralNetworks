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

class Neuron(object):
    def __init__(self, f, W, b, index=-1, tolayer=None):
        '''
        a basic neuron

        :parameters:
            tolayer: object of Layer
                to pass the parameters
            f: function
                activation function
            W: vector
                weight vector or matrix
                if current object is a single neuron, then pass
                    in a vector
                if it belongs to a layer, then a weight matrix
                    should be passed, each neuron's weight 
                    vector is the indexth row of weight matrix
            b: float
                bias
            index: integer
                neuron's id of a layer, to get the 
                corresponding weight row in weight matrix W
        '''
        self.tolayer = tolayer
        self.index = index
        self._f = f
        self._W = W
        self._b = b

    @property
    def f(self):
        return self.tolayer.f if self.tolayer else self._f

    @property
    def W(self):
        if self.index != -1:
            return self.tolayer.W[self.index]
        else:
            return self._W

    @property
    def b(self):
        return self.tolayer.b if self.tolayer else self._b

    def activate(self, X):
        '''
        :parameters: 
            X: vector
                the input
        '''
        v = sum(self.W * X.T)
        return self.f(v) 

    def _init(self):
        if None in (self.f, self.W):
            raise Exception("None in self.f, self.W")


class Layer(object):
    '''
    a layer of neural network
    input activations of the previous layer
    and output activations of the current layer
    '''
    def __init__(self, W=None, f=None, b=None, n_neurons=None, n_features=None):
        '''
        :parameters:
            f: function
                activation function
            W: matrix(array)
                n_neurons*n_features weight matrix
            b: vector
                n_neurons*1 bias
            n_neurons: integer
                number of current layer's neurons
            n_features: integer
                number of previous layer's neurons
        '''
        self.W = W
        self.f = f
        self.b = b
        self.n_neurons = n_neurons
        self.n_features = n_features
        self.next_layer = None

    def _init(self):
        '''
        init parameters
        '''
        self.W = self.W if self.W else np.zeros((self.n_neurons, self.n_features))
        self.b = self.b if self.b is not None else 0
        self.f = self.f if self.f else utils.sigmoid
        # create neurons
        self.neurons = [
            Neuron(tolayer=self,
                index=i) for i in range(self.n_neurons) ]

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer


    def get_activations(self, X):
        '''
        :parameters:
            X: vector
                input
                length: n_neurons of the previous layer
        output: array
            activations of current layer
            length: n_neurons

        '''
        #v = self.W * X.T + self.b
        return np.array([self.neurons[i].activate() for 
                    i in range(self.n_neurons)])
        #return self.f(v)


class Network(object):
    '''
    base class of a neural network
    '''
    Layer = Layer

    def __init__(self, X):
        '''
        :parameters: 
            X: array 
            input
        '''
        self.X = X

    def init_layers(self, n_neurons):
        '''
        :parameters:
            n_neurons: list of integer
            number of each hidden layer's neurons
        '''
        len_inputs = len(self.X)
        self.n_neurons = [len_inputs] + n_neurons
        self.layers = [
            self.Layer(
                n_neurons=n_neurons[i+1],
                n_features=n_neurons[i])
            for i in range(len(n_neurons))]
        for i in xrange(len(n_neurons)):
            self.layers[i].set_next_layer(self.layers[i+1])

    def output(self):
        pass


if __name__ == "__main__":
    f = utils.tanh
    W = np.array([0.5, 0.5, 0.5])
    b = 0.1
    neu = Neuron(f, W, b)
    X = np.array([-0.7, 0.2 , 0.1])
    print neu.activate(X)
