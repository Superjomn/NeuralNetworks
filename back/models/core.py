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

alpha = 0.01

class BaseLayer(object):
    def set_next_layer(self, layer):
        self.next_layer = layer

    def set_last_layer(self, layer):
        self.last_layer = layer

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

class InputLayer(BaseLayer):
    def __init__(self, X=None):
        if not X is None:
            self.set_input(X)

    def set_input(self, X):
        self.X = X

    def _init(self):
        length = len(self.X)
        self.W = np.identity(length)
        self.b = np.zeros(length)

    def forward(self):
        pass

    def backward(self):
        pass


class HiddenLayer(BaseLayer):
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
        self._z = None
        self._a = None
        self._init()

    def forward(self, X):
        '''
        :parameters:
            X: vector
                input
                length: n_neurons of the previous layer
        output: array
            activations of current layer
            length: n_neurons

        '''
        self.b = self.b.reshape(self.n_neurons, 1)
        self._z = np.sum(self.W * X + \
                self.b * np.ones((1, self.n_features)),
                    axis=1)
        self._a = self.f(self._z)
        return self._a

    def backward(self, upl_cost):
        '''
        :parameters:
            upl_cost: vector
                theta of upper layer
        '''
        par_W = np.dot(
            upl_cost.reshape(self.n_neurons, 1), 
            self.a.reshape(1, self.n_neurons))
        #self.a.reshape((self.n_neurons, 1)) * (upl_cost.T * np.ones((self.n_neurons, self.n_features)))
        print 'par_w'
        print par_W
        par_b = upl_cost

        self.W -= alpha * par_W 
        self.b -= alpha * np.squeeze(par_b)
        cost = np.array(np.sum(self.W * upl_cost) * \
                utils.sigmoid_der(self.z))
        cost = cost.reshape((1, self.n_neurons))

        return cost


    @property
    def z(self):
        return self._z

    @property
    def a(self):
        return self._a

    def _init(self):
        '''
        init parameters
        '''
        self.W = self.W if self.W else np.random.random((self.n_neurons, self.n_features))
        self.b = self.b if self.b is not None else np.random.random( self.n_neurons)
        self.f = self.f if self.f else utils.sigmoid

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



class OutputLayer(HiddenLayer):
    def __init__(self, W=None, f=None, b=None, n_neurons=None, n_features=None):
        HiddenLayer.__init__(self, W, f, b, n_neurons, n_features)

    def forward(self, X):
        return X

    def backward(self, oY):
        '''
        Y: vector
            the neural forward output
        oY: vector
            the labels
        '''
        cost = -(oY - self.a) * utils.sigmoid_der(self.z)
        self.par_W = self.a * cost
        self.par_b = cost
        self.W -= alpha * self.par_w
        self.b -= alpha * self.par_b
        return cost


class Network(object):
    '''
    base class of a neural network
    '''
    HiddenLayer = HiddenLayer
    OutputLayer = OutputLayer

    def __init__(self, X=None):
        '''
        :parameters: 
            X: array 
            input
        '''
        self.X = X

    def set_input(self, X):
        self.X = X

    def init_layers(self, n_neurons):
        '''
        :parameters:
            n_neurons: list of integer
            number of each hidden layer's neurons
        '''
        len_inputs = len(self.X)
        self.n_neurons = [len_inputs] + n_neurons
        # add hidden layers
        self.layers = [
            self.HiddenLayer(
                n_neurons=n_neurons[i+1],
                n_features=n_neurons[i])
            for i in range(len(n_neurons)-1)]
        # add output layer
        self.layers.append(
            OutputLayer(
                n_neurons=n_neurons[-2],
                n_features=n_neurons[-1]
                ))
        for i in xrange(len(n_neurons)-1):
            self.layers[i].set_next_layer(self.layers[i+1])

    def forward(self):
        # the first hidden layer
        layer = self.layers[0]
        output = self.X
        while layer:
            output = layer.forward(output)
            layer = layer.next_layer
        return output

    def backward(self, Y, oY):
        output_layer = self.layers[-1]
        Y = output_layer.a
        self.network.reverse()
        cost = None
        for layer in self.layers:
            if cost is None:
                cost = Y
            cost = layer.backward(cost)
        self.layers.reverse()


if __name__ == "__main__":
    def test_neuron():
        f = utils.tanh
        W = np.array([0.5, 0.5, 0.5])
        b = 0.1
        neu = Neuron(f, W, b)
        X = np.array([-0.7, 0.2 , 0.1])
        print neu.activate(X)

    def test_layer():
        print 'test_layer'
        layer = HiddenLayer(n_neurons=3, n_features=6)
        layer.show()
        print 'layer', layer
        X = np.array([0.5, 0.3, 1, 0.7, 1, 1])
        print layer.forward(X)

    def test_network():
        network = Network()
        X = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0])
        network.set_input(X)
        n_neurons = [
            len(X), 8, 6, 2, 1 ]
        network.init_layers(n_neurons)
        print '-' * 50
        print "show network layers"
        for layer in network.layers:
            layer.show()
        print "<" * 50
        print network.forward()


    print 'test_layer'
    test_layer()
    test_network()
