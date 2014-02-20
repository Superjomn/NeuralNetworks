#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import sys
import unittest
import numpy as np
sys.path.append('..')
from models.core import *
from models.utils import *


class TestNeuron(unittest.TestCase):
    def setUp(self):
        pass

    def test_single_active(self):
        X = np.array([1, 0, 1, 1, 0, 0])
        print 'X', X
        n_features = len(X)
        f = sigmoid
        W = np.random.random((1, n_features))
        b = 0.1

        neuron = Neuron(f, W, b)
        neuron.activate(X)
        self.assertTrue((W == neuron.W).all())
        self.assertTrue(b == neuron.b)
        print 'a', neuron.a
        print 'z', neuron.z
        self.assertEqual(neuron.z, np.sum(W * X)+b)
        self.assertEqual(neuron.a, sigmoid(np.sum(W * X)+b))

    def test_layer_active(self):
        X = np.array([1, 0, 1, 1, 0, 0])
        layer = HiddenLayer(n_neurons=1, n_features=len(X))
        layer.show()
        neuron = layer.neurons[0]
        neuron.show()
        W = layer.W
        b = layer.b
        self.assertTrue((W[0] == neuron.W).all())
        self.assertTrue((b[0] == neuron.b).all())



class TestHiddenLayer(unittest.TestCase):

    def setUp(self):
        self.n_neurons = 3
        self.X = np.array(range(5))
        self.n_features = len(self.X)
        self.layer = HiddenLayer(n_neurons=self.n_neurons, n_features=len(self.X))

    def test_n_dimensions(self):
        '''
        check demensions
        '''
        W = self.layer.W
        b = self.layer.b
        # W's row number = n_neurons
        self.assertEqual(W.shape[0], self.n_neurons)
        # w's column number = n_features
        self.assertEqual(W.shape[1], self.n_features)
        # b's length = n_neurons
        self.assertEqual(W.shape[0], len(b))
        self.layer.forward(self.X)

    def test_neuron_status(self):
        W = self.layer.W
        b = self.layer.b
        neurons = self.layer.neurons
        self.assertEqual(len(neurons), self.n_neurons)
        for i,neuron in enumerate(neurons):
            self.assertTrue((neuron.W == W[i]).all())
            self.assertTrue(neuron.b == b[i])

    def test_forward(self):
        W = self.layer.W
        b = self.layer.b
        self.layer.forward(self.X)
        zs = []
        for i in range(self.n_neurons):
            zs.append(
                np.sum(W[i] * self.X)+ b[i])
        zs = np.array(zs)
        print 'zs'
        print zs
        print 'z'
        print self.layer.z
        self.assertTrue(
            (zs == self.layer.z).all())
        _as = sigmoid(zs)
        self.assertTrue(
            (_as == self.layer.a).all())

    def test_backward(self):
        upl_cost = self.X - 1



        









if __name__ == "__main__":
    unittest.main()

