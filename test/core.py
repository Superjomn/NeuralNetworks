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







if __name__ == "__main__":
    unittest.main()

