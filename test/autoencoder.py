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
from models.autoencoder import *
from models.utils import *

class TestBaseLayer(unittest.TestCase):
    def setUp(self):
        self.layer = BaseLayer()

    def test_set_next_layer(self):
        layer1 = BaseLayer()
        layer2 = BaseLayer()
        self.layer.set_lower_layer(layer1)
        self.layer.set_upper_layer(layer2)
        self.assertEqual(self.layer.lower_layer, layer1)
        self.assertEqual(self.layer.upper_layer, layer2)


class TestHiddenLayer(unittest.TestCase):
    def test_forward(self):
        layer1 = HiddenLayer(n_neurons=8, n_features=4, f=sigmoid)
        layer2 = HiddenLayer(n_neurons=4, n_features=2, f=tanh)
        layer2.set_lower_layer(layer1)
        layer1.show()
        layer2.show()
        last_W = layer1.W
        last_b = layer1.b
        X = np.array(range(8))
        layer2.forward(X)
        cur_z = layer2.z
        cur_a = layer2.a
        print 'cur_a', cur_a
        zz = last_W * (X * np.ones((4, 8)))
        ssum = np.sum(zz, axis=1)
        print 'ssum'
        print ssum
        z = ssum + last_b
        self.assertTrue( (cur_z == z).all())
        self.assertTrue( (cur_a == tanh(z)).all())

    def test_backward(self):
        def get_cost(n_neurons, W, upl_cost, z):
            thetas = []
            for i in range(n_neurons):
                print 'get_cost>i', i
                theta_i = sum(\
                    W[j][i] * upl_cost[j] \
                    for j in range(len(upl_cost))) \
                    * sigmoid_der(z[i])
                print 'test theta_i', theta_i

                thetas.append(theta_i)
            thetas = np.array(thetas)
            return thetas

        layer1 = HiddenLayer(n_neurons=8, n_features=4, f=sigmoid)
        layer2 = HiddenLayer(n_neurons=4, n_features=2, f=sigmoid)
        layer2.set_lower_layer(layer1)
        layer1.show()
        layer2.show()
        last_W = layer1.W
        last_b = layer1.b
        X = np.array([1, 0, 0, 1, 1, 0, 1, 1])
        layer2.forward(X)
        cur_z = layer2.z
        cur_a = layer2.a
        cur_cost = np.array([1, 0])
        my_cost = get_cost(
                layer2.n_neurons,
                layer2.W,
                cur_cost,
                layer2.z)
        print 'my_cost'
        print my_cost
        cost = layer2.backward(cur_cost)
        print 'layer2.cost'
        print cost
        self.assertTrue(
            (my_cost == cost).all())


class TestOutputLayer(unittest.TestCase):
    def test_forward(self):
        pass

    def test_backword(self):
        pass




if __name__ == "__main__":
    unittest.main()

