# -*- coding: utf-8 -*-
'''
Created on Feb 24, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import sys
sys.path.append('..')
import unittest
import numpy

from models.recursive_autoencoder import BinaryAutoencoder


class TestBinaryAutoencoder(unittest.TestCase):

    def setUp(self):
        self.bae = BinaryAutoencoder(
            len_vector = 8,
            )

    def test_get_cost_updates(self):
        self.bae.get_cost_updates()

    def test_hidden_fn(self):
        x = [0.1 for i in range(16)]
        x = numpy.array(x, dtype='float32')
        print 'hidden:', self.bae.hidden_fn(x)

    def test_train_fn(self):
        x = [0.1 for i in range(16)]
        x = numpy.array(x, dtype='float32')
        lcount = 4
        rcount = 12
        for i in range(100):
            print 'cost', self.bae.train_fn(x, lcount, rcount)

    def test_predict_fn(self):
        x = [0.1 for i in range(16)]
        x = numpy.array(x, dtype='float32')
        lcount = 4
        rcount = 12
        for i in range(100):
            hidden, cost = self.bae.predict_fn(x, lcount, rcount)
            print 'cost', cost

    def test_train_iter(self):
        print 'test train iter ...'
        x = [0.1 for i in range(16)]
        x = numpy.array(x, dtype='float32')
        lcount = 4
        rcount = 12
        for i in range(100):
            cost = self.bae.train_iter(x, lcount, rcount)
            print 'cost', cost




if __name__ == "__main__":
    unittest.main()
