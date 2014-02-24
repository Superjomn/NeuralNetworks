#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
import numpy
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from autoencoder import AutoEncoder


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, numpy_rng=None, input=None, n_visible=8, n_hidden=4,
            W=None, bhid=None, bvis=None, theano_rng=None):

        AutoEncoder.__init__(self, 
            numpy_rng=numpy_rng,
            input = input, 
            n_visible = n_visible,
            n_hidden = n_hidden,
            W = W,
            bhid = bhid,
            bvis = bvis
            )

        if not theano_rng:
            theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 3))
        self.theano_rng = theano_rng

    def get_corrupted_input(self, input, level=0.1):
        ''' add noise to input '''
        #return self.theano_rng.binomial(
            #size = input.shape,
            #n = 1, p = 1-level) * input
        return self.theano_rng.binomial(size=input.shape, n=1, p=1 - level) * input

    def get_cost_updates(self, learning_rate, currupt_level=0.1):
        corrupt_X = self.get_corrupted_input(self.X, currupt_level)
        #print 'corrupt_X type', corrupt_X.dtype
        y = self.get_hidden_values(corrupt_X)
        #y = self.get_hidden_values(self.X)
        z = self.get_reconstructed_input(y)
        #L = - T.sum(self.X * T.log(z) + (1 - self.X) * T.log(1 - z), axis=1)
        #L = T.sqrt(T.sum( (self.X - z)**2, axis=1))
        L = T.sqrt(T.sum( (self.X - z)**2)) + 0.01 * T.sum((self.W ** 2))
        # mean cost of all records
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return cost, updates
    
    def train(self, data=None, n_iters=1000, learning_rate=0.1):
        n_features, n_items = data.shape
        # compile function
        cost, updates = self.get_cost_updates(learning_rate=0.1)
        #c_data = self.get_corrupted_input(self.X, 0.5)
        #corrupt = theano.function([self.X], c_data)
        #getx = theano.function([self.X], self.X, allow_input_downcast=True)
        #print 'original ...'
        #print getx(data[0])
        #print getx(data[0]).dtype
        #print 'corrupt ...'
        #print corrupt(data[0])
        #print corrupt(data[0]).dtype

        trainer = theano.function([self.X], cost, 
        updates=updates, 
        allow_input_downcast=True)
        for i in range(n_iters):
            costs = []
            for j in range(n_items):
                d = data[j]
                cost = trainer(d)
                costs.append(cost)
            print i, 'cost', numpy.mean(numpy.array(costs))




if __name__ == "__main__":
    print 'floatX', theano.config.floatX
    rng = numpy.random
    data = rng.randn(400, 60).astype('float32')

    auto = DenoisingAutoEncoder(
        n_visible = 60,
        n_hidden = 30 )

    auto.train(data)
