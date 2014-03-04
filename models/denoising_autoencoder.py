#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Feb 24, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import numpy
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from autoencoder import AutoEncoder


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, numpy_rng=None, input=None, n_visible=8, n_hidden=4,
            corrupt_level=0.0,
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

        self.corrupt_level = corrupt_level

    def get_corrupted_input(self, input, level=None):
        ''' add noise to input '''
        #return self.theano_rng.binomial(
            #size = input.shape,
            #n = 1, p = 1-level) * input
        if level is None:
            level = self.corrupt_level

        return  self.theano_rng.binomial(
                size=input.shape, n=1,
                p=1 - level,
                dtype=theano.config.floatX) * input

    def get_cost_updates(self, learning_rate, corrupt_level=None):
        tilde_x = self.get_corrupted_input(self.x, corrupt_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        L2 = T.mean(T.sqrt(T.sum((self.x - z)**2, axis=1)))
        cost = L2
        #L2 = L
        #cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            update = T.cast(
                param - learning_rate * gparam,
                theano.config.floatX)
            updates.append((param, update))

        return (cost, updates, L2)

    def compile_train_fns(self, learning_rate=0.01):
        cost, updates, L2 = self.get_cost_updates(
                corruption_level=0.,
                learning_rate=learning_rate
            )
        self.trainer = theano.function(
            [self.x],
            outputs=[cost, L2], 
            updates=updates)

        self.predict = theano.function(
            [self.x],
            outputs = [self.get_reconstructed_input(self.x)],
            )
        return self.trainer, self.predict
    
    def train(self, data=None, n_iters=1000, learning_rate=0.1):
        n_features, n_items = data.shape
        trainer, predict = self.compile_train_fns(learning_rate)

        for i in range(n_iters):
            costs = []
            L2_costs = []
            for j in range(n_items):
                d = data[j]
                print 'origin', d
                print 'predict', predict(d)
                cost, L2_cost = trainer(d)
                costs.append(cost)
                L2_costs.append(L2_cost)
            print i, 'cost', numpy.array(L2_costs).mean(), \
                    numpy.mean(numpy.array(costs))


class BatchDenoisingAutoEncoder(DenoisingAutoEncoder):
    def __init__(self, numpy_rng=None, input=None, n_visible=8, n_hidden=4,
            corrupt_level=0.0,
            W=None, bhid=None, bvis=None, theano_rng=None):

        if not input:
            input = T.dmatrix(name='x')

        DenoisingAutoEncoder.__init__(self,
            numpy_rng = numpy_rng,
            input = input,
            n_visible = n_visible,
            n_hidden = n_hidden,
            corrupt_level = corrupt_level,
            W = W,
            bhid = bhid, bvis = bvis,
            theano_rng = theano_rng
            )

    def train(self, data=None, batch_size=3, 
                n_iters=1000, learning_rate=0.01):

        cost, updates, L2 = self.get_cost_updates(
                corrupt_level = 0.,
                learning_rate=0.01)

        train_da = theano.function(
            [self.x],
            outputs=[cost, L2], 
            updates=updates)

        n_records = data.shape[0]
        n_batchs = int(n_records / batch_size)

        for no in xrange(n_iters):
            costs = []
            L2s = []
            for i in xrange(n_batchs):
                d = data[i*batch_size : (i+1)*batch_size]
                cost, L2 = train_da(d)
                costs.append(cost)
                L2s.append(L2)
            print "%d\t%f\t%f" % (no, numpy.array(costs).mean(), numpy.array(L2s).mean())




if __name__ == "__main__":
    rng = numpy.random
    data = rng.randn(400, 728).astype(theano.config.floatX)

    def test_one():
        auto = DenoisingAutoEncoder(
            n_visible = 728,
            n_hidden = 30,
            )
        auto.train(data,
            n_iters = 10000)

    def test_batch():

        auto = BatchDenoisingAutoEncoder(
            n_visible = 728,
            n_hidden = 1000, 
            )
        auto.train(data, batch_size=256, n_iters=10000)


    test_batch()
    #test_one()
