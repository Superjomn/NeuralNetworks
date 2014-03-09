#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Feb 23, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import numpy
import theano
from theano import tensor as T

def kl_divergence(p, p_hat):
    term1 = p * T.log(p)
    term2 = p * T.log(p_hat)
    term3 = (1-p) * T.log(1 - p)
    term4 = (1-p) * T.log(1 - p_hat)
    return term1 - term2 + term3 - term4


class AutoEncoder(object):
    def __init__(self, numpy_rng=None, input=None, 
            n_visible=8, n_hidden=4,
            sparsity=0.05, beta=0.001,
            W=None, bhid=None, bvis=None):
        '''
        :parameters:
        input: array(vectors)
            each line is a record vector

        n_visible: integer
            number of visible nodes -- record's dimentions

        n_hidden: integer
            number of hidden nodes -- the hidden code
            as a hign level abstract representation of the
            original code.

        W : float32 matrix
            the weights

        bhid: vector
            hidden bias

        bvis: vector
            visible bias

        sparsity: sparsity parameter

        beta: sparsity weight
        '''
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not numpy_rng:
            numpy_rng=numpy.random.RandomState(1234)

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)), 
                dtype=theano.config.floatX)
            W = theano.shared(
                value=initial_W, 
                name='W'
                )

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, 
                dtype=theano.config.floatX),
                borrow = True,
                name='bvis')

        if not bhid:
            bhid = theano.shared(value = numpy.zeros(n_hidden,
                dtype=theano.config.floatX),
                borrow = True,
                name='bhid')
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.numpy_rng = numpy_rng
        # sparsity
        self.sparsity = sparsity
        self.beta = beta

        self.x = input
        if not self.x:
            self.x = T.fvector(name='x')

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_sparty_cost(self, activations):
        sparsity_level = T.extra_ops.repeat(self.sparsity, self.n_hidden)
        ave_act = activations.mean(axis=0)
        kl_div = kl_divergence(sparsity_level, ave_act)
        return kl_div.sum() * self.beta

    def get_cost_updates(self, learning_rate):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        #L = T.sqrt(T.sum( (self.x - z)**2, axis=1))
        L = T.sqrt(T.sum( (self.x - z)**2)) + 0.01 * T.sum((self.W ** 2))
        # mean cost of all records
        #sparcity_cost = y 
        cost = T.mean(L) 
        # add sparcity
        if self.sparsity != 0.0:
            cost += self.get_sparty_cost(y)

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            update = param - learning_rate * gparam
            update = T.cast(update, theano.config.floatX)
            updates.append((param, update))
        return cost, updates

    def train(self, data=None, n_iters=1000, learning_rate=0.1):
        n_features, n_items = data.shape
        # compile function
        cost, updates = self.get_cost_updates(learning_rate=0.1)
        trainer = theano.function([self.x], cost, updates=updates)
        for i in range(n_iters):
            costs = []
            for j in range(n_items):
                d = data[j]
                cost = trainer(d)
                costs.append(cost)
            print i, 'cost', numpy.mean(numpy.array(costs))


class BatchAutoEncoder(AutoEncoder):
    '''
    a batch version of autoencoder
    should run much faster using CPU or GPU
    '''
    def __init__(self, numpy_rng=None, 
            input=None, 
            n_visible=8, n_hidden=4,
            W=None, bhid=None, bvis=None,
            sparsity=0.05, beta=0.001,
            ):

        if not input:
            input = T.dmatrix(name='x')

        AutoEncoder.__init__(self,
            numpy_rng = numpy_rng,
            input = input,
            n_visible = n_visible,
            n_hidden = n_hidden,
            W = W,
            bhid = bhid,
            bvis = bvis,
            sparsity = sparsity,
            beta = beta,
            )

    def train(self, data=None, n_iters=1000, batch_size=3, learning_rate=0.1):
        n_features, n_items = data.shape
        n_batchs = int(n_features / batch_size)
        # compile function
        cost, updates = self.get_cost_updates(learning_rate=0.1)
        trainer = theano.function([self.x], cost, updates=updates)
        for i in range(n_iters):
            costs = []
            for j in range(n_batchs):
                d = data[batch_size * j : batch_size * (j+1)]
                cost = trainer(d)
                costs.append(cost)
            print i, 'cost', numpy.mean(numpy.array(costs).mean())

    def train_iter(self, x, learning_rate=0.02):
        '''
        one iteration of training

        :parameters:
            x: a row of data matrix
        '''
        if not self.train_fn:
            cost, updates = self.get_cost_updates(learning_rate=0.1)
            self.train_fn = theano.function([self.x], cost, updates=updates)
        cost = self.train_fn(x)
        return cost



if __name__ == "__main__":
    rng = numpy.random

    autoencoder = BatchAutoEncoder(numpy_rng=numpy.random.RandomState(1234), n_visible=60, n_hidden=30)
    #print 'autoencoder.W', type(autoencoder.W)
    data = rng.randn(400, 60).astype(theano.config.floatX)

    autoencoder.train(data, batch_size=64, n_iters=100)
