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

class AutoEncoder(object):
    def __init__(self, numpy_rng=None, input=None, n_visible=8, n_hidden=4,
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
                    size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W')

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, 
                dtype=theano.config.floatX),
                name='bvis')

        if not bhid:
            bhid = theano.shared(value = numpy.zeros(n_hidden,
                dtype=theano.config.floatX),
                name='bhid')
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        if input == None:
            #self.X = T.dmatrix(name='input')
            self.X = T.vector(name='X', dtype=theano.config.floatX)
        else:
            self.X = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, learning_rate):
        y = self.get_hidden_values(self.X)
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
        trainer = theano.function([self.X], cost, updates=updates)
        for i in range(n_iters):
            costs = []
            for j in range(n_items):
                d = data[j]
                cost = trainer(d)
                costs.append(cost)
            print i, 'cost', numpy.mean(numpy.array(costs))






if __name__ == "__main__":
    rng = numpy.random
    def train_all():
        autoencoder = AutoEncoder(numpy_rng=numpy.random.RandomState(1234), n_visible=60, n_hidden=30)
        data = rng.randn(400, 60).astype(theano.config.floatX)
        cost, updates = autoencoder.get_cost_updates(learning_rate=0.1)
        train = theano.function([autoencoder.X], cost, updates=updates)
        for i in range(40000):
            cost = train(data)
            print i, 'cost', cost


    def train_one():
        autoencoder = AutoEncoder(numpy_rng=numpy.random.RandomState(1234), n_visible=60, n_hidden=30)
        autoencoder.X = T.vector(name='X', dtype=theano.config.floatX)
        data = rng.randn(400, 60).astype(theano.config.floatX)
        cost, updates = autoencoder.get_cost_updates(learning_rate=0.1)
        train = theano.function([autoencoder.X], cost, updates=updates)
        for i in range(40000):
            costs = []
            for j in range(400):
                d = data[j]
                cost = train(d)
                costs.append(cost)
            print i, 'cost', numpy.mean(numpy.array(costs))

    autoencoder = AutoEncoder(numpy_rng=numpy.random.RandomState(1234), n_visible=60, n_hidden=30)
    data = rng.randn(400, 60).astype(theano.config.floatX)

    autoencoder.train(data)
