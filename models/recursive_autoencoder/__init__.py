#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 12, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com

Implementation of Recursive Autoencoder

For detail, read <R. Socher, J. Pennington, E. H. Huang, A. Y. Ng, and C. D. Manning, "Semi-Supervised Recursive Autoencoder for Predicting Sentiment Distributions">
'''
import sys
sys.path.append('..')
import theano
from theano import tensor as T
import numpy
from exec_frame import BaseModel


class BinaryAutoencoder(BaseModel):
    '''
    autoencoder which input are two vectors
    and the cost function will be based on the two vectors' 
    reconstruction errors
    '''
    def __init__(self, numpy_rng=None, input=None, 
            len_vector=8,
            alpha=0.001, learning_rate=0.01,
            W=None, bhid=None, bvis=None):
        '''
        :parameters:
            input: tensor of concation of two vectors
            alpha: weight of sturctural cost
        '''
        # the n_visible is len_vector * 2
        self.len_vector = len_vector
        n_visible = len_vector * 2
        n_hidden = len_vector

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
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.x = input
        if not self.x:
            self.x = T.fvector(name='x')
        # count of left's children
        self.lcount = T.bscalar('c1')
        # count of right's children
        self.rcount = T.bscalar('c2')
        
        # compiled functions
        self._train_fn = None
        self._predict_fn = None
        self._hidden_fn = None

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)
    
    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        # vectors of original input
        c1 = self.x[0, 0:self.len_vector]
        c2 = self.x[0, self.n_visible:]
        # reconstruction of two vectors
        _c1 = z[0, 0:self.len_vector]
        _c2 = z[0, self.n_visible:]
        # weight of left vector
        lw = (self.lcount + 0.0) / (self.lcount + self.rchild)

        L = T.sqrt(T.sum( 
            lw * (c1 - _c1) ** 2 + \
            (1 - lw) * (c2 - _c2) ** 2))  \
                + self.alpha * T.sum((self.W ** 2))
        # mean cost of all records
        #sparcity_cost = y 
        cost = T.mean(L) 

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            update = param - self.learning_rate * gparam
            update = T.cast(update, theano.config.floatX)
            updates.append((param, update))
        return cost, updates


    @property
    def hidden_fn(self):
        if not self._hidden_fn:
            self._hidden_fn = theano.function(
                [self.x],
                T.nnet.sigmoid(T.dot(self.x, self.W) + self.b))
        return self._hidden_fn


    @property
    def train_fn(self):
        cost, updates = self.get_cost_updates(learning_rate=0.1)
        if not self._train_fn:
            self._train_fn = theano.function(
                    [self.x, self.lcount, self.rcount], 
                    cost, updates=updates)
        return self._train_fn

    @property
    def predict_fn(self):
        if not self._predict_fn:
            cost, updates = self.get_cost_updates(learning_rate=0.1)
            hidden_value = self.get_hidden_values(self.x)

            self._predict_fn = theano.function(
                    [self.x, self.lcount, self.rcount],
                    [hidden_value, cost])
        return self._predict_fn

    def train_iter(self, x, lcount, rcount):
        '''
        one iteration of the training process

        :parameters:
            x: the concatenation of two vectors(left and right)
            lcount: count of left node's children
            rcount: count of right node's children
        '''
        cost = self.train_fn(x, lcount, rcount)
        return cost

    def predict(self, x, lcount, rcount):
        '''
        :returns:
            hidden_value
            cost
        '''
        return self.predict_fn(x, lcount, rcount)


