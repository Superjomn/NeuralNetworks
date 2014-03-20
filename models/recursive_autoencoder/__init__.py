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
            W=None, W_prime=None, bhid=None, bvis=None):
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

        '''
        if not W_prime:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_hidden, n_visible)), 
                dtype=theano.config.floatX)
            W_prime = theano.shared(
                value=initial_W, 
                name='W'
                )
        '''

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, 
                dtype=theano.config.floatX) + 0.0001,
                borrow = True,
                name='bvis') 

        if not bhid:
            bhid = theano.shared(value = numpy.zeros(n_hidden,
                dtype=theano.config.floatX) + 0.0001,
                borrow = True,
                name='bhid') 

        self.W = W
        #self.W_prime = W_prime
        self.W_prime = W.T
        self.b = bhid
        self.b_prime = bvis
        self.numpy_rng = numpy_rng
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.x = input
        if not self.x:
            self.x = T.fvector(name='x')
        # count of left's children
        self.lcount = T.fscalar('c1')
        # count of right's children
        self.rcount = T.fscalar('c2')
        
        # private compiled functions
        self._forward_train_fn = None
        self._predict_fn = None
        self._update_fn = None
        self._hidden_fn = None
        self._cost_fn = None
        self._back_recon_train_fn = None

        #self.params = [self.W, self.W_prime, self.b, self.b_prime]
        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.tanh(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.tanh(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self):
        '''
        used in forword proporation
        '''
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        # vectors of original input
        c1 = self.x[0:self.len_vector]
        c2 = self.x[self.len_vector:]
        # reconstruction of two vectors
        _c1 = z[0:self.len_vector]
        _c2 = z[self.len_vector:]
        # weight of left vector
        lw = (self.lcount) / (self.lcount + self.rcount)
        self.lw = lw

        self.L = T.sqrt(T.sum( 
            lw *       (c1 - _c1) ** 2 + \
            (1 - lw) * (c2 - _c2) ** 2))  
                #+ self.alpha * T.sum((self.W ** 2))
        #self.L = T.sqrt(T.sum( (z - self.x) ** 2))
        '''
        self.L = T.sqrt(T.sum(
            (c1-_c1)**2 + (c2-_c2)**2
            ))
        '''
        # mean cost of all records
        #sparcity_cost = y 
        #cost = T.mean(L) 
        cost = self.L

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            update = param - self.learning_rate * gparam
            update = T.cast(update, theano.config.floatX)
            updates.append((param, update))
        return cost, updates

    def get_back_recon_cost_updates(self):
        self.pre_lvec = T.fvector(name='pre_lvec')
        self.pre_rvec = T.fvector(name='pre_rvec')
        self.fath_vec = T.fvector(name='fath_vec')

        rec_vec = self.get_reconstructed_input(self.fath_vec)
        _lvec = rec_vec[:self.len_vector]
        _rvec = rec_vec[self.len_vector:]

        lw = self.lcount / (self.lcount + self.rcount)

        L = T.sqrt(T.sum(
            lw *    (self.pre_lvec - _lvec) ** 2 + \
            (1-lw) *(self.pre_rvec - _rvec) ** 2)) 

        cost = L
        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            update = param - self.learning_rate * gparam
            update = T.cast(update, theano.config.floatX)
            updates.append((param, update))
        return cost, updates

    @property
    def cost_fn(self):
        if not self._cost_fn:
            self._cost_fn = theano.function(
                [self.x, self.lcount, self.rcount],
                self.L,
                on_unused_input='ignore',
                )
        return self._cost_fn

    @property
    def hidden_fn(self):
        if not self._hidden_fn:
            self._hidden_fn = theano.function(
                [self.x],
                T.tanh(T.dot(self.x, self.W) + self.b),
                allow_input_downcast=True,
                )
        return self._hidden_fn

    # -------------- two trainning methods ----------------
    @property
    def forward_train_fn(self):
        '''
        local updates with single node's forward and backward
        reconstruction cost
        '''
        if not self._forward_train_fn:
            cost, updates = self.get_cost_updates()
            self._forward_train_fn = theano.function(
                    [self.x, self.lcount, self.rcount], 
                    cost, updates=updates,
                    allow_input_downcast=True,
                    on_unused_input='ignore',
                    )
        return self._forward_train_fn

    @property
    def back_recon_train_fn(self):
        if not self._back_recon_train_fn:
            cost, updates = self.get_back_recon_cost_updates()
            self._back_recon_train_fn = theano.function(
                    [self.fath_vec, 
                        self.pre_lvec, self.pre_rvec,
                        self.lcount, self.rcount],
                    cost,
                    update = updates)
        return self._back_recon_train_fn

    @property
    def predict_fn(self):
        if not self._predict_fn:
            cost, updates = self.get_cost_updates()
            hidden_value = self.get_hidden_values(self.x)
            reconstructed_value = self.get_reconstructed_input(hidden_value)

            self._predict_fn = theano.function(
                    [self.x, self.lcount, self.rcount],
                    [reconstructed_value, cost],
                    allow_input_downcast=True,
                    on_unused_input='ignore',
                    )
        return self._predict_fn

    @property
    def update_fn(self):
        if not self._update_fn:
            cost, updates = self.get_cost_updates()
            gparams = T.grad(cost, self.params)
            self._update_fn = theano.function(
                [self.x, self.lcount, self.rcount],
                gparams+[self.lw],
                on_unused_input='ignore',
                )
        return self._update_fn

    def recon_fn(self):
        pass


    def train_iter(self, x, lcount, rcount):
        '''
        one iteration of the training process

        :parameters:
            x: the concatenation of two vectors(left and right)
            lcount: count of left node's children
            rcount: count of right node's children
        '''
        assert lcount > 0
        assert rcount > 0
        cost = self.train_fn(x, lcount, rcount)
        return cost

    def predict(self, x, lcount, rcount):
        '''
        :returns:
            hidden_value
            cost
        '''
        return self.predict_fn(x, lcount, rcount)


