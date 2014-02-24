#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Feb 24, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import sys
sys.path.append('..')
import numpy
import theano
from theano import tensor as T
from softmax_regression import SoftmaxRegression

class HiddenLayer(object):

    ''' a layer of neurons '''

    def __init__(self, input,  n_visible, n_output, rng, 
            activation=T.tanh, W=None, b=None):

        if not rng:
            rng = numpy.random.RandomState(1234)

        self.rng = rng

        if not W:
            initial_W = numpy.asarray(
                rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_output + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_output + n_visible)),
                    size=(n_visible, n_output)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W')

            if activation == theano.tensor.nnet.sigmoid:
                W *= 4

        if not b:
            b = theano.shared(value=numpy.zeros(n_output, 
                dtype=theano.config.floatX),
                name='b')

        self.X = input
        self.W = W
        self.b = b
        self.n_visible, self.n_output = n_visible, n_output
        self.activation = activation
        self.params = [self.W, self.b]
        # a output hock
        self.output = self.activation(
            T.dot(self.X, self.W) + self.b)


class MultiLayerPerceptron(object):
    def __init__(self, rng=None, input=None, n_visible=100, n_hidden=50, n_output=10,
            L1_reg=0.0, L2_reg=0.001, learning_rate=0.001):
        '''
        :parameters:
            n_visible: int
                number of visible(input) nodes
            n_hidden: int
                number of hidden nodes
        '''
        self.x = input
        self.L1_reg, self.L2_reg = L1_reg, L2_reg

        # methods mapper
        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors

        if not input:
            self.x = T.fvector('x')

        self.hidden_layer = HiddenLayer(
            rng = rng,
            input = input,
            n_visible = n_visible,
            n_output = n_hidden,
            activation = T.tanh)
        
        self.output_layer = SoftmaxRegression(
            input = self.hidden_layer.activate(),
            n_features = n_hidden,
            n_states = n_output,
            )

    def get_cost(self):
        self.y = T.bscalar('y')

        self.L1 = abs(self.hidden_layer.W).sum() \
                + abs(self.output_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.W ** 2).sum() \
                + (self.output_layer.W ** 2).sum()

        self.params = self.hidden_layer.params + self.output_layer.params
        self.cost = self.negative_log_likelihood(self.y) \
            + self.L1_reg * self.L1 \
            + self.L2_reg * self.L2_sqr
        return self.cost

    def compile(self):
        cost = self.get_cost()
        # predict model
        self.predict = theano.function(
            inputs = [self.x, self.y],
            outputs = self.errors(self.y))

        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append(
                (param, param - self.learning_rate * gparam))
        # train model
        self.train_model = theano.function( 
            inputs = [self.x, self.y],
            outputs = self.errors(self.y),
            updates = updates)







if __name__ == "__main__":
    pass

