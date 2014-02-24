#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Aug 9, 2013

@author: Chunwei Yan @ pkusz
@mail:  yanchunwei@outlook.com
'''
import theano
import numpy
from theano import tensor as T
rng = numpy.random

class LogisticRegression(object):
    '''
    pass in the dataset as a matrix
    '''
    def __init__(self, n_features):
        self.n_features = n_features
        self.x = T.matrix("x")
        self.y = T.vector("y")
        self.W = theano.shared(rng.randn(n_features).astype(theano.config.floatX), name="W")
        self.b = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), name="b")

    def grad(self):
        p_1 = 1/ (1 + T.exp(-T.dot(self.x, self.W) - self.b))
        self.prediction = p_1 > 0.5
        self.xent = - self.y * T.log(p_1) - (1 - self.y) * (1 - p_1) 
        cost = self.xent.mean() + 0.01 * ( self.W ** 2).sum()
        gw, gb = T.grad(cost, [self.W, self.b])
        return gw, gb

    def compile(self):
        gw, gb = self.grad()
        self.trainer = theano.function(
                inputs = [self.x, self.y],
                outputs = [self.prediction, self.xent],
                updates = [ (self.W, self.W - 0.01 * gw),
                            (self.b, self.b - 0.01 * gb)],
                name = 'train')
        self.predict = theano.function( 
                inputs = [self.x],
                outputs = self.prediction,
                name = 'predict')

    def test(self):
        D = (rng.randn(400, self.n_features).astype(theano.config.floatX),
                rng.randint(size=400, low=0, high=2).astype(theano.config.floatX))
        training_steps = 5000

        for i in range(training_steps):
            pred, err = self.trainer(D[0], D[1])
            print 'error:', numpy.sum(err ** 2) / len(D[0])

        print "target values for D"
        print D[1]

        print "prediction on D"
        print self.predict(D[0])


class LogisticRegressionOne(object):
    '''
    pass in one record each time
    '''
    def __init__(self, n_features):
        self.n_features = n_features
        self.x = T.fvector("x")
        self.y = T.bscalar("y")
        self.W = theano.shared(rng.randn(n_features).astype(theano.config.floatX), name="W")
        self.b = theano.shared(numpy.asarray(0., dtype=theano.config.floatX), name="b")

    def grad(self):
        p_1 = 1/ (1 + T.exp(-T.dot(self.x, self.W) - self.b))
        self.prediction = p_1 > 0.5
        self.xent = - self.y * T.log(p_1) - (1 - self.y) * (1 - p_1) 
        cost = self.xent.mean() + 0.01 * ( self.W ** 2).sum()
        gw, gb = T.grad(cost, [self.W, self.b])
        return gw, gb

    def compile(self):
        gw, gb = self.grad()
        self.trainer = theano.function(
                inputs = [self.x, self.y],
                outputs = [self.prediction, self.xent],
                updates = [ (self.W, self.W - 0.01 * gw),
                            (self.b, self.b - 0.01 * gb)],
                allow_input_downcast=True,
                name = 'train')
        self.predict = theano.function( 
                inputs = [self.x],
                outputs = self.prediction,
                allow_input_downcast=True,
                name = 'predict')

    def test(self):
        data, label = (rng.randn(400, self.n_features).astype(theano.config.floatX),
                rng.randint(size=400, low=0, high=2).astype(theano.config.floatX))
        training_steps = 5000

        for i in range(1000): 
            errs = []
            for i in range(400):
                test_data, test_label = (numpy.array(data[i])).astype(theano.config.floatX), label[i]
                pred, err = self.trainer(test_data, test_label)
                errs.append(err)
            print 'err:', numpy.array(errs).sum() / 400



        



if __name__ == "__main__":
    l = LogisticRegressionOne(784)
    l.compile()
    l.test()
