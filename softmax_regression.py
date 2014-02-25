#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Feb 22, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import theano
import numpy
from theano import tensor as T

rng = numpy.random
#theano.config.compute_test_value = 'warn'
theano.config.compute_test_value = 'off'
#theano.config.device = 'cpu'
theano.config.floatX = 'float32'
theano.config.mode = 'FAST_COMPILE'


class SoftmaxRegression(object):
    def __init__(self, input=None, n_features=500, n_states=10, learning_rate=0.01):
        self.n_features = n_features
        self.n_states = n_states
        # x is a vector
        self.x = input
        if not self.x:
            self.x = T.fvector('x')

        # y is a label(0 1 2 3 ..)
        self.y = T.bscalar('y')
        # test value
        #self.x.tag.test_value = rng.random(n_features).astype(
        #    theano.config.floatX)
        #self.y.tag.test_value = 3
        #self.y = T.cast(y, 'int32')
        
        self.b = theano.shared(numpy.zeros((n_states)),
            name = 'b',
            borrow = True,
            )
        self.W = theano.shared(
            value = numpy.zeros(( n_states, n_features),
                dtype=theano.config.floatX),
            name = 'W',
            borrow = True,
            )
        self.p_y_given_x = T.nnet.softmax(
                 T.dot(self.W, self.x)  + self.b)
        # get the max index
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.get_y_pred = theano.function(
            inputs = [self.x],
            allow_input_downcast=True,
            outputs = self.y_pred)

        self.learning_rate = learning_rate

        self.params = [self.W, self.b]


    def negative_log_likelihood(self, y):
        '''
        y is the right target
        '''
        self.get_p_y_given_x = \
            theano.function(
                inputs = [self.x],
                allow_input_downcast=True,
                outputs = self.p_y_given_x)

        # - likelihood
        loss = - T.mean(T.log(self.p_y_given_x[0][y]))
        return loss

    def errors(self, y):
        '''
        '''
        # neq : not equal
        return T.mean(T.neq(self.y_pred, y))


    def train(self, set_x, set_y, n_trainset=400, n_iters=1000):
        cost = self.negative_log_likelihood(self.y) + 0.01 * T.mean(self.W ** 2)
        gW = T.grad(cost, self.W)
        gb = T.grad(cost, self.b)

        updates = [
            (self.W, self.W - self.learning_rate * gW),
            (self.b, self.b - self.learning_rate * gb)]



        train_model = theano.function(
            inputs = [self.x, self.y],
            updates = updates,
            allow_input_downcast=True,
            outputs = cost)

        #theano.printing.pydotprint(train_model)
        for no in xrange(n_iters):
            costs = []
            y_preds = []
            for i in xrange(n_trainset):
                x = numpy.array(set_x[i]).astype(
                        theano.config.floatX)
                y = set_y[i]
                #print 'x', x
                #print 'y', y
                #print 'get_p_y_given_x'
                #print self.get_p_y_given_x(x)
                cost = train_model(x, y)
                if no == (n_iters - 1):
                    y_pred = self.get_y_pred(x)
                    y_preds.append(y_pred)
                costs.append(cost)

            c = numpy.array(costs).mean()
            print no, c
        print 'target :10'
        print set_y[:30]
        print [y[0] for y in y_preds[:30]]
        print no, c


if __name__ == "__main__":
    x_set = rng.randn(400, 300).astype(theano.config.floatX)
    y_set = rng.randint(size=400, low=0, high=10).astype(theano.config.floatX)
    
    s = SoftmaxRegression(
        n_features=300)
    s.train(x_set, y_set, n_iters=400)



