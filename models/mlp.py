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
            activation=T.tanh, W=None, b=None, learning_rate=0.01):

        if not rng:
            rng = numpy.random.RandomState(1234)

        self.rng = rng
        #print 'n_output, n_visible', n_output, n_visible

        if not W:
            initial_W = numpy.asarray(
                rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_output + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_output + n_visible)),
                    size=(n_visible, n_output)), dtype=theano.config.floatX)

            if activation == theano.tensor.nnet.sigmoid:
                initial_W = numpy.asarray(
                    rng.uniform(
                        low=-16 * numpy.sqrt(6. / (n_output + n_visible)),
                        high=16 * numpy.sqrt(6. / (n_output + n_visible)),
                        size=(n_visible, n_output)), dtype=theano.config.floatX)

            W = theano.shared(
                value=initial_W, 
                name='W', 
                borrow=True,
                )

            T.unbroadcast(W)

        if not b:
            b_values = numpy.zeros((n_output,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.X = input
        self.W = W
        self.b = b
        self.learning_rate = learning_rate
        self.n_visible, self.n_output = n_visible, n_output
        self.activation = activation
        self.params = [self.W, self.b]
        # a output hock
        self.output = self.activation(
            T.dot(self.X, self.W) + self.b)


class MultiLayerPerceptron(object):
    def __init__(self, rng=None, input=None, n_visible=100, n_hidden=50, n_output=10,
            L1_reg=0.0, L2_reg=0.01, learning_rate=0.001):
        '''
        a network with two layers
        :parameters:
            n_visible: int
                number of visible(input) nodes
            n_hidden: int
                number of hidden nodes
        '''
        self.x = input
        self.learning_rate = learning_rate
        self.L1_reg, self.L2_reg = L1_reg, L2_reg

        if not input:
            self.x = T.fvector('x')
        # create two layers
        self.hidden_layer = HiddenLayer(
            rng = rng,
            input = input,
            n_visible = n_visible,
            n_output = n_hidden,
            activation = T.tanh
            )
        self.output_layer = SoftmaxRegression(
            input = self.hidden_layer.output,
            n_features = n_hidden,
            n_states = n_output,
            )

        # methods mapper
        self.negative_log_likelihood = self.output_layer.negative_log_likelihood
        self.errors = self.output_layer.errors

    def get_cost(self):
        self.y = T.bscalar('y')

        self.L1 = abs(self.hidden_layer.W).sum() \
                + abs(self.output_layer.W).sum()

        self.L2_sqr = (self.hidden_layer.W ** 2).sum() \
                + (self.output_layer.W ** 2).sum()

        self.params = self.hidden_layer.params + self.output_layer.params
        self.cost = self.negative_log_likelihood(self.y) \
            + self.L2_reg * self.L2_sqr 
        #+ self.L1_reg * self.L1 
        return self.cost

    def compile(self):
        cost = self.get_cost()
        # predict model
        self.predict = theano.function(
            inputs = [self.x],
            outputs = self.output_layer.y_pred
            )
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = []
        for param, gparam in zip(self.params, gparams):
            up = T.cast(param - self.learning_rate * gparam, 
                    theano.config.floatX)
            updates.append(
                (param, up))
        #print 'updates', updates
        # train model
        self.trainer = theano.function( 
            inputs = [self.x, self.y],
            outputs = self.errors(self.y),
            updates = updates)




if __name__ == '__main__':
    x = T.fvector('x')
    mlp = MultiLayerPerceptron(
            input = x,
            n_visible = 50,
            n_hidden = 20,
            n_output = 5,
            learning_rate = 0.03,
            )
    print 'type of W', type(mlp.hidden_layer.W)
    mlp.compile()
    rng = numpy.random
    x_set = rng.randn(400, 50).astype(theano.config.floatX)
    y_set = rng.randint(size=400, low=0, high=5).astype(theano.config.floatX)
    n_rcds = x_set.shape[0]

    #print 'hid.b:\t', mlp.hidden_layer.b.eval()
    #print 'output.b:\t', mlp.output_layer.b.eval()
    for no in xrange(100):
        errors = []
        y_preds = []
        for i  in xrange(n_rcds):
            x = numpy.array(x_set[i]).astype(
                    theano.config.floatX)
            y = y_set[i]
            y_pred = mlp.predict(x)[0]
            error = mlp.trainer(x, y)
            #print 'error', error
            errors.append(error)
            y_preds.append(y_pred)
        e = numpy.array(errors).mean()
        print "%dth\t%f" % (no, e)
        print "original:\t", y_set[:30]
        print "predict:\t", y_preds[:30]
        #print 'hid.b:\t', mlp.hidden_layer.b.eval()
        #print 'output.b:\t', mlp.output_layer.b.eval()





if __name__ == "__main__":
    pass

