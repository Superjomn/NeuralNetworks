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

        return self.theano_rng.binomial(size=input.shape, n=1, p=1 - level) * input

    def get_cost_updates(self, learning_rate, corrupt_level=0.1):
        tilde_x = self.get_corrupted_input(self.x, corrupt_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        L2 = T.mean(T.sqrt(T.sum((self.x - z)**2, axis=1)))
        cost = L2
        L2 = L
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

    def compile_train_fns(self):
        cost, updates, L2_cost = self.get_cost_updates(learning_rate=0.1)

        self.trainer = theano.function(
            [self.x], outputs=[cost, L2_cost], 
            updates=updates, 
            allow_input_downcast=True)

        self.predict = theano.function(
            [self.x],
            outputs = [self.get_reconstructed_input(self.x)],
            )
        return self.trainer, self.predict
    
    def train(self, data=None, n_iters=1000, learning_rate=0.1):
        n_features, n_items = data.shape
        # compile function
        #c_data = self.get_corrupted_input(self.x, 0.5)
        #corrupt = theano.function([self.x], c_data)
        #getx = theano.function([self.x], self.x, allow_input_downcast=True)
        #print 'original ...'
        #print getx(data[0])
        #print getx(data[0]).dtype
        #print 'corrupt ...'
        #print corrupt(data[0])
        #print corrupt(data[0]).dtype
        trainer, predict = self.compile_train_fns()

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


class BatchDenoisingAutoEncoder2(DenoisingAutoEncoder):
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
                n_iters=1000, learning_rate=0.1):

        n_features, n_records = data.shape
        n_batchs = int(n_records / batch_size)
        # compile function
        cost, updates, L2_cost= self.get_cost_updates(learning_rate=0.1)
        trainer = theano.function([self.x], 
            outputs=[cost, L2_cost],
            updates=updates, 
            allow_input_downcast=True
            )
        for i in range(n_iters):
            costs = []
            L2_costs = []
            for j in range(n_batchs):
                d = data[j * batch_size : (j+1) * batch_size]
                cost, L2_cost = trainer(d)
                #print 'cost', cost
                costs.append(cost.mean())
                L2_costs.append(L2_cost)
            print i, 'cost', numpy.array(costs).mean(), numpy.array(L2_costs).mean() / batch_size


class BatchDenoisingAutoEncoder(object):
    """Denoising Auto-Encoder class (dA)
    """

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        L2 = T.mean(T.sqrt(T.sum((self.x - z)**2, axis=1)))
        cost = L2
        L2 = L
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

    def train(self, data=None, batch_size=3, n_iters=100):

        cost, updates, L2 = self.get_cost_updates(corruption_level=0.,
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
    data = rng.randn(400, 60).astype(theano.config.floatX)

    def test_one():
        print 'floatX', theano.config.floatX

        auto = DenoisingAutoEncoder(
            n_visible = 60,
            n_hidden = 30,
            )
        auto.train(data,
            n_iters = 10000)

    def test_batch():
        print 'floatX', theano.config.floatX

        auto = BatchDenoisingAutoEncoder2(
            n_visible = 60,
            n_hidden = 30, 
            )
        auto.train(data, batch_size=40, n_iters=10000)

    def test_da():
        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        d = dA(
            rng, theano_rng,
            n_visible=60,
            n_hidden=100)
        d.train(data, batch_size=64)

    test_batch()
    #test_one()
    #test_da()
