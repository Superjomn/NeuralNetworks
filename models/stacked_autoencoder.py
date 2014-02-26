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

from mlp import HiddenLayer
from denoising_autoencoder import DenoisingAutoEncoder
from softmax_regression import SoftmaxRegression
from theano.tensor.shared_randomstreams import RandomStreams

# TODO link learning_rate to each layer

class StackedAutoEncoder(object):
    def __init__(self, numpy_rng=None, theano_rng=None, n_visible=30,
                hidden_struct=[400, 300], 
                n_output=10, corrupt_levels=[0.1, 0.1], learning_rate=0.001):
        '''
        :parameters:
            hidden_struct: list of ints
                number of hidden layers' nodes
        '''

        self.n_layers = len(hidden_struct)

        if not numpy_rng:
            numpy_rng=numpy.random.RandomState(1234)

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 3))

        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.n_visible = n_visible
        self.hidden_struct = hidden_struct
        self.n_output = n_output
        self.corrupt_levels = corrupt_levels
        self.learning_rate = learning_rate
        # create variables
        self.x = T.fvector('x')
        self.y = T.bscalar('y')
        # layer lists
        #self.sigmoid_layers = []
        self.dA_layers = []
        self.hidden_layers = []
        self.params = []

        self._init_layers()

    def _init_layers(self):
        for no in xrange(self.n_layers):
            n_visible = self.n_visible if no == 0 \
                    else self.hidden_struct[no-1]
            input = self.x if no == 0 else\
                    self.hidden_layers[no-1].output
            # create a hidden layer
            hidden_layer = HiddenLayer(
                rng = self.numpy_rng,
                input = input,
                n_visible = n_visible,
                n_output = self.hidden_struct[no],
                activation = T.nnet.sigmoid)
            print 'hidden_layer.parameters:'
            print 'W', type(hidden_layer.W)
            print 'b', type(hidden_layer.b)
            self.hidden_layers.append(hidden_layer)
            
            self.params += hidden_layer.params

            # share each hidden layers' parameters 
            # with denoising auto-encoder
            _dA_layer = DenoisingAutoEncoder(
                numpy_rng = self.numpy_rng,     
                input = input,
                n_visible = n_visible,
                n_hidden = self.hidden_struct[no],
                # TODO something wrong?
                corrupt_level = self.corrupt_levels[0],
                W = hidden_layer.W,
                bhid = hidden_layer.b
                )
            self.dA_layers.append(_dA_layer)

        # add a final output layer
        self.output_layer = SoftmaxRegression(
            input = self.hidden_layers[-1].output,
            n_features = self.hidden_struct[-1],
            n_states = self.n_output,
            learning_rate = self.learning_rate
            )

        self.params += self.output_layer.params
    
        # top layer's cost
        self.fineture_cost = \
                self.output_layer.negative_log_likelihood(self.y)

        self.errors = self.output_layer.errors(self.y)

    def compile_pretrain_funcs(self):
        '''
        just give records' record without label
        to generate an abstruct representation of the original
        dataset
        '''
        #corrupt_level = T.scalar('corrupt_level')
        #learning_rate = T.scalar('learning_rate')
        fns = []
        # use autoencoders to encode trainset
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(
                self.corrupt_levels[0], self.learning_rate)

            print 'updates:', [ [a.dtype for a in u] for u in updates]
            print 'hidden_layer.W', dA.W.dtype
            # create function link
            fn = theano.function(
                inputs = [self.x],
                outputs = cost,
                updates = updates,
                allow_input_downcast=True,
                )
            fns.append(fn)
        return fns


    def compile_finetune_funcs(self):
        gparams = T.grad(self.fineture_cost, self.params)
        # updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append(
                (param, param - gparam * self.learning_rate))

        train_fn = theano.function(
            inputs = [self.x, self.y],
            outputs = self.fineture_cost,
            updates = updates
            )

        predict_fn = theano.function(
            inputs = [self.x, self.y],
            outputs = self.fineture_cost
            )
        return train_fn, predict_fn

    def pretrain(self, dataset, n_iters=3):

        pretraining_fns = self.compile_pretrain_funcs()
        # finetune functions
        #ft_train_fn, ft_predict_fn = self.compile_finetune_funcs()

        print 'pretraining ...'
        ### pretraining
        n_records = dataset.shape[0]

        for no in xrange(self.n_layers):
            for t in xrange(n_iters):
                costs = []
                for rid in xrange(n_records):
                    print '> train record:\t%d' % rid
                    x = dataset[rid]
                    c = pretraining_fns[no]( x) 
                    costs.append(c)

                print 'pretraining layer %d\tcost\t%f' % (
                            no, numpy.array(costs).mean()
                        )
            

    def finetune(self, records, labels, n_iters=5):
        '''
        '''
        print '... finetunning the model'
        # TODO scan the parameter space and get the
        # best parameters and stop training
        train_fn, predict_fn = self.compile_finetune_funcs()
        n_records = records.shape[0]
        costs = []
        for t in xrange(n_iters):
            for i in xrange(n_records):
                x, y = records[i], labels[i]
                cost = train_fn(x, y)
                costs.append(cost)
            print 'fineture error:\t%f' % numpy.array(costs).mean()




if __name__ == "__main__":
    stacked_autoencoder = StackedAutoEncoder(
        n_visible = 30,
        hidden_struct = [20, 10],
        n_output = 5,
        learning_rate = 0.01,
            )

    numpy_rng=numpy.random.RandomState(1234)
    data = numpy_rng.randn(400, 30).astype(theano.config.floatX)
    labels = numpy_rng.randint(size=400, low=0, high=5).astype(theano.config.floatX)

    #stacked_autoencoder.init_layers()
    stacked_autoencoder.pretrain(data, n_iters=20)
    stacked_autoencoder.finetune(data, labels, n_iters=20)

