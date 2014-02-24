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

from mlp import HiddenLayer, MultiLayerPerceptron
from denoising_autoencoder import DenoisingAutoEncoder

# TODO link learning_rate to each layer

class StackedAutoEncoder(object):
    def __init__(self, numpy_rng, theano_rng=None, n_visible=30,
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
            theano_rng = RandomStreams(self.numpy_rng.randint(2 ** 3))

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
        self.sigmoid_layers = []
        self.dA_layers = []

    def init_layers(self):
        for no in xrange(self.n_layers):
            n_visible = self.n_visible if no == 0 \
                    else self.hidden_struct[no-1]
            input = self.x if no == 0 else\
                    self.hidden_layers[no-1].output
            # create a hidden layer
            sigmoid_layer = HiddenLayer(
                rng = rng,
                input = input,
                n_visible = n_visible,
                n_output = self.hidden_struct[no],
                activation = T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            
            self.params += sigmoid_layer.params

            # create a denoising autoencoder with shared parameters
            _dA_layer = DenoisingAutoEncoder(
                numpy_rng = self.numpy_rng,     
                input = input,
                n_visible = n_visible,
                n_hidden = self.hidden_struct[no],
                # TODO something wrong?
                corrupt_level = corrupt_level[0],
                W = sigmoid_layer.W,
                bhid = sigmoid_layer.b
                )
            self.dA_layers.append(_dA_layer)

        # add a final output layer
        self.output_layer = SoftmaxRegression(
            input = self.sigmoid_layers[-1].output,
            n_features = self.hidden_struct[-1],
            n_states = self.n_output,
            learning_rate = self.learning_rate
            )

        self.params += self.output_layer.params
    
    def get_cost(self):
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
                self.corrupt_level[0], self.learning_rate)

            # create function link
            fn = theano.function(
                inputs = [self.x, self.y],
                outputs = cost,
                updates = updates
                )
            fns.append(fn)
        return fns


    def compile_finetune_funcs(self, trainset):
        gparams = T.grad(self.fineture_cost, self.params)
        # updates
        for param, gparam in zip(self.params, gparams):
            updates.append(
                (param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs = [self.x, self.y],
            outputs = self.fineture_cost,
            updates = updates
            )

        predict_fn = theano.function(
            inputs = [self.x, self.y],
            outputs = self.fineture_cost
            )

    def train(self, pre_train_set, finetune_set):

        print 'pretraining ...'
        pretraining_fns = self.compile_pretrain_funcs()
        n_records = pre_train_set.shape[0]

        for no in xrange(self.n_layers):
            costs = []
            for rid in xrange(n_records):
                x = pre_train_set[rid]
                c = pretraining_fns[no]( x) 
                costs.append(c)
            print 'pretraining layer %d\tcost\t%f' % (
                        no, numpy.array(costs).mean()
                    )

# TODO fine turn here

            












        

    def _init(self):
        <++ args ++>


if __name__ == "__main__":
    <++ code ++>

