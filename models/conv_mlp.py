# -*- coding: utf-8 -*-
'''
Created on March 04, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import sys
sys.path.append('..')
import theano
import numpy
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from mlp import HiddenLayer
from softmax_regression import BatchSoftmaxRegression

class LeNetConvPoolLayer(object):
    '''
    Pool Layer of a convolutional network 
    '''
    def __init__(self, 
            input,
            filter_shape,
            image_shape=(800, 1, 28, 28,),
            rng=None, 
            pool_size=(2,2)):
        '''
        image_shape: 
            [mini-batch size, number of input feature maps, image height, image width]

        filter_shape: 
            [number of feature maps at layer m, number of feature maps at layer m-1, filter height, filter width]
        '''
        self.input = input
        self.rng = rng
        if not self.rng:
            self.rng = numpy.random.RandomState(23455)

        fan_in = numpy.prod(filter_shape[1:])
        # size of the output_after pooling
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(pool_size))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=pool_size, ignore_border=True)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class ConvMLP(object):
    def __init__(self, 
            rng=None,
            image_shape=(28,28),
            batch_size=40,
            n_kerns=(),
            learning_rate = 0.02,
            ):
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.image_shape = image_shape
        self.n_kerns = n_kerns
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rng = rng
        if not self.rng:
            self.rng = numpy.random.RandomState(23455)
        # init
        self.init_layers()

    def init_layers(self):

        layer0_input = self.x.reshape(
                (self.batch_size, 1, 28, 28))
        layer0 = LeNetConvPoolLayer(
                rng=self.rng, input=layer0_input,
                image_shape=(self.batch_size, 1, 28, 28),
                filter_shape=(20, 1, 5, 5), 
                        pool_size=(2, 2)
            )
        layer1 = LeNetConvPoolLayer(rng=self.rng, 
                input=layer0.output,
                image_shape=(self.batch_size, 20, 12, 12),
                filter_shape=( 50, 20, 5, 5), 
                    pool_size=(2, 2))

        layer2_input = layer1.output.flatten(2)
        layer2 = HiddenLayer(rng=self.rng, 
                input=layer2_input, n_visible=50 * 4 * 4,
                             n_output=500, activation=T.tanh)

        layer3 = BatchSoftmaxRegression(input=layer2.output, n_features=500, n_states=10)

        self.layers = [layer0, layer1, layer2, layer3]

    def get_updates(self):
        params = []
        cost = self.layers[-1].negative_log_likelihood(self.y)
        for layer in self.layers:
            params += layer.params
        grads = T.grad(cost, params)
        updates = []
        for param, grad in zip(params, grads):
            update = param - self.learning_rate * grad
            updates.append(
                (param, update,))
        return updates

    def compile_train_fn(self):
        cost = self.layers[-1].negative_log_likelihood(self.y)
        updates = self.get_updates()
        train_model = theano.function(
            [self.x, self.y], cost, updates=updates
            )
        return train_model

    def train(self, dataset, n_iters):
        records, labels = dataset
        n_records = records.shape[0]
        n_batches = int(n_records / self.batch_size)
        train_fn = self.compile_train_fn()
        for no in xrange(n_batches):
            costs = []
            for i in xrange(n_records):
                x = records[i*self.batch_size: (i+1)*self.batch_size]
                y = labels[i*self.batch_size: (i+1)*self.batch_size]
                if not len(x) > 1:
                    continue
                cost = train_fn(x, y)
                costs.append(cost)
            c = numpy.mean(costs)
            print '%d\tcost:%f' % (no, c)



if __name__ == '__main__':
    import cPickle as pickle

    def load_dataset(path):
        '''
        load (records, labels) from pickle file
        '''
        print 'load dataset from:\t', path
        with open(path, 'rb') as f:
            data = pickle.load(f)
        records, labels = data
        #print 'r:', records[0]
        #print 'l:', labels[0]
        return data

    dataset = load_dataset(
        '/home/chunwei/Lab/NeuralNetworks/apps/126/data/sample-3000.pk')
    cov_mlp = ConvMLP(
        n_kerns=(20, 50),
            )
    cov_mlp.train(dataset, 10)


