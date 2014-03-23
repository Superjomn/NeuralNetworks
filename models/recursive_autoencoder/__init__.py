# -*- coding: utf-8 -*-
'''
Created on Feb 24, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import theano 
from theano import tensor as T
import numpy 


class Param(object):

    def __init__(self, numpy_rng=None, 
            len_vector=50,
            W1=None, W2=None, W3=None, 
            b1=None, b2=None, b3=None):
        '''
        :parameters:

            W1: forward proporation weight for children
            W2: backward proporation weight for left child
            W3: backward proporation weight for right child
            b1: forward bias for children 
            b2: backward proporation bias for left child
            b3: backward proporation bias for right child
        '''
        self.len_vector = len_vector
        self.numpy_rng = numpy_rng

        self.W1, self.W2, self.W3 = W1, W2, W3
        self.b1, self.b2, self.b3 = b1, b2, b3

        self._init_parameters()


    def _init_model_parameters(self):
        if not self.numpy_rng:
            self.numpy_rng=numpy.random.RandomState(1234)

        if not self.W1:
            self.W1 = theano.shared(
                value = self._initial_random_weight_value(2*self.len_vector, self.len_vector),
                name="W1"
                )
        if not self.W2:
            self.W2 = theano.shared(
                value = self._initial_random_weight_value(self.len_vector, self.len_vector),
                name="W2"
                )
        if not self.W3:
            self.W3 = theano.shared(
                value = self._initial_random_weight_value(self.len_vector, self.len_vector),
                name="W3"
                )
        if not self.b1:
            self.b1 = theano.shared(value=numpy.zeros(2*self.len_vector, 
                dtype=theano.config.floatX),
                borrow = True,
                name='b1'
                ) 
        if not self.b2:
            self.b2 = theano.shared(value=numpy.zeros(self.len_vector, 
                dtype=theano.config.floatX),
                borrow = True,
                name='b2'
                ) 
        if not self.b3:
            self.b3 = theano.shared(value=numpy.zeros(self.len_vector, 
                dtype=theano.config.floatX),
                borrow = True,
                name='b3'
                ) 
        
    def _initial_random_weight_value(self, n, m):
        initial_W = numpy.asarray(
            self.numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n + m)),
                high=4 * numpy.sqrt(6. / (n + m)),
                size=(n, m)), 
            dtype=theano.config.floatX)
        return initial_W


class Autoencoder(Param):
    def __init__(self, numpy_rng=None, len_vector=50,
            W1=None, W2=None, W3=None, 
            b1=None, b2=None, b3=None):
        Param.__init__(self, 
            numpy_rng = numpy_rng, len_vector = len_vector,
            W1 = W1, W2 = W2, W3 = W3, b1 = b1, 
            b2 = b2, b3 = b3
            )

        # theano functions
        self._hidden_fn = None
        self._reconstruct_fn = None
        self._delta_out1_fn = None
        self._delta_out2_fn = None
        self._Y1C1_fn = None
        self._Y2C2_fn = None
        # theano parameters
        self.x = T.fvector(name='x')
        # parent's feature
        self.p = T.fvector(name='p')
        self.c1 = T.fvector(name='c1')
        self.c2 = T.fvector(name='c2')

        self.f = T.tanh
        self.f_der = T.grad(self.f(self.x), self.x)


    def get_hidden_value(self):
        return T.tanh(
            T.dot(self.c1, self.W1) + \
            T.dot(self.c2, self.W2) + self.param.b1)

    def get_recons_value(self):
        '''
        reconstruction
        '''
        y1 = self.f(self.W3 * self.p + self.b2)
        y2 = self.f(self.W4 * self.p + self.b3)
        return y1, y2

    @property
    def hidden_fn(self):
        '''
        :parameters:
            x: the concation of both children's features
        '''
        if not self._hidden_fn:
            hidden = self.get_hidden_value()
            self._hidden_fn = theano.function(
                [self.c1, self.c2],
                hidden,
                allow_input_downcast=True,
                )
        return self._hidden_fn

    def delta_out1_fn(self):
        '''
        f'(Y)(Y-C)
        '''
        if not self._delta_out1_fn:
            y1, y2 = self.get_recons_value()
            self._delta_out1_fn = theano.function(
                [self.p, self.c1],
                self.f_der(y1) * (y1-self.c1)
                )
        return self._delta_out1_fn

    def delta_out2_fn(self):
        '''
        f'(Y)(Y-C)
        '''
        if not self._delta_out2_fn:
            y1, y2 = self.get_recons_value()
            self._delta_out2_fn = theano.function(
                [self.p, self.c2],
                self.f_der(y2) * (y2-self.c2)
                )
        return self._delta_out2_fn

    def Y1C1_fn(self):
        '''
        Y1 - C1
        '''
        if not self._Y1C1_fn:
            y1, y2 = self.get_recons_value()
            self._Y1C1_fn = theano.function(
                [self.p, self.c1],
                y1 - self.c1
                )
        return self._Y1C1_fn

    def Y2C2_fn(self):
        '''
        Y2 - C2
        '''
        if not self._Y2C2_fn:
            y1, y2 = self.get_recons_value()
            self._Y2C2_fn = theano.function(
                [self.p, self.c2],
                y2 - self.c2
                )
        return self._Y2C2_fn



class RecursiveAutoencoder(object):

    def __init__(self, len_vector=50, ae = None,):
        self.len_vector = len_vector
        self.ae = ae

    def set_tree(self, root):
        self.root = root

    def forward_prop(self):
        # update entire tree's feature
        self.root.pred_index += 1
        # both left and right child should exist
        def update(node):
            lvec = self.get_vec(node.lchild, self.root.pred_index)
            rvec = self.get_vec(node.rchild, self.root.pred_index)
            if lvec is None:
                return
            #print 'vec', vec
            hidden = self.ae.hidden_fn(lvec, rvec)
            node.vector = hidden
            node.pred_index += 1
            assert node.pred_index == self.root.pred_index
            # reconstruction
            node.delta_out1 = self.ae.delta_out_fn(hidden, lvec)
            node.delta_out2 = self.ae.delta_out_fn(hidden, rvec)
            # store Y1 - C1
            node.Y1C1 = self.ae.Y1C1_fn(hidden, lvec)
            # store Y2 - C2
            node.Y2C2 = self.ae.Y2C2_fn(hidden, lvec)
        # recursively update the entire tree
        update(self.root)
        self.root.pred_index -= 1

    def backward_prop(self):
        Ws = [numpy.zeros((self.len_vector, self.len_vector)),
                self.W1, self.W2
                ]
        stack = []
        stack.append([self.root, 0 , None])
        self.root.parent_delta = numpy.zeros(self.len_vector)
        Y0C0 = numpy.zeros((self.len_vector, 1))
        while len(stack) > 0:
            top = stack.pop()
            current_node, left_or_right, parent_node = top
            YCSelector = [Y0C0, None, None] if \
                    parent_node == None \
                    else  [Y0C0, parent_node.Y1C1, parent_node.Y2C2]
            nodeW = Ws[left_or_right]
            delta = YCSelector[left_or_right]
            
            # put in children
            if not current_node.is_leaf():
                stack.push([current_node.lchild, 1, current_node])
                stack.push([current_node.rchild, 2, current_node])
            A1 = current_node.vector
            ND1, ND2 = current_node.delta_out1, current_node.delta_out2
            PD = current_node.parent_delta
            activation = self.ae.W3.T * ND1 + self.ae.W4.T * ND2 + nodeW.T * PD - delta
            current_delta = self.ae.f_der(A1) * activation
            current_node.lchild.parent_delta = current_delta
            current_node.rchild.parent_delta = current_delta
            # update
            GW1_upd = current_delta * current_node.lchild.vector.T
            GW2_upd = current_delta * current_node.rchild.vector.T
            GW3_upd = ND1 * A1.T
            GW4_upd = ND2 * A1.T
            # change parameters
            self.ae.W1 += GW1_upd
            self.ae.W2 += GW2_upd
            self.ae.W3 += GW3_upd
            self.ae.W4 += GW4_upd


    def get_vec(self, node, pred_index):
        '''
        get a node's pred_indexth updated vector 
        '''
        if not node:
            return
        if node.is_leaf() or node.pred_index == pred_index:
            assert node.vector != None
            return node.vector

        lvec = self.get_vec(node.lchild, pred_index)
        rvec = self.get_vec(node.rchild, pred_index)
        assert lvec is not None
        assert rvec is not None

        node.vector = self.ae.hidden_fn(lvec, rvec)
        node.pred_index += 1
        assert node.pred_index == pred_index
        return node.vector
