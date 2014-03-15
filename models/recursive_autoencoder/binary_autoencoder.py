#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 14, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import sys
sys.path.append('..')
sys.path.append('../..')
import theano
from theano import tensor as T
import numpy
from tree import BaseNode
from exec_frame import BaseModel
from recursive_autoencoder import BinaryAutoencoder

class BinaryNode(BaseNode):
    def __init__(self):
        # index to determine wheather to update vectors
        self.pred_index = 0

class BinaryTree(object):
    def __init__(self, root, ae):
        self.root = root
        # autoencoder to merge two children's vectors  to paraent's vector
        self.ae = ae

    def update_vec(self):
        '''
        update each node's vector
        '''
        node = self.root
        self.root.index += 1
        # both left and right child should exist
        hidden_fn = self.ae.get_hidden_fn()
        def update(node):
            lvec = self.get_vec(node.lchild, self.root.index)
            rvec = self.get_vec(node.rchild, self.root.index)
            vec = numpy.append(lvec, rvec)
            hidden = hidden_fn(vec)
            node.vector = hidden
            node.index += 1

    def get_vec(self, node, index):
        if node.is_leaf() or node.index == index:
            return node.vector

        lvec = self.get_vec(node.lchild, index)
        rvec = self.get_vec(node.rchild, index)
        x = numpy.append(lvec, rvec)

        node.vector = self.ae.hidden_fn(x)
        node.index += 1
        
    def get_vec_batch(self, index):
        '''
        :parameters:
            index: int 
            
        returns:
            matrix
        '''
        children_vecs = []
        def get_children_vec(node):
            if node.is_leaf():
                return None
            children_vecs.append( 
                (node.lchild, node.rchild))
            get_children_vector(node.lchild)
            get_children_vector(node.rchild)




class BinaryTreeAutoencoder(object):
    '''
    a binary tree format autoencoder
    using a BinaryAutoencoder to re-construct input
    '''
    def __init__(self, numpy_rng=None, input=None, 
            len_vector=8,
            alpha=0.001, learning_rate=0.01,
            W=None, bhid=None, bvis=None):

        self.bae = BinaryAutoencoder(
                numpy_rng = numpy_rng,
                input = input,
                len_vector = len_vector,
                alpha = alpha,
                learning_rate = learning_rate,
                W = W,
                bhid = bhid, bvis = bvis)

    def train_by_tree(self, tree):
        vecs = []
        index = 0
        def get_children_vector(node):
            lvector = get_children_vector(node.lchild)
            rvector = get_children_vector(node.rchild)
            vec = numpy.append(lvector, rvector)
            vecs.append(vec)
            return vec 

    def get_cost_updates(self):
        pass

