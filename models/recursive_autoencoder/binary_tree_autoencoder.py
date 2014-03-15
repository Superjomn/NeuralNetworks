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
import numpy
import  theano
from    theano import tensor as T
import config
from tree import BinaryNode
from exec_frame import BaseModel
from recursive_autoencoder import BinaryAutoencoder

class BinaryTree(object):
    '''
    used in BinaryTreeAutoencoder
    '''
    def __init__(self, root, ae):
        '''
        :parameters:
            root: object of BinaryNode
                the root of a tree 
        '''
        self.root = root
        assert issubclass(
            type(self.root), BinaryNode)
        # autoencoder to merge two children's vectors  to paraent's vector
        self.ae = ae
        self.len_vector = self.ae.len_vector

        self._init_child_count()

    def _init_child_count(self):
        def child_count(node):
            if node.is_leaf():
                return 0
            node.n_children = child_count(node.lchild) \
                    + child_count(node.rchild) + 1
            return node.n_children

    def update_vec(self):
        '''
        update each node's vector
        '''
        self.root.pred_index += 1
        # both left and right child should exist
        hidden_fn = self.ae.get_hidden_fn()
        def update(node):
            lvec = self.get_vec(node.lchild, self.root.pred_index)
            rvec = self.get_vec(node.rchild, self.root.pred_index)
            vec = numpy.append(lvec, rvec)
            hidden = hidden_fn(vec)
            node.vector = hidden
            node.pred_index += 1
        # recursively update the entire tree
        update(self.root)
        self.root.pred_index -= 1

    def get_vec(self, node, pred_index):
        if node.is_leaf() or node.pred_index == pred_index:
            return node.vector

        lvec = self.get_vec(node.lchild, pred_index)
        rvec = self.get_vec(node.rchild, pred_index)
        x = numpy.append(lvec, rvec)

        node.vector = self.ae.hidden_fn(x)
        node.pred_index += 1
        assert node.pred_index == pred_index
        
    def get_vec_batch(self):
        '''
        predict each node's 2-vector

        :parameters:
            index: int 
            
        returns:
            a list of vectors
        '''
        children_vecs = []
        child_counts = []
        n_child_pairs = 0

        def get_children_vec(node):
            if node.is_leaf():
                return None

            n_child_pairs += 1

            children_vecs.append(
                numpy.append(
                    node.lchild.vector,
                    node.rchild.vector))
            child_counts.append(
                (node.lchild.n_children,
                    node.rchild.n_children))
            # recursively 
            get_children_vec(node.lchild)
            get_children_vec(node.rchild)

        return children_vecs, child_counts

        




class BinaryTreeAutoencoder(object):
    '''
    a binary tree tructure autoencoder
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
        '''
        get a tree's updated 2-vectors as a matrix
        and reconstruct it with the binary autoencoder

        :parameters:
            tree: object of BinaryTree

        :returns:
            batch
        '''
        vecs = []
        vecs, child_counts = tree.get_vec_batch()
        costs = []
        for i,vec in enumerate(vecs):
            n_child = child_counts[i]
            cost = self.bae.train_fn(vec, *n_child)
            costs.append(cost)
        return numpy.mean(cost)

    def get_root_vector(self, tree):
        return tree.root.vector
