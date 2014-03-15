#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 7, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import sys
sys.path.append('..')
sys.path.append('../..')
import numpy as np
import  theano
from    theano import tensor as T
import config
from tree import BinaryNode
from exec_frame import BaseModel
from models.recursive_autoencoder import BinaryAutoencoder


# ------------------- greedy ---------------------------
class GreedyNode(BinaryNode):
    '''
    base model of the node of Huffman tree
    '''
    def __init__(self, vector=None, lchild=None, rchild=None):
        self.vector = vector
        self.lchild = lchild  
        self.rchild = rchild
        self.n_children = 0 if not lchild \
            else lchild.n_children + rchild.n_children


class GreedyTree(object):
    '''
    base model of Huffman Tree
    '''

    GreedyNode = GreedyNode

    def __init__(self, nodes): 
        self.nodes = nodes
        self.root = None

    def build_tree(self):
        '''
        construct the binary tree representation by selecting
            the tree that offers the minimum cost
        '''
        while len(self.nodes) > 1:
            min_error = np.inf
            j = -1
            new_node = None

            for i in range(len(self.nodes)-1):
                lnode = self.nodes[i]
                rnode = self.nodes[i+1]
                vector, cost = self.get_merge_cost(lnode, rnode)
                if cost < min_error:
                    min_error = cost
                    j = i
                    new_node = self.GreedyNode(vector, lnode, rnode)
            # replace two nodes with the merge node
            self.nodes[j] = new_node
            del self.nodes[j+1]

        self.root = new_node
        return self.root


    def get_merge_cost(self, lnode, rnode):
        '''
        :returns:
            merge error, merge vector
        '''
        raise NotImplemented




class GreedyTreeAutoencoder(BaseModel, GreedyTree):
    '''
    the model form a binary tree representation of a 
        sequence in a greedy way

    use a BinaryAutoencoder which is pretrained with sentences
        and parse trees, 
        params of the BinaryAutoencoder will not be changed.
    '''
    def __init__(self, nodes, len_vector, bae, 
            sparsity=0.05, beta=0.001):
        '''
        :parameters:

            bae: object of BinaryAutoencoder
                bae should be pre-trained using sentence 
                    and parse trees
        '''
        GreedyTree.__init__(self)
        self.bae = bae

    def __call__(self):
        '''
        build the greedy binary tree and return 
        the vector of the sequence
        like a sentence or a content or keywords
        '''
        return self.build_tree()

    def get_merge_cost(self, lnode, rnode):
        lvector = lnode.vector
        x = np.append(lvector, rnode.vector)
        hidden, cost = self.bae.predict(x, 
                lnode.n_children, rnode.n_children)
        return cost
