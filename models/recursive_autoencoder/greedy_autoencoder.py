#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 7, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import sys
import theano
from theano import tensor as T
sys.path.append('..')
import numpy
from exec_frame import BaseModel


class BaseParseNode(BaseNode):
    '''
    base model of the node of parse tree
    should be implemented when use ParseTreeAutoencoder
    '''
    def __init__(self):
        self.predict_iter = 0

    def get_word(self):
        raise NotImplemented


class BaseParseTree(object):
    '''
    base model of parse tree
    should be implemented when use ParseTreeAutoencoder
    '''
    pass
    


class ParseTreeAutoencoder(BaseModel):
    '''
    the trainning process is based on a pre-defined tree structure
    '''
    def __init__(self, len_vector, alpha=0.001, 
            learning_rate=0.01):
        '''
        :parameters:
            alpha: weight of sturctural cost
        '''
        self.len_vector = len_vector
        self.predict_fn = None
        self.autoencoder = BinaryAutoencoder(
            len_vector = len_vector,
            n_hidden = len_vector,
        )

    def get_vector(self, word):
        '''
        :parameters:
            word: string

        :returns:
            word vector
        '''
        raise NotImplemented

    def get_tree(self):
        '''
        yield a tree object
        '''
        raise NotImplemented


    def train_iter(self):
        '''
        one iteration of the trainning process
        '''
        tree = self.get_tree()
        self._train_node(tree.root)

    def _train_node(self, node, predict=False):
        '''
        node: object of BaseNode
        predict: bool
            to get the merged vector or update the value
        '''
        if not node:
            return
        if node.is_leaf():
            node.vector = self.get_vector(node.get_word())
        else:
            lvector = self.train_node(node.lchild)
            rvector = self.train_node(node.rchild)
            x = numpy.append(lvector, rvector)
            if not predict:
                self.autoencoder.train_iter(lvector, rvector)
            node.vector = self.get_merged_value(x)
        return node.vector


    def get_merged_value(self, x):
        '''
        x: matrix
        '''
        #print 'x', x, len(x)
        x = x.reshape((1, 2*self.len_vector))
        hidden, cost = self.autoencoder.predict(x)
        return hidden

    def get_vector_batch(self, tree):
        '''
        get vectors
        '''
        predict_index = 0
        def get_leafs_vector(root):
            pass

            
        


# ------------------- greedy ---------------------------

class GreedyNode(BaseNode):
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
            min_error = numpy.inf
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
        x = numpy.append(lvector, rnode.vector)
        hidden, cost = self.bae.predict(x, 
                lnode.n_children, rnode.n_children)
        return cost
