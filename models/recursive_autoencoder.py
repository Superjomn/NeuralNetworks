# -*- coding: utf-8 -*-
'''
Created on March 12, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com

Implementation of Recursive Autoencoder

For detail, read <R. Socher, J. Pennington, E. H. Huang, A. Y. Ng, and C. D. Manning, "Semi-Supervised Recursive Autoencoder for Predicting Sentiment Distributions">
'''
import sys
import theano
sys.path.append('..')
import numpy
from exec_frame import BaseModel
from autoencoder import BatchAutoEncoder

class BaseNode(object):
    '''
    base model of tree's node
    '''
    def __init__(self):
        self.lchild = None
        self.rchild = None

    def is_leaf(self):
        return not (self.lchild or self.rchild)

# ------------------ parse tree -----------------------

class BaseParseNode(BaseNode):
    '''
    base model of the node of parse tree
    should be implemented when use ParseTreeAutoencoder
    '''

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
    def __init__(self, len_vector, sparsity=0.05, beta=0.001):
        self.len_vector = len_vector
        self.predict_fn = None
        self.autoencoder = BatchAutoEncoder(
            n_visible = 2 * len_vector,
            n_hidden = len_vector,
            sparsity = sparsity,
            beta = beta,
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
        if not self.predict_fn:
            self.predict_fn = theano.function(
                [self.autoencoder.x],
                self.autoencoder.get_hidden_values(self.autoencoder.x)
                )
        #print 'x', x, len(x)
        x = x.reshape((1, 2*self.len_vector))
        return self.predict_fn(x)


# ------------------- greedy ---------------------------

class GreedyNode(BaseNode):
    '''
    base model of the node of Huffman tree
    '''
    def __init__(self, vector=None, lchild=None, rchild=None):
        self.vector = vector
        self.lchild = lchild  
        self.rchild = rchild


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
    '''
    def __init__(self, nodes, len_vector, sparsity=0.05, beta=0.001):
        GreedyTree.__init__(self)
        self.autoencoder = BatchAutoEncoder(
            n_visible = 2 * len_vector,
            n_hidden = len_vector,
            sparsity = sparsity,
            beta = beta,
        )

    def get_merge_cost(self, lnode, rnode):
        lvector = lnode.vector
        x = numpy.append(lvector, rnode.vector)
        if not self.predict_fn:
            self.predict_fn = theano.function(
                [self.autoencoder.x],
                self.autoencoder.get_hidden_values(self.autoencoder.x)
                )








