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
from theano import tensor as T
sys.path.append('..')
import numpy
from exec_frame import BaseModel


class BinaryAutoencoder(BaseModel):
    '''
    autoencoder which input are two vectors
    '''
    def __init__(self, numpy_rng=None, input=None, 
            len_vector=8, n_hidden=4,
            alpha=0.001, learning_rate=0.01,
            W=None, bhid=None, bvis=None):
        '''
        :parameters:
            input: tensor of concation of two vectors
            alpha: weight of sturctural cost
        '''
        # the n_visible is len_vector * 2
        self.len_vector = len_vector
        self.n_hidden = n_hidden
        n_visible = len_vector * 2

        if not numpy_rng:
            numpy_rng=numpy.random.RandomState(1234)

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)), 
                dtype=theano.config.floatX)
            W = theano.shared(
                value=initial_W, 
                name='W'
                )

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, 
                dtype=theano.config.floatX),
                borrow = True,
                name='bvis')

        if not bhid:
            bhid = theano.shared(value = numpy.zeros(n_hidden,
                dtype=theano.config.floatX),
                borrow = True,
                name='bhid')

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.numpy_rng = numpy_rng
        self.alpha = alpha
        self.learning_rate = learning_rate

        self.x = input
        if not self.x:
            self.x = T.fvector(name='x')
        # count of left's children
        self.lcount = T.bscalar('c1')
        # count of right's children
        self.rcount = T.bscalar('c2')

        self.train_fn = None
        self.predict_fn = None

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, learning_rate):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        # vectors of original input
        c1 = self.x[0, 0:self.len_vector]
        c2 = self.x[0, self.n_visible:]
        # reconstruction of two vectors
        _c1 = z[0, 0:self.len_vector]
        _c2 = z[0, self.n_visible:]
        # weight of left vector
        lw = (self.lcount + 0.0) / (self.lcount + self.rchild)

        L = T.sqrt(T.sum( 
            lw * (c1 - _c1) ** 2 + \
            (1 - lw) * (c2 - _c2) ** 2))  \
                + self.alpha * T.sum((self.W ** 2))
        # mean cost of all records
        #sparcity_cost = y 
        cost = T.mean(L) 

        gparams = T.grad(cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            update = param - learning_rate * gparam
            update = T.cast(update, theano.config.floatX)
            updates.append((param, update))
        return cost, updates

    def get_train_fn(self):
        cost, updates = self.get_cost_updates(learning_rate=0.1)
        if not self.train_fn:
            self.train_fn = theano.function(
                    [self.x, self.lcount, self.rcount], 
                    cost, updates=updates)
        return self.train_fn

    def get_predict_fn(self):
        if not self.predict_fn:
            cost, updates = self.get_cost_updates(learning_rate=0.1)
            hidden_value = self.get_hidden_values(self.x)

            self.predict_fn = theano.function(
                    [self.x, self.lcount, self.rcount],
                    [hidden_value, cost])
        return self.predict_fn

    def train_iter(self, x, lcount, rcount):
        '''
        one iteration of the training process

        :parameters:
            x: the concatenation of two vectors(left and right)
            lcount: count of left node's children
            rcount: count of right node's children
        '''
        train_fn = self.get_train_fn()
        cost = train_fn(x, lcount, rcount)
        return cost

    def predict(self, x, lcount, rcount):
        '''
        :returns:
            hidden_value
            cost
        '''
        predict_fn = self.get_predict_fn()
        return predict_fn(x, lcount, rcount)



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
