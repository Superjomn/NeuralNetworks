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
import numpy as np
import  theano
from    theano import tensor as T
import config
from tree import BinaryNode
from exec_frame import BaseModel
from models.recursive_autoencoder import BinaryAutoencoder

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
            if not node:
                return np.float32(0.0)

            if node.is_leaf():
                return np.float32(1.0)

            if node.n_children != np.float32(1.0):
                return node.n_children

            node.n_children = child_count(node.lchild) \
                    + child_count(node.rchild) + np.float32(1.0)
            return node.n_children

        child_count(self.root)


    def forward_update_vec(self):
        '''
        update each node's vector
        '''
        self.root.pred_index += 1
        # both left and right child should exist
        def update(node):
            lvec = np.nan_to_num(self.get_vec(node.lchild, self.root.pred_index))
            rvec = np.nan_to_num(self.get_vec(node.rchild, self.root.pred_index))
            vec = np.append(lvec, rvec)
            hidden = self.ae.hidden_fn(vec)
            node.vector = hidden
            node.pred_index += 1
            assert node.pred_index == self.root.pred_index
        # recursively update the entire tree
        update(self.root)
        self.root.pred_index -= 1

    def backward_recon_vec(self):
        pass




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

        x = np.append(lvec, rvec)
        node.vector = self.ae.hidden_fn(x)
        node.pred_index += 1
        assert node.pred_index == pred_index
        return node.vector

    def get_sub_trees(self):
        pass

        
    def get_vec_batch(self):
        '''
        predict each node's 2-vector
        and return a list of vectors as a batch

        :parameters:
            index: int 
            
        returns:
            a list of vectors
        '''
        self.forward_update_vec()

        children_vecs = []
        child_counts = []
        n_child_pairs = [0]

        def get_children_vec(node):
            if node.is_leaf():
                return None

            n_child_pairs[0] += 1


            children_vecs.append(
                np.append(
                    node.lchild.vector, node.rchild.vector))

            child_counts.append(
                (node.lchild.n_children,
                    node.rchild.n_children))
            # recursively 
            get_children_vec(node.lchild)
            get_children_vec(node.rchild)

        get_children_vec(self.root)

        return children_vecs, child_counts

    @property
    def n_children(self):
        '''
        return the number of children
        '''
        return self.root.n_children



class GlobalBinaryTree(BinaryTree):
    '''
    global updates
    '''
    def __init__(self, root, ae):
        self.root = root
        assert issubclass(
            type(self.root), BinaryNode)
        # autoencoder to merge two children's vectors  to paraent's vector
        self.ae = ae
        self.len_vector = self.ae.len_vector

        self._init_child_count()




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


    # --------------- two kinds of train --------------
    def forward_train(self, vec, lcount, rcount):
        '''
        training with forward update and reconstruction of
        each node
        '''
        cost = self.bae.train_fn(vec, lcount, rcount)
        return cost

    def back_recon_train(self, fath_vec, pre_lvec, pre_rvec, lcount, rcount):
        '''
        training with reverse remodeling loss
        '''
        cost = self.bae.back_recon_train_fn(
                fath_vec, pre_lvec, pre_rvec, lcount, rcount)
        return cost

    # the overall training function
    def train_with_tree(self, tree):
        '''
        get a tree's updated 2-vectors as a matrix
        and reconstruct it with the binary autoencoder

        :parameters:
            tree: object of BinaryTree

        :returns:
            batch
        '''
        assert issubclass(
            type(tree), BinaryTree)
        vecs = []
        vecs, child_counts = tree.get_vec_batch()
        costs = []
        for i,vec in enumerate(vecs):
            n_child = child_counts[i]
            vec = np.nan_to_num(vec)
            if not np.isnan(np.sum(vec)):
                assert n_child[0] > 0.0
                assert n_child[1] > 0.0
                cost = self.bae.forward_train_fn(vec, n_child[0], n_child[1])
                assert not np.isnan(cost), \
                    "vec: %s\nW:%s" % (str(vec), str(self.bae.W.get_value()))
                costs.append(cost)
            else:
                print '!> NaN in vec ...'
        return np.mean(costs)

    def back_recon_train(self, fath_vec, pre_lvec, pre_rvec, lcount, rcount):
        '''
        :parameters:
            pre_lvec: lchild's vector
            pre_rvec: rchild's vector
            fath_vec: father's vector
        '''
        cost = self.bae.back_recon_train_fn(fath_vec, pre_lvec, pre_rvec, lcount, rcount)




    def get_root_vector(self, tree):
        return tree.root.vector
