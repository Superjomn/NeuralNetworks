#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 7, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import sys
import numpy as np
sys.path.append('..')
sys.path.append('../../')
from paper import config as cg
from utils import obj_from_file
from models.recursive_autoencoder import BinaryAutoencoder
from _word2vec import Trainer as Word2Vec
from models.recursive_autoencoder.binary_tree_autoencoder import BinaryTree, BinaryTreeAutoencoder
from syntax_tree.parse_tree import SyntaxTreeParser
from exec_frame import BaseModel, ExecFrame
from parse_tree_autoencoder import ParseTreeAutoencoder


class Tree2Vec(ParseTreeAutoencoder):
    def __init__(self, word2vec, bae):
        ParseTreeAutoencoder.__init__(self, word2vec)
        # replace bae with saved model
        self.bta.bae = bae
        self.bae = bae

    def get_vec_from_stree(self, stree):
        '''
        :parameters:
            stree: string of tree
        '''
        parse_tree = self.create_tree(stree)
        bt = BinaryTree(parse_tree.root, self.bae)
        bt.forward_update_vec()
        sentence_vec =  bt.root.vector
        return sentence_vec




if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print 'cat strees | ./cmd.py w2v_ph bae_ph topath'
        sys.exit(-1)
    strees = sys.stdin.read().split()
    # args
    w2v_ph, bae_ph, topath = args
    # load word2vec
    _word2vec = Word2Vec()
    _word2vec.model_fromfile(w2v_ph)
    # load bae
    bae = obj_from_file(bae_ph)
    tree2vec = Tree2Vec(_word2vec, bae)
    sentence_vecs = []
    for stree in strees:
        sentence_vec = tree2vec.get_vec_from_stree(stree)
        str_vec = '\t'.join(sentence_vec)
        sentence_vecs.append(str_vec)

    with open(topath, 'w') as f:
        f.write('\n'.join(sentence_vecs))
