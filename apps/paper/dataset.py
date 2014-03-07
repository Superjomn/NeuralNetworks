#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 7, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import nltk


class DUC(object):
    def __init__(self, path):
        self.path = path

    def get_sentences(self):
        '''
        :return: 
            list of sentences(str)
        '''
        with open(self.path) as f:
            sentences = [s.strip() for s in f.readlines()]
        return sentences

    def split_words(self, sentence):
        return [w.lower() for w in nltk.word_tokenize(sentence)]




if __name__ == '__main__':
    duc = DUC('/home/chunwei/Lab/NeuralNetworks/apps/paper/data/duc06/sent_raw/D0601A/APW19990707.0181.sent')

    sentences = duc.get_sentences()
    for s in sentences:
        print duc.split_words(s)
