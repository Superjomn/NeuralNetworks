#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 7, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division


class DUC(object):
    def __init__(self, path):
        self.path = path

    def get_text(self):
        '''
        get lines of the TEXT
        '''
        content = []
        with open(self.path) as f:
            begin = False
            for line in f.readlines():
                if line.find('<TEXT>') != -1:
                    begin = True
                elif line.find('</TEXT>') != -1:
                    break
                elif begin == True:
                    content.append(line.strip())
        return content

    def get_sentences(self):
        '''
        :return: 
            list of sentences(str)
        '''
        content = ' '.join(self.get_text())
        sentences = content.split('.')
        return sentences




if __name__ == '__main__':
    duc = DUC('/home/chunwei/Lab/NeuralNetworks/apps/paper/data/duc2005_docs/d301i/FT921-10162')
    duc.get_sentences()
