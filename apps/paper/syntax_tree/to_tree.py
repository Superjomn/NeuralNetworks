#!/usr/bin/jython
# -*- coding: utf-8 -*-
'''
Created on March 6, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com

generate syntax tree using Standford Parser
    with the wrapper: standford-parser-in-jython
        https://github.com/vpekar/stanford-parser-in-jython

'''
import os
import sys
sys.path.append('/home/chunwei/Lab/stanford-parser-in-jython')
from stanford import StanfordParser, PySentence

PARSER = StanfordParser('/home/chunwei/Lab/stanford-parser/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz')

def to_tree(path, topath):
    '''
    each line is a sentence
    '''
    f = open(path)
    trees = []
    while True:
        line = f.readline().strip()
        if not line:
            break
        sentence = PARSER.parse_xml(line)
        subtrees = []
        for subtree in sentence.parse.subTrees():
            subtrees.append(subtree)
        subtrees = sorted(subtrees, key=lambda x:len(x), reverse=True)
        tree = str(subtrees[1])
        print 'tree', tree
        trees.append(tree)
    f.close()
    c = '\n'.join(trees)
    f = open(topath, 'w') 
    f.write(c)
    f.close()


class TreeGener(object):
    def __init__(self):
        # list of string 
        self.trees = []

    def parse_file(self, path):
        '''
        path:
            path to files with a syntax tree each line
        '''
        f = open(path)
        while True:
            line = f.readline().strip()
            if not line:
                break
            sentence = PARSER.parse_xml(line)
            subtrees = []
            for subtree in sentence.parse.subTrees():
                subtrees.append(subtree)
            subtrees = sorted(subtrees, key=lambda x:len(x), reverse=True)
            tree = str(subtrees[1])
            #print 'tree', tree
            self.trees.append(tree)
        f.close()

    def get_trees(self):
        return self.trees


if __name__ == "__main__":


    paths = sys.stdin.read().split()

    for path in paths:
        print 'parse file:', path
        tree_gener = TreeGener()
        tree_gener.parse_file(path)
        trees = tree_gener.get_trees()
        topath = path+'.tree'
        content = '\n'.join(trees)
        tf = open(topath, 'w') 
        tf.write(content)
        tf.close()
