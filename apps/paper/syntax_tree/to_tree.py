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
        tree = str(subtrees[1])
        print 'tree', tree
        trees.append(tree)
    f.close()
    c = '\n'.join(trees)
    f = open(topath, 'w') 
    f.write(c)
    f.close()



if __name__ == "__main__":
    if len(sys.argv) == 1:
        print '%s inpath outpath'
        print '>> each line of the file should be a sentence'
        exit(-1)
    path, topath = sys.argv[1:]
    print 'parse tree from [%s] to [%s]' % (path, topath)
    to_tree(path, topath)
