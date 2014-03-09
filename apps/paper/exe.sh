#!/bin/bash

set -e
set -x

PROJECT_ROOT=/home/chunwei/Lab/NeuralNetworks/apps/paper
DATA_ROOT=$PROJECT_ROOT/data

DATA_PH=${DATA_ROOT}/duc06/sent_raw


train_word2vec()
{
    toroot=$DATA_ROOT/models
    topath=$toroot/1.w2v
    path_list=$DATA_ROOT/duc.path.list
    mkdir -p $toroot
    cd $DATA_ROOT/DUC
    find `pwd` -name *.sent > $path_list
    cd $PROJECT_ROOT
    cat $path_list | ./word2vec.py $topath
}


gen_syntax_tree()
{
    topath=$DATA_ROOT/syntax_trees.txt
    path_list=$DATA_ROOT/duc.path.list
    cd $DATA_ROOT/DUC
    #cd $DATA_ROOT/test
    find `pwd` -name *.sent > $path_list
    cd $PROJECT_ROOT/syntax_tree
    cat $path_list | ./to_tree.py $topath
}


#train_word2vec
gen_syntax_tree
