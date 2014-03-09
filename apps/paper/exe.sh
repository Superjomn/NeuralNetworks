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
    ./to_tree.py $DATA_ROOT
}


train_word2vec
