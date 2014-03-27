#!/bin/bash

set -e
set -x

PROJECT_ROOT=/home/chunwei/Lab/NeuralNetworks/apps/paper
DATA_ROOT=$PROJECT_ROOT/data
CLEAN_DATA_PATH=$DATA_ROOT/clean.sentences.txt
WORD2VEC_MODEL_PH=$DATA_ROOT/models/3.w2v

DATA_PH=${DATA_ROOT}/duc06/sent_raw

# clean the DUC data
clean_data()
{
    DUC_path=${DATA_ROOT}/DUC
    paths=`find \`pwd\` -name *.sent`
    cd $PROJECT_ROOT
    for path in $paths; do
        echo "cleaninng $path"
        topath=$path.clean
        cat $path |  ./clean_sentence.py > $topath
    done
}


train_word2vec()
{
    toroot=$DATA_ROOT/models
    topath=$WORD2VEC_MODEL_PH
    path_list=$DATA_ROOT/duc.path.list
    cd $PROJECT_ROOT/data/DUC/duc07
    find `pwd` -name *.sent.clean.tree.sent > $path_list
    #mkdir -p $toroot
    cd $PROJECT_ROOT
    #cat $path_list | ./_word2vec.py $topath
    #echo $DATA_ROOT/DUC/duc07/sentencs_text8.stem.sent | ./_word2vec.py $topath
    cat $path_list | ./_word2vec.py $topath
}

_split_array()
{
    local n_total_parts=$1
    local i_mod_part=$2
    local content=$3
    echo $content | awk -v n_total_parts=$n_total_parts -v i_mod_part=$i_mod_part \
    '{
    for(i=1; i<=NF; i++)
        {
            if(i%n_total_parts == i_mod_part)
            {
                printf "%s ", $i;
            }
        }
    }'
}

gen_syntax_tree()
{
    topath=$DATA_ROOT/syntax_trees.txt
    path_list=$DATA_ROOT/duc.path.list
    cd $DATA_ROOT/DUC
    #cd $DATA_ROOT/test
    find `pwd` -name *.sent.clean > $path_list
    cd $PROJECT_ROOT/syntax_tree
    n_total_parts=$1
    n_total_parts=$n_total_parts-1
    for((i=0;i<=$n_total_parts;i++)); do
    {
        echo `_split_array $n_total_parts $i "\`cat $path_list\`"` | ./to_tree.py
    }&
    done&
    wait 
    echo "done"
}

train_parse_tree_autoencoder()
{
    path_list=$DATA_ROOT/duc.path.list
    w2v_path=$DATA_ROOT/models/3.w2v
    model_root=$DATA_ROOT/models/pta07
    cd $DATA_ROOT/DUC/duc07/sent_raw
    #cd $DATA_ROOT/DUC
    find `pwd` -name *.sent.clean.tree > $path_list
    cd $PROJECT_ROOT
    cat $path_list | ./parse_tree_autoencoder.py $model_root $w2v_path 
}

stree_to_sentence_vec()
{
    path_list=$DATA_ROOT/duc.path.list
    bae_path=$DATA_ROOT/models/pta_full/0-3-0.749304.pk
    cd $DATA_ROOT/DUC/duc07/sent_raw
    #cd $DATA_ROOT/DUC
    find `pwd` -name *.sent.clean.tree > $path_list
    cd $PROJECT_ROOT
    cat $path_list | ./parse_tree_to_sentence_vec.py $WORD2VEC_MODEL_PH $bae_path
}

train_graph_tree_autoencoder()
{
    path_list=$DATA_ROOT/duc.path.list
    cd $DATA_ROOT/DUC/duc06/sent_raw
    #cd $DATA_ROOT/DUC
    find `pwd` -name *.sent.clean.tree.vec > $path_list
    cd $PROJECT_ROOT
    cat $path_list | ./graph_tree_autoencoder.py 
}

vector_to_validate_format()
{
    dataroot=$PROJECT_ROOT/data/DUC/duc07/sent_raw
    valid_root=$PROJECT_ROOT/data/valid

    # merge contents
    for dir in `ls -1 $dataroot`; do
        to_root=$dataroot/$dir
        cat `ls -1 $dataroot/$dir/*.sent.valid | sort` > $to_root/$dir.sent.all
        cat `ls -1 $dataroot/$dir/*.sent.clean.tree.vec | sort` > $to_root/$dir.sent.vec.all
        output_root=$valid_root/$dir/10
        mkdir -p $output_root
        $PROJECT_ROOT/tools/vector_to_validate_format.py $to_root/$dir.sent.all $to_root/$dir.sent.vec.all $output_root $dir
    done
}

stree_to_sentence()
{
    path_list=$DATA_ROOT/duc.path.list
    duc_path=$DATA_ROOT/DUC/duc07
    cd $duc_path/sent_raw
    #cd $DATA_ROOT/DUC
    find `pwd` -name *.sent.clean.tree > $path_list
    cd $PROJECT_ROOT/tools
    
    for path in `cat $path_list`; do
        ./stree_to_sentence.py $path $path.sent
    done
}

stem_sentence_to_one_file()
{
    duc_path=$DATA_ROOT/DUC/duc07
    path_list=$DATA_ROOT/duc.path.list
    topath=$duc_path/sentencs_text8.stem.sent
    cd $duc_path/sent_raw
    #cd $DATA_ROOT/DUC
    find `pwd` -name *.sent.clean.tree.sent > $path_list

    cd $PROJECT_ROOT/tools
    cat ../data/text8 `cat $path_list` | ./stemer.py $topath
}




#clean_data
#train_word2vec
#gen_syntax_tree 4
train_parse_tree_autoencoder
#stree_to_sentence_vec
#train_graph_tree_autoencoder
#vector_to_validate_format
#stree_to_sentence
#stem_sentence_to_one_file
