#!/bin/bash

in_ph=$1
out_ph=$2

awk -F"," '{
    for (i=2; i<NF; i++)
    {
        printf "%s,", $i 
    }
    print $NF;
}' $in_ph > $out_ph
