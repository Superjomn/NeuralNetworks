#!/bin/bash

test_ph=$1
to_ph=./predicts.txt

./validate.py -d $test_ph -t predict -m ./17-21-0.000199-0.990595.pk -f $to_ph

echo "write predictions to '$to_ph'" 
