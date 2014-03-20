#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Feb 24, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import sys
import json

args = sys.argv[1:]

if len(args) < 1:
    print './cmd.py sent_ph vec_ph sent_out_ph vec_out_ph'
    sys.exit(-1)

sent_ph, vec_ph, sent_out_ph, vec_out_ph = args

sents = [" "]
vecs = []

with open(sent_ph) as f:
    for line in f.readlines():
        if line.strip():
            sents.append(line)

with open(vec_ph) as f:
    for line in f.readlines():
        ls = line.split('\r')
        vecs.append(ls[0])

# write to file

with open(sent_out_ph, 'w') as f:
    cont = json.dumps(sents)
    f.write(cont)

with open(vec_out_ph, 'w') as f:
    f.write(
        '\n'.join(vecs))
