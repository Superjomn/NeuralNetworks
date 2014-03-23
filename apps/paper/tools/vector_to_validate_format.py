#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Feb 24, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import os
import sys
import json

args = sys.argv[1:]

if len(args) < 1:
    print './cmd.py sent_ph vec_ph output_root dir_name'
    sys.exit(-1)

sent_ph, vec_ph, output_root , dir_name = args

sents = []
vecs = []

with open(sent_ph) as f:
    for line in f.readlines():
        line = line.strip()
        if line.endswith('.'):
            line = line[:-1]
        ls = line.split()
        sent = [" ", line]
        sents.append(sent)


with open(vec_ph) as f:
    for line in f.readlines():
        ls = line.split('\r')
        vecs.append(ls[0])

print 'len sents', len(sents)
print 'len vecs', len(vecs)
assert len(sents) == len(vecs)
# write to file

sent_out_ph = os.path.join(output_root, "%s_mapall.json" % dir_name)
vec_out_ph = os.path.join(output_root, "%s.txt" % dir_name)

with open(sent_out_ph, 'w') as f:
    cont = json.dumps(sents)
    f.write(cont)

with open(vec_out_ph, 'w') as f:
    f.write(
        '\n'.join(vecs))
