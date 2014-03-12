#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on March 10, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import re
import sys

while True:
    sentence = sys.stdin.readline().strip()
    if not sentence:
        break

    sentence = re.sub("(('')|[_].|\(.*\))|([.]$)", "", sentence)

    sys.stdout.write(sentence+"\n")
