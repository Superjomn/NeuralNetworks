#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Feb 26, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import time

class Timeit(object):
    def __init__(self, ori_start_time=None):
        self.ori_start_time = ori_start_time
        self.start_time = time.time()

    def print_time(self):
        end_time = time.time()
        print '> used time:\t%d seconds'  % int(end_time - self.start_time)
        if self.ori_start_time != None:
            print '>total used time:\t%d seconds'  % int(end_time - self.ori_start_time)




if __name__ == "__main__":
    t = Timeit()
    t.print_time()

