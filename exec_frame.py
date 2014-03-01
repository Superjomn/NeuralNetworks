#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on March 1, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
import os
import cPickle as pk

class BaseModel(object):
    '''
    a base model for each trainer with common interfaces
    '''
    def train_iter(self):
        '''
        a iteration of trainning
        return cost
        '''
        raise NotImplemented

    def predict(self):
        '''
        predict the records
        return cost
        '''
        raise NotImplemented

    def validate(self, records, labels):
        '''
        validate the records and compare result with labels
        '''
        raise NotImplemented



class WindowCosts(object):
    def __init__(self, space=10):
        self.space = space
        self.costs = []

    def add(self, cost):
        if len(self.costs) >= self.space:
            del self.costs[0]
        self.costs.append(cost)

    def mean(self):
        return sum(self.costs) / len(self.costs)


class ExecStatus(object):
    '''
    an execute framework
    determines wheather to continue running
    '''
    def __init__(self, n_iters=1000, window=5, tolerance=0.03):
        '''
        window: int
            get the mean of window-span costs

        tolerance: float
            if the mean costs's difference of the last two windows exceeds the tolerance 
            then it will stop iteration.

        n_iters: int
            max iteration steps
        '''
        self.window = window
        self.tolerance = tolerance
        self.model_root = model_root
        # cost of last window
        self.last_window_cost = None
        self.latest_window = WindowErrors()

    def update_cost(self, cost):
        self.latest_window.add(cost)
        last_mean = self.latest_window.mean()

    def continue_run(self):
        return last_mean - self.last_cost > self.tolerance


class ExecFrame(object):
    '''
    a execution framework
    '''
    def __init__(self, model, model_root="", n_iters=1000, n_step2save=100,
            window=5, tolerance=0.03):
        '''
        model: object

        n_step2save: int
            save model to pickle files every <n_step2save> steps

        model_root: string
            root of model files to save
        '''
        self.model = model
        self.n_iters = n_iters
        self.n_step2save = n_step2save
        self.window = window
        self.tolerance = tolerance
        # init
        self.exec_status = ExecStatus(
            n_iters = n_iters,
            window = window,
            tolerance = tolerance
            )

    def run(self):
        self.iter_index = 0
        for self.iter_index in xrange(self.n_iters):
            cost = self.model.train_iter()
            self.exec_status.add(cost)
            if not self.exec_status.continue_run():
                self.save_model()
            elif self.iter_index % self.n_step2save == 0:
                self.save_model()
            self.last_cost = cost

    def save_model(self):
        name = os.path.join(
                self.model_root, 
                "%d-%f.pk" % (self.index, self.last_cost)
            )
        with open(name, 'wb') as f:
            print 'save model to\t', name
            pk.dump(self.model, f)
            

    def load_model_from_file(self, path):
        with open(path, 'rb') as f:
            print 'load model from\t', path
            self.model = pk.load(f)




if __name__ == "__main__":
    pass

