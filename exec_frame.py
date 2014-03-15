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

    def get_model(self):
        '''
        get the model to save to file
        should contain the parameters without dataset
        '''
        return self



class WindowCosts(object):
    def __init__(self, space=10):
        self.space = space
        self.costs = []
        self.last_mean = -1
        # times of update cost
        self._index = -1
        # index of last window
        self._last_record_index = -1

    def add(self, cost):
        self._index += 1
        if self._index - self._last_record_index >= self.space:
            self.last_mean = self.mean()
            self._last_record_index = self._index

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
        # cost of last window
        self.last_window_cost = None
        self.cur_window = WindowCosts(space=window)

    def update_cost(self, cost):
        self.cur_window.add(cost)

    def continue_run(self):
        return self.cur_window.last_mean - self.cur_window.mean()  > self.tolerance 



class ExecFrame(object):
    '''
    a execution framework
    用于检测model是否停止迭代
    '''
    def __init__(self, model, model_root="", n_iters=1000, n_step2save=250,
            window=5, tolerance=0.03):
        '''
        model: object

        n_step2save: int
            save model to pickle files every <n_step2save> steps
            if n_step2save == -1: do not save the model

        model_root: string
            root of model files to save
        '''
        self.model = model
        self.n_iters = n_iters
        self.n_step2save = n_step2save
        self.window = window
        self.tolerance = tolerance
        self.model_root = model_root
        # init
        self.exec_status = ExecStatus(
            n_iters = n_iters,
            window = window,
            tolerance = tolerance
            )
        # the current turn id
        self.times = -1
        self.father = None

    def run(self, father=None):
        # to get validate cost
        self.father = father
        self.iter_index = 0
        self.times += 1
        #for self.iter_index in xrange(self.n_iters):
        while True:
            self.iter_index += 1
            print 'iter:\t', self.iter_index
            self.last_cost = self.model.train_iter()
            self.exec_status.update_cost(self.last_cost)
            if (not self.exec_status.continue_run()) and self.iter_index > self.n_iters:
                print 'interation break!'
                self.save_model()
                break
            elif self.iter_index % self.n_step2save == 0:
                self.save_model()
        print 'end ...'

    def save_model(self):
        if self.n_step2save == -1:
            return
            name = os.path.join(
                    self.model_root, 
                    "%d-%d-%f.pk" % (self.times, self.iter_index, self.last_cost)
                )
        else:
            # the last float is validation's correct rate
            c_rate = self.father.cur_c_rate
            name = os.path.join(
                    self.model_root, 
                    "%d-%d-%f-%f.pk" % (self.times, self.iter_index, self.last_cost, c_rate)
                )

        with open(name, 'wb') as f:
            print 'save model to\t', name
            pk.dump(self.model.get_model(), f)




if __name__ == "__main__":
    pass

