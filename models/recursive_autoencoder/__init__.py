# -*- coding: utf-8 -*-
'''
Created on Feb 24, 2014

@author: Chunwei Yan @ PKU
@mail:  yanchunwei@outlook.com
'''
from __future__ import division
import sys
import numpy as np
from numpy import linalg as LA

class Param(object):
    def __init__(self, len_vec=50):
        '''
        :parameters:

            W1: forward proporation weight for children
            W2: backward proporation weight for left child
            W3: backward proporation weight for right child
            b1: forward bias for children 
            b2: backward proporation bias for left child
            b3: backward proporation bias for right child
        '''
        self.numpy_rng=np.random.RandomState(1234)
        self.len_vec = len_vec
        self.W1 = self._initial_random_weight_value(len_vec, len_vec)
        self.W2 = self._initial_random_weight_value(len_vec, len_vec)
        self.W3 = self._initial_random_weight_value(len_vec, len_vec)
        self.W4 = self._initial_random_weight_value(len_vec, len_vec)
        self.b1 = np.zeros((len_vec, 1))
        self.b2 = np.zeros((len_vec, 1))
        self.b3 = np.zeros((len_vec, 1))
        self.initGW()

    def initGW(self):
        self.n_nodes = 0
        self.GW1 = np.zeros((self.len_vec, self.len_vec))
        self.GW2 = np.zeros((self.len_vec, self.len_vec))
        self.GW3 = np.zeros((self.len_vec, self.len_vec))
        self.GW4 = np.zeros((self.len_vec, self.len_vec))
        self.Gb1 = np.zeros((self.len_vec, 1))
        self.Gb2 = np.zeros((self.len_vec, 1))
        self.Gb3 = np.zeros((self.len_vec, 1))

    def _initial_random_weight_value(self, n, m):
        initial_W = np.asarray(
            self.numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n + m)),
                high=4 * np.sqrt(6. / (n + m)),
                size=(n, m)))
        return initial_W

    def vector_norm(self, v):
        return v / LA.norm(v)


class Autoencoder(Param):
    def __init__(self, len_vec=50):
        Param.__init__(self, 
                len_vec = len_vec)
    
    def f(self, x):
        t = np.tanh(x)
        return t #/ LA.norm(t)

    def f_norm(self, x):
        t = np.tanh(x)
        return t #/ LA.norm(t)

    def f_der(self, M):
        n = M.shape[0]
        norm = LA.norm(M)
        squared =  M ** 2
        y = M - squared * M
        p1 = np.eye(n) *  ((-1 * squared + 1) * np.ones((n, n))) / norm
        p2 = np.dot(y, M.T) / norm ** 3
        return p1 - p2

    '''
    def f_der(self, v):
        return 1 - np.tanh(v) ** 2
    '''

    def get_hidden(self, lvec, rvec):
        a = np.dot(self.W1, lvec) + np.dot(self.W2 ,rvec) + self.b1
        return self.f(a)

    def get_recons(self, p):
        c1 = self.f(np.dot(self.W3, p) + self.b2)
        c2 = self.f(np.dot(self.W4, p) + self.b3)
        return c1, c2



class RecursiveAutoencoder(Autoencoder):
    def __init__(self, len_vec=50):
        Autoencoder.__init__(self, 
                len_vec=len_vec)
        self.len_vec = len_vec

    def forward_prop(self):
        costs = []
        def update(node, pred_index):
            #print '.. updating', node
            if node.is_leaf() or node.pred_index == pred_index:
                return node.vector
            c1 = update(node.lchild, pred_index)
            c2 = update(node.rchild, pred_index)
            p = self.get_hidden(c1, c2)
            p_norm = p / LA.norm(p)
            node.pred_index += 1
            # reconstruction
            y1, y2 = self.get_recons(p_norm)
            y1_norm, y2_norm = y1 / LA.norm(y1), y2 / LA.norm(y2)
            node.Y1C1 = y1_norm - c1
            node.Y2C2 = y2_norm - c2
            node.delta_out1 = np.dot(self.f_der(y1), node.Y1C1)
            node.delta_out2 = np.dot(self.f_der(y2), node.Y2C2)
            J = (node.Y1C1) ** 2 + (node.Y2C2) ** 2
            # set values
            node.vector = p_norm
            node.norm_vector = p_norm
            node.unnorm_vector = p
            # y_r_ps
            node.Y1, node.Y2 = y1, y2
            costs.append(np.sum(J))
            return node.vector
        # recursively update the entire tree
        update(self.root, self.root.pred_index+1)
        cost = np.mean(costs)
        return cost 

    def train_with_tree(self, tree):
        self.root = tree.root
        cost = self.forward_prop()
        print 'forward cost', cost
        self.backward_prop()

    def backward_prop(self):
        Ws = [np.zeros((self.len_vec, self.len_vec)),
                self.W1, self.W2 ]
        stack = []
        stack.append([self.root, 0 , None])
        self.root.parent_delta = np.zeros((self.len_vec, 1))
        Y0C0 = np.zeros((self.len_vec, 1))
        while len(stack) > 0:
            self.n_nodes += 1
            top = stack.pop()
            current_node, left_or_right, parent_node = top
            YCSelector = [Y0C0, None, None] if \
                    parent_node == None \
                    else  [Y0C0, parent_node.Y1C1, parent_node.Y2C2]
            nodeW = Ws[left_or_right]
            delta = YCSelector[left_or_right]
            # put in children
            if not current_node.is_leaf():
                stack.append([current_node.lchild, 1, current_node])
                stack.append([current_node.rchild, 2, current_node])
                A1 = current_node.unnorm_vector
                A1Norm = current_node.norm_vector
                ND1, ND2 = current_node.delta_out1, current_node.delta_out2
                PD = current_node.parent_delta
                #print 'PD', PD.shape, PD
                # change parameters
                #print 'ND1', ND1
                #print 'ND2', ND2
                activation = np.dot(self.W3.T, ND1) + np.dot(self.W4.T, ND2)
                #print 'active', activation
                activation += np.dot(nodeW.T, PD) - delta
                #print 'activation', activation

                #sys.exit(-1)

                current_delta = np.dot(self.f_der(A1), activation)

                current_node.lchild.parent_delta = current_delta
                current_node.rchild.parent_delta = current_delta
                #print 'current_delta', current_delta
                # change parameters
                #GW1_upd = np.dot(LY1C1, A1.T)
                self.GW1 += np.dot(current_delta , current_node.lchild.vector.T)
                self.GW2 += np.dot(current_delta , current_node.rchild.vector.T)
                self.GW3 += np.dot(ND1, A1Norm.T)
                self.GW4 += np.dot(ND2, A1Norm.T)
                self.Gb1 += current_delta
                self.Gb2 += ND1
                self.Gb3 += ND2
            else:
                current_delta = np.dot(nodeW.T, PD) - delta
                current_node.current_delta = current_delta
        # clear data
        self.accumulate()
        self.initGW()

    def accumulate(self):
        self.W1 -= self.GW1 / self.n_nodes
        self.W2 -= self.GW2 / self.n_nodes
        self.W3 -= self.GW3 / self.n_nodes
        self.W4 -= self.GW4 / self.n_nodes
        self.b1 -= self.Gb1 / self.n_nodes
        self.b2 -= self.Gb2 / self.n_nodes
        self.b3 -= self.Gb3 / self.n_nodes


    def get_vec(self, node, pred_index):
        '''
        get a node's pred_indexth updated vector 
        '''
        if not node:
            return
        if node.is_leaf() or node.pred_index == pred_index:
            assert node.vector != None
            return node.vector

        lvec = self.get_vec(node.lchild, pred_index)
        rvec = self.get_vec(node.rchild, pred_index)
        assert lvec is not None
        assert rvec is not None

        node.vector = self.ae.hidden_fn(lvec, rvec)
        node.pred_index += 1
        assert node.pred_index == pred_index
        return node.vector



if __name__ == '__main__':
    ae = Autoencoder()
    x = np.array([1, 2, 3, 4]).reshape((4, 1))
    #x = np.array([0.1, 0.2, 0.3, 0.4]).reshape((4, 1))
    print 'x', x
    print 'f_der', ae.f_der(x)
    print 'f', ae.f(x)
