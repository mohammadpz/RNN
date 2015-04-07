# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 15:06:22 2015

@author: Eloi
"""
import numpy
import theano
from theano import tensor
from blocks.bricks import Tanh
from blocks import initialization
from blocks.bricks import Identity
from blocks.bricks.recurrent import BaseRecurrent, recurrent, SimpleRecurrent

class CWRNN(BaseRecurrent):
    def __init__(self, dim, **kwargs):
        super(CWRNN, self).__init__(**kwargs)
        self.dim = dim
        self.first_module = SimpleRecurrent(dim=self.dim, activation=Tanh(), name='first_recurrent_layer', weights_init=initialization.Identity())
        self.second_module = SimpleRecurrent(dim=self.dim, activation=Tanh(), name='first_recurrent_layer', weights_init=initialization.Identity())
        self.children = [self.first_module, self.second_module]
        
        
    @recurrent(sequences=['inputs'], contexts=[], states=['first_states', 'second_states'], outputs=['first_states', 'second_states'])
    def apply(self, inputs, first_states=None, second_states=None):
        first_h = self.first_module.apply(inputs = inputs, states = first_states + second_states, iterate = False)
        second_h = self.second_module.apply(inputs = inputs, states = second_states, iterate = False)
        return first_h, second_h
        
    def get_dim(self, name):
        return (self.dim if name in ('inputs', 'first_states', 'second_states') else super(CWRNN, self).get_dim(name))
        
        

x = tensor.tensor3('x')
cwrnn = CWRNN(dim=3)
cwrnn.initialize()
first_h, second_h = cwrnn.apply(inputs=x)
f = theano.function([x], [first_h, second_h])
for states in f(numpy.ones((3,1,3), dtype = theano.config.floatX)):
    print states