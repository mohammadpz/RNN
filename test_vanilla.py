# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 11:37:20 2015

@author: Eloi
"""

import vanilla
import numpy as np
import theano

f = vanilla.vanilla().rnn()

seq = np.random.randn(n_seq, time_steps, n_u)
targets = np.zeros((n_seq, time_steps, n_y))

print(f(numpy.ones((3, 1, 3), dtype=theano.config.floatX))) 