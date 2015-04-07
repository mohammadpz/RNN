import itertools

import numpy as np

import theano

from theano import tensor
from blocks import initialization
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Tanh, MLP
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS

floatX = theano.config.floatX

x = tensor.tensor3('x')
target = tensor.tensor3('target')
lr = 0.01


lstm = LSTM(dim=3, activation=Tanh(), weights_init = initialization.IsotropicGaussian(), biases_init=initialization.IsotropicGaussian())
lstm.initialize()
h, c = lstm.apply(x, iterate = True)
get_h = theano.function([x], [h])

mlp = MLP(activations= [Tanh(name='activation_0')], dims=[3,3], weights_init = initialization.IsotropicGaussian(), biases_init=initialization.Constant(0.01))
pred = mlp.apply(h[0])
get_pred = theano.function([x], [pred])

cost = ((pred - target)**2).sum()
cg = ComputationGraph(cost)
to_be_learned = VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)
all_grad = tensor.grad(cost, to_be_learned)

#train = theano.function(inputs=[x, target], outputs=[pred, cost], updates=([(var, var - lr * g_var) for var, g_var in zip(to_be_learned,all_grad)]))
train = theano.function(inputs=[x, target], outputs=[pred, cost])
############
#    TEST
############

#Parameters
n_u = 3 # input vector size (not time at this point)
n_y = 3 # output vector size
time_steps = 15 # number of time-steps in time
n_seq = 100 # number of sequences for training

#Build input and output
seq = np.random.randn(time_steps, n_seq, n_u)
seq = np.concatenate((seq,seq,seq,seq), axis = 2)
seq = seq.astype(floatX, copy=False)

targets = np.zeros((time_steps, n_seq, n_y), dtype = floatX)

targets[1:, :, 0] = seq[:-1, :, 0] # 1 time-step delay between input and output
targets[2:, :, 1] = seq[:-2, :, 1] # 2 time-step delay
targets[3:, :, 2] = seq[:-3, :, 2] # 3 time-step delay
targets += 0.01 * np.random.standard_normal(targets.shape)

for i in range(3):
    prediction = get_pred(seq)
    print prediction
    pred, cost = train(seq,targets)
    print pred    
    print cost

