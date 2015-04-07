import numpy as np

import theano

from theano import tensor
from blocks import initialization
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Tanh, Linear
from blocks.graph import ComputationGraph

from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS

floatX = theano.config.floatX

#Parameters
n_u = 3 # input vector size (not time at this point)
n_y = 3 # output vector size
time_steps = 15 # number of time-steps in time
n_seq = 100 # number of sequences for training


x = tensor.tensor3('x')
target = tensor.tensor3('target')
lr = 0.0001

#build model

linear = Linear(input_dim = n_u, output_dim = 4 * 9, name="first_layer")
x_to_h = linear.apply(x)
lstm = LSTM(dim=9, activation=Tanh())
h_to_h1,_ = lstm.apply(x_to_h)
linear2 = Linear(input_dim = 9, output_dim = n_y, name="output_layer")
h1_to_out = linear2.apply(h_to_h1)


#def give_shape(object, x):
#    print np.shape(x)
#    
#op_shape = theano.printing.Print(global_fn = give_shape)
#h = op_shape(h)

        
#training,
cost = ((h1_to_out - target)**2).sum()
cg = ComputationGraph(cost)

to_be_learned = VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)
all_grad = tensor.grad(cost, to_be_learned)

for brick in (lstm, linear, linear2):
    brick.weights_init = initialization.IsotropicGaussian(0.01)
    brick.biases_init = initialization.Constant(0)
    brick.initialize()
    
train = theano.function(inputs=[x, target], outputs=cost, updates=([(var, var - lr * g_var) for var, g_var in zip(to_be_learned,all_grad)]))

############
#       TEST
############


#Build input and output
seq = np.random.randn(time_steps, n_seq, n_u)
seq = seq.astype(floatX, copy=False)

targets = np.zeros((time_steps, n_seq, n_y), dtype = floatX)

targets[1:, :, 0] = seq[:-1, :, 0] # 1 time-step delay between input and output
targets[2:, :, 1] = seq[:-2, :, 1] # 2 time-step delay
targets[3:, :, 2] = seq[:-3, :, 2] # 3 time-step delay
targets += 0.01 * np.random.standard_normal(targets.shape)

for i in range(300):
    cost = train(seq,targets)
    print cost
