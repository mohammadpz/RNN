import theano
from theano import tensor
import numpy as np
from blocks.bricks import Sigmoid, Linear
from blocks.bricks.recurrent import ClockWork

floatX = theano.config.floatX

x = tensor.tensor3('x', dtype=floatX)
h_initial = tensor.tensor3('h_initial', dtype=floatX)
time = tensor.vector('time', dtype='int16')


# Parameters
n_u = 225 # input vector size (not time at this point)
n_y = n_u # output vector size
n_seq = 200 # number of sequences for training
iteration = 10 # number of epochs of gradient descent
lr = 0.02 # learning rate
module = 5
unit = 100
periods = np.array([1,2,4,8,16], dtype = floatX)



clockwork = ClockWork(input_dim=n_u, module=module, periods=periods, unit=unit, activation=Sigmoid(), name="clockwork rnn")
linear = Linear(input_dim = unit * module, output_dim = n_y, name="output_layer")
h = clockwork.apply(time, x, states=h_initial, iterate=False)
predict = Sigmoid().apply(linear.apply(h))



generate1 = theano.function([x, time, h_initial], [predict, h])