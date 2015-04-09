import theano
from theano import tensor
import numpy as np
from blocks.bricks import Sigmoid, Linear
from blocks.bricks.recurrent import ClockWork
from datasets import single_bouncing_ball, save_as_gif


floatX = theano.config.floatX

x = tensor.tensor3('x', dtype=floatX)
one_x = tensor.matrix('one_x', dtype=floatX)
h_initial = tensor.matrix('h_initial', dtype=floatX)
time = tensor.vector('time', dtype='int16')
one_time = tensor.wscalar('one_time')

# Parameters
n_u = 225 # input vector size (not time at this point)
n_y = n_u # output vector size
n_seq = 200 # number of sequences for training
iteration = 10 # number of epochs of gradient descent
lr = 0.02 # learning rate
module = 5
unit = 100
periods = np.array([1,2,4,8,16], dtype = floatX)

# Build the model
clockwork = ClockWork(input_dim=n_u, module=module, periods=periods, unit=unit, activation=Sigmoid(), name="clockwork rnn")
linear = Linear(input_dim = unit * module, output_dim = n_y, name="output_layer")
h = clockwork.apply(x, time)
predict = Sigmoid().apply(linear.apply(h))

# only for generation B x h_dim
h_testing = clockwork.apply(inputs=one_x, time=one_time, states=h_initial, iterate=False)
y_hat_testing = Sigmoid().apply(linear.apply(h_testing))
y_hat_testing.name = 'y_hat_testing'


generate1 = theano.function([x, time], [predict, h])
generate2 = theano.function([one_x, one_time, h_initial], [y_hat_testing, h_testing])

time_val = np.zeros((200), dtype = np.int16)
for i in range(200):
    time_val[i] = i
    
p_val, h_val = generate1(np.zeros((200,1,225), dtype = np.float32) , time_val)
predict_val, h_valu = generate2(np.zeros((1,1,225), dtype = np.float32)[0], 5 , np.zeros((1,1,500), dtype = np.float32)[0])

inputs = single_bouncing_ball(10,10,200,15,2)
initial_seq = inputs[0, :20, 0:1, :]
generated_seq = initial_seq[:, 0]

print np.shape(predict_val[:, 0])
print np.shape(generated_seq)