"""
Implements a ClockWork RNN
author : Eloi Zablocki
"""

import numpy as np

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Linear, Sigmoid
from recurrent import ClockWork
from blocks.graph import ComputationGraph
from blocks.bricks.cost import SquaredError
from blocks.algorithms import (GradientDescent, CompositeRule,
                               BasicMomentum, RMSProp)
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS

from fuel.datasets import IterableDataset
from fuel.streams import DataStream

from datasets import sine_wave

import matplotlib.pyplot as plt

floatX = theano.config.floatX

# Parameters
n_u = 1  # input vector size (not time at this point)
n_y = n_u  # output vector size

iteration = 200  # number of epochs of gradient descent
module = 3
unit = 30
periods = np.array([1, 2, 4], dtype=floatX)

print "Building Model"
# Symbolic variables
x = tensor.tensor3('x', dtype=floatX)
one_x = tensor.matrix('one_x', dtype=floatX)
target = tensor.tensor3('target', dtype=floatX)
time = tensor.vector('time', dtype='int16')
one_time = tensor.wscalar('one_time')
h_initial = tensor.matrix('h_initial', dtype=floatX)

# Build the model

clockwork = ClockWork(input_dim=n_u, module=module, periods=periods,
                      unit=unit, activation=Sigmoid(), name="clockwork rnn")
linear = Linear(input_dim=unit * module,
                output_dim=n_y,
                name="output_layer")
h = clockwork.apply(x, time)
predict = Sigmoid().apply(linear.apply(h))

# only for generation B x h_dim
h_testing = clockwork.apply(inputs=one_x, time=one_time,
                            states=h_initial, iterate=False)
y_hat_testing = Sigmoid().apply(linear.apply(h_testing))
y_hat_testing.name = 'y_hat_testing'

# Cost function
cost = SquaredError().apply(predict, target)

# Initialization
for brick in (clockwork, linear):
    brick.weights_init = initialization.IsotropicGaussian(0.01)
    brick.biases_init = initialization.Constant(0)
    brick.initialize()

cg = ComputationGraph(cost)
print(VariableFilter(roles=[WEIGHT, BIAS])(cg.variables))

# Training process
algorithm = GradientDescent(cost=cost, params=cg.parameters,
                            step_rule=CompositeRule([RMSProp(learning_rate=0.0001,
                                                             decay_rate=0.95,
                                                             max_scaling=1e5),
                                                    BasicMomentum(0.9)]))
monitor_cost = TrainingDataMonitoring([cost], prefix="train", after_epoch=True)

print "Model built"

############
#       TEST
############


# Build input and output

inputs = sine_wave(10, 10, 200)

plotinputs = inputs[0, :, 0, :].reshape(200)
lin_time = np.arange(220)
plt.plot(lin_time[:200], plotinputs)

outputs = np.zeros(inputs.shape, dtype=floatX)
outputs[:, 0:-1, :, :] = inputs[:, 1:, :, :]
time_val = np.zeros((10, 200), dtype=np.int16)

for i in range(10):
    for j in range(200):
        time_val[i, j] = j
print time_val.dtype
print inputs.dtype
print outputs.dtype


print 'Bulding DataStream ...'
dataset = IterableDataset({'x': inputs, 'time': time_val, 'target': outputs})
stream = DataStream(dataset)

model = Model(cost)
main_loop = MainLoop(data_stream=stream,
                     algorithm=algorithm,
                     extensions=[monitor_cost,
                                 FinishAfter(after_n_epochs=iteration),
                                 Printing()],
                     model=model)

print 'Starting training ...'
main_loop.run()

generate1 = theano.function([x, time], [predict, h])
generate2 = theano.function([one_x, one_time, h_initial],
                            [y_hat_testing, h_testing])

initial_seq = inputs[0, :20, 0:1, :]
current_output, current_hidden = generate1(initial_seq, time_val[0, :20])
current_output, current_hidden = current_output[-1:], current_hidden[-1:]
generated_seq = initial_seq[:, 0]
next_input = current_output
prev_state = current_hidden


next_input = next_input[0]
prev_state = prev_state[0]

print np.shape(next_input)
print np.shape(prev_state)


for i in range(200):
    current_output, current_hidden = generate2(next_input, i, prev_state)
    next_input = current_output
    prev_state = current_hidden
    generated_seq = np.vstack((generated_seq, current_output))

print generated_seq.shape
plt.plot(lin_time, generated_seq.reshape((generated_seq.shape[0], 1)))
plt.show()
