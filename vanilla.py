"""
Implements a Vanilla RNN
This code was highly inspired by Mohammad Pezeshki
author : Eloi Zablocki
"""

import numpy as np

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Tanh, Linear, Sigmoid
from blocks.graph import ComputationGraph
from blocks.bricks.cost import SquaredError
from blocks.algorithms import GradientDescent, Scale, CompositeRule, StepClipping
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS

from fuel.datasets import IterableDataset
from fuel.streams import DataStream

from datasets import single_bouncing_ball, save_as_gif


floatX = theano.config.floatX

# Parameters
n_u = 225 # input vector size (not time at this point)
n_y = 225 # output vector size
n_h = 500 # numer of hidden units

iteration = 200 # number of epochs of gradient descent

print "Building Model"
# Symbolic variables
x = tensor.tensor3('x', dtype=floatX)
target = tensor.tensor3('target', dtype=floatX)

# Build the model
linear = Linear(input_dim = n_u, output_dim = n_h, name="first_layer")
rnn = SimpleRecurrent(dim=n_h, activation=Tanh())
linear2 = Linear(input_dim = n_h, output_dim = n_y, name="output_layer")
sigm = Sigmoid()

x_transform = linear.apply(x)
h = rnn.apply(x_transform)
predict = sigm.apply(linear2.apply(h))


# only for generation B x h_dim
h_initial = tensor.tensor3('h_initial', dtype=floatX)
h_testing = rnn.apply(x_transform, h_initial, iterate=False)
y_hat_testing = linear2.apply(h_testing)
y_hat_testing = sigm.apply(y_hat_testing)
y_hat_testing.name = 'y_hat_testing'


# Cost function
cost = SquaredError().apply(predict,target)

# Initialization
for brick in (rnn, linear, linear2):
    brick.weights_init = initialization.IsotropicGaussian(0.01)
    brick.biases_init = initialization.Constant(0)
    brick.initialize()


cg = ComputationGraph(cost)
print(VariableFilter(roles=[WEIGHT, BIAS])(cg.variables))

# Training process
algorithm = GradientDescent(cost=cost, params=cg.parameters, step_rule=CompositeRule([StepClipping(10.0),Scale(4)]))
monitor_cost = TrainingDataMonitoring([cost], prefix="train", after_epoch=True)

print "Model built"

############
#       TEST
############


# Build input and output

inputs = single_bouncing_ball(10,10,200,15,2)

outputs = np.zeros(inputs.shape, dtype = floatX)
outputs[:, 0:-1, :, :] = inputs[:, 1:, :, :]


print inputs.dtype
print outputs.dtype

print 'Bulding DataStream ...'
dataset = IterableDataset({'x': inputs, 'target': outputs})
stream = DataStream(dataset)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm, extensions=[monitor_cost, FinishAfter(after_n_epochs=iteration), Printing()], model=model)

print 'Starting training ...'
main_loop.run()

generate1 = theano.function([x], [predict, h])
generate2 = theano.function([x, h_initial], [y_hat_testing, h_testing])

initial_seq = inputs[0, :20, 0:1, :]
current_output, current_hidden = generate1(initial_seq)
current_output, current_hidden = current_output[-1:], current_hidden[-1:]
generated_seq = initial_seq[:, 0]
next_input = current_output
prev_state = current_hidden


print np.shape(next_input)
print np.shape(prev_state)


for i in range(200):
    current_output, current_hidden = generate2(next_input, prev_state)
    next_input = current_output
    prev_state = current_hidden
    generated_seq = np.vstack((generated_seq, current_output[:, 0]))

print generated_seq.shape
save_as_gif(generated_seq.reshape(generated_seq.shape[0],
                                  np.sqrt(generated_seq.shape[1]),
                                  np.sqrt(generated_seq.shape[1])), "results/vanilla2.gif")
                                  