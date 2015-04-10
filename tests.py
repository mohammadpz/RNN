"""
Implements a Vanilla RNN
This code was highly inspired by Mohammad Pezeshki
author : Eloi Zablocki
"""

import numpy as np

import theano
from theano import tensor


from blocks import initialization
from recurrent import SimpleRecurrent
from blocks.bricks import Tanh, Linear, Sigmoid
from blocks.graph import ComputationGraph
from blocks.bricks.cost import SquaredError
from blocks.algorithms import GradientDescent, Scale, CompositeRule, StepClipping, BasicMomentum
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS

from fuel.datasets import IterableDataset
from fuel.streams import DataStream

from datasets import random_noise, sine_wave

import matplotlib.pyplot as plt

floatX = theano.config.floatX

# Parameters
n_u = 1 # input vector size (not time at this point)
n_y = 1 # output vector size
n_h = 9 # numer of hidden units

iteration = 400 # number of epochs of gradient descent

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
    brick.weights_init = initialization.IsotropicGaussian(0.1)
    brick.biases_init = initialization.Constant(0)
    brick.initialize()


cg = ComputationGraph(cost)
print(VariableFilter(roles=[WEIGHT, BIAS])(cg.variables))

# Training process
algorithm = GradientDescent(cost=cost, params=cg.parameters, step_rule=CompositeRule([Scale(0.0003), BasicMomentum(0.95), StepClipping((10.0))]))
monitor_cost = TrainingDataMonitoring([cost], prefix="train", after_epoch=True)

print "Model built"

############
#       TEST
############

# Build input and output

inputs = sine_wave(10,10,200)


plotinputs = inputs[0, :, 0, :].reshape(200)
plotinputs2 = inputs[0, :, 1, :].reshape(200)
plotinputs3 = inputs[1, :, 0, :].reshape(200)
lin_time = np.arange(220)
outputs = np.zeros(inputs.shape, dtype = floatX)
outputs[:, 0:-1, :, :] = inputs[:, 1:, :, :]

for i in range(10):
    plt.plot(lin_time[:200], inputs[0, :, i, :].reshape(200))
    plt.plot(lin_time[:200], outputs[0, :, i, :].reshape(200))

