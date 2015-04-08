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
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS

from fuel.datasets import IterableDataset
from fuel.streams import DataStream


floatX = theano.config.floatX

# Parameters
n_u = 3 # input vector size (not time at this point)
n_y = 3 # output vector size
n_h = 20 # numer of hidden units
time_steps = 50 # number of time-steps in time
n_seq = 200 # number of sequences for training
iteration = 500 # number of epochs of gradient descent
lr = 0.1 # learning rate

print "Building Model"
# Symbolic variables
x = tensor.tensor3('x', dtype=floatX)
target = tensor.tensor3('target', dtype=floatX)

# Build the model
linear = Linear(input_dim = n_u, output_dim = n_h, name="first_layer")
recurrent = SimpleRecurrent(dim=n_h, activation=Tanh())
linear2 = Linear(input_dim = n_h, output_dim = n_y, name="output_layer")

predict = Sigmoid().apply(linear2.apply(recurrent.apply(linear.apply(x))))

# Cost function
cost = SquaredError().apply(predict,target)

# Initialization
for brick in (recurrent, linear, linear2):
    brick.weights_init = initialization.IsotropicGaussian(0.01)
    brick.biases_init = initialization.Constant(0)
    brick.initialize()


cg = ComputationGraph(cost)
print(VariableFilter(roles=[WEIGHT, BIAS])(cg.variables))

# Training process
algorithm = GradientDescent(cost=cost, params=cg.parameters, step_rule=Scale(lr))
monitor_cost = TrainingDataMonitoring([cost], prefix="train", after_epoch=True)

print "Model built"

############
#       TEST
############


#Build input and output
seq = np.random.randn(1,time_steps, n_seq, n_u)
seq = seq.astype(floatX, copy=False)

targets = np.zeros((1, time_steps, n_seq, n_y), dtype = floatX)
targets[:,1:, :, 0] = seq[:,:-1, :, 0] # 1 time-step delay between input and output
targets[:,2:, :, 1] = seq[:,:-2, :, 1] # 2 time-step delay
targets[:,3:, :, 2] = seq[:,:-3, :, 2] # 3 time-step delay
targets += 0.01 * np.random.standard_normal(targets.shape)

dataset = IterableDataset({'x': seq, 'target': targets})
stream = DataStream(dataset)


model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm, extensions=[monitor_cost, FinishAfter(after_n_epochs=iteration), Printing()], model=model)

print 'Starting training ...'
main_loop.run()