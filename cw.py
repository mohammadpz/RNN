"""
Implements a ClockWork RNN
author : Eloi Zablocki
"""

import numpy as np

import theano
from theano import tensor

from blocks import initialization
from blocks.bricks import Linear, Sigmoid
from blocks.bricks.recurrent import  ClockWork
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
lr = 0.01 # learning rate
module = 4
unit = 10
periods = np.array([1,2,4,8], dtype = floatX)

print "Building Model"
# Symbolic variables
x = tensor.tensor3('x', dtype=floatX)
target = tensor.tensor3('target', dtype=floatX)
time = tensor.vector('time', dtype='int16')

# Build the model
linear = Linear(input_dim = n_u, output_dim = unit * module, name="first_layer")

clockwork = ClockWork(module=module, periods=periods, unit=unit, activation=Sigmoid())

linear2 = Linear(input_dim = unit * module, output_dim = n_y, name="output_layer")

predict = Sigmoid().apply(linear2.apply(clockwork.apply(time, linear.apply(x))))

# Cost function
cost = SquaredError().apply(predict,target)

# Initialization
for brick in (clockwork, linear, linear2):
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

targets = np.zeros((1,time_steps, n_seq, n_y), dtype = floatX)
targets[:,1:, :, 0] = seq[:,:-1, :, 0] # 1 time-step delay between input and output
targets[:,2:, :, 1] = seq[:,:-2, :, 1] # 2 time-step delay
targets[:,3:, :, 2] = seq[:,:-3, :, 2] # 3 time-step delay
targets += 0.01 * np.random.standard_normal(targets.shape)

time = np.arange(time_steps).reshape(1,time_steps)
time = time.astype(np.int16)

dataset = IterableDataset({'x': seq, 'time': time, 'target': targets})
stream = DataStream(dataset)


model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm, extensions=[monitor_cost, FinishAfter(after_n_epochs=iteration), Printing()], model=model)

print 'Starting training ...'
main_loop.run()