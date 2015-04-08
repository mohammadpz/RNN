import theano
from theano import tensor
from theano.ifelse import ifelse
import numpy
from blocks.bricks import Sigmoid

module = 5
unit = 10
periods = numpy.array([1,2,4,8,16])
time = 4

batch = 20
features = 3

input_size = 3

floatX = theano.config.floatX

states = numpy.zeros((batch, unit * module), dtype = floatX)
inputs = numpy.ones((batch, features), dtype = floatX)


Wh = {}
Wi = {}
for i in range(module):
    for j in range(i, module):
        Wh[str(i)+'_'+str(j)] = numpy.random.rand(unit, unit)

    Wi[i] = numpy.random.rand(features, unit)


W_temp = {}

next_states = numpy.zeros((0, unit))

for i in range(module):
    period = periods[i]
    
    #creates W_temp[i]
    W_temp[i] = numpy.zeros((unit,unit * i))
    
    for j in range(i, module):
        W_temp[i] = numpy.concatenate((W_temp[i], Wh[str(i)+'_'+str(j)]), axis = 1)
    

    if (time % period == 0):
        next_states = numpy.concatenate((next_states, numpy.dot(inputs, Wi[i]) + numpy.dot(states, W_temp[i].T)), axis = 0)
        print numpy.shape(next_states)
    else:
        next_states = numpy.concatenate((next_states,states[:,]), axis = 0)
