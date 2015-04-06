import theano

from theano import tensor
from blocks import initialization
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks import Tanh

floatX = theano.config.floatX

x = tensor.tensor3('x')
ri = tensor.tensor3('ri')

gru = GatedRecurrent(dim=3, activation=Tanh(), gate_activation=Tanh(), weights_init = initialization.IsotropicGaussian(), biases_init=initialization.IsotropicGaussian(), use_update_gate=False, seed=1)

gru.initialize()
h = gru.apply(x, reset_inputs=ri)

f = theano.function([x, ri], [h])


