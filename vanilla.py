import theano

from theano import tensor
from blocks import initialization
from blocks.bricks import Tanh
from blocks.bricks.recurrent import SimpleRecurrent

x = tensor.tensor3('x')
rnn = SimpleRecurrent(dim=3, activation=Tanh(), weights_init = initialization.IsotropicGaussian(), biases_init=initialization.IsotropicGaussian())
rnn.initialize()
h = rnn.apply(x)
f = theano.function([x],h)

