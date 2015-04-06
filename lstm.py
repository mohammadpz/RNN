import theano

from theano import tensor
from blocks import initialization
from blocks.bricks.recurrent import LSTM
from blocks.bricks import Tanh

floatX = theano.config.floatX

x = tensor.tensor3('x')

lstm = LSTM(dim=3, activation=Tanh(), weights_init = initialization.IsotropicGaussian(), biases_init=initialization.IsotropicGaussian())

lstm.initialize()
h,c = lstm.apply(x)

f = theano.function([x], [h])
