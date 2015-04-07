import theano

from theano import tensor
from blocks import initialization
from blocks.bricks import Tanh
from blocks.bricks.recurrent import SimpleRecurrent

class vanilla():
    def rnn(self, dim):
        rnn = SimpleRecurrent(dim=dim, activation=Tanh(), weights_init = initialization.IsotropicGaussian(), biases_init=initialization.IsotropicGaussian())
        self.rnn = rnn
        return rnn
    
    def init(self):
        self.rnn.initialize()

    def rnn_function(self):
        x = tensor.tensor3('x')
        y = self.rnn.apply(x)
        self.rnn_function = theano.function([x], y)
        
    def cost(self,x,target):
        cost = ((self.rnn_function(x) - target)**2).sum()
        self.cost = theano.function([x,target], cost)