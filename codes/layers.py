import theano
import theano.tensor as T
import numpy as np
from utils import sharedX


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable

    def forward(self, inputs):
        pass

    def params(self):
        pass


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, inputs):
        # Your codes here


class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)

    def forward(self, inputs):
        # Your codes here


class Linear(Layer):
    def __init__(self, name, inputs_dim, num_output, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.W = sharedX(np.random.randn(inputs_dim, num_output) * init_std, name=name + '/W')
        self.b = sharedX(np.zeros((num_output)), name=name + '/b')

    def forward(self, inputs):
        # Your codes here

    def params(self):
        return [self.W, self.b]


class RNN(Layer):
    def __init__(self, name, hidden_dim, input_dim, init_std):
        super(RNN, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        self.Wx = sharedX(np.random.randn(input_dim, hidden_dim) * init_std, name=name + '/Wx')
        self.Wh = sharedX(np.random.randn(hidden_dim, hidden_dim) * init_std, name=name + '/Wh')
        self.b = sharedX(np.zeros((hidden_dim)), name=name + '/b')

    def forward(self, inputs):
        """
        Args:
            inputs: a sequence size of 4 x 20, stands for two 4-digital numbers

        Returns:
            hs:
                hs[0]: a sequence size of 4 x hidden_dim,
                       stands for step function output at each time
                hs[1]: a sequence size of 4 x hidden_dim,
                       stands for hidden states at each timestamp
        Notice:
            In this simple RNN, step function output is same as hidden states
            But for LSTM, since hidden states also include cell, hs[0] and hs[1:] are different
        """
        hs, _ = theano.scan(
            self.step,          # forward input to output with step function for each time
            sequences=inputs,
            outputs_info=[None, T.zeros(self.hidden_dim)])  # initial values for output and hidden states

        return hs[0]

    def step(self, x_t, h_t_prev):
        """
        Args:
            x_t: input at current timestamp
            h_t_prev: hidden states at previous timestamp

        Returns:
            [h_t, h_t]: output value and updated hidden states value
        """
        h_t = T.tanh(T.dot(x_t, self.Wx) + T.dot(h_t_prev, self.Wh) + self.b)
        return [h_t, h_t]

    def params(self):
        return [self.Wx, self.Wh, self.b]


class LSTM(Layer):
    def __init__(self, name, hidden_dim, input_dim, init_std):
        super(LSTM, self).__init__(name, trainable=True)
        self.hidden_dim = hidden_dim
        # Your codes here, do weights initilization

    def forward(self, inputs):
        results, _ = theano.scan(
            self.step,
            sequences=inputs,
            outputs_info=[None, T.zeros(self.hidden_dim), T.zeros(self.hidden_dim)])

        return results[0]

    def step(self, x_t, h_t_prev, c_t_prev):
        """
        Args:
            x_t: input at current timestamp
            h_t_prev: hidden states at previous timestamp
            c_t_prev: cell states at previous timestamp
        """
        # Your codes here

    def params(self):
        # Your codes here

