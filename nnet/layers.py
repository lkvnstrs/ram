"""module for neural network layers implemented in theano"""

import numpy as np
import theano
import theano.tensor as T


class DenseLayer(object):
    """class implementing a dense fully-connected layer"""

    def __init__(self, rng, ins, n_in, n_out, W=None, b=None, activ=None):
        """creates a FullyConnectedLayer

        :param rng: random number generator
        :type rng: numpy.random.RandomState

        :param ins: input values
        :type ins: theano.tensor.dmatrix

        :param n_in: number of inputs
        :type n_in: int

        :param n_out: number of outputs
        :type n_out: int

        :param W: weight matrix initialization
        :type W: numpy.ndarray

        :param b: bias initialization
        :type b: numpy.ndarray

        :param activ: activation function or None to default to linear
        :type activ: function or None
        """

        self.ins = ins

        W = self._weight_init(rng, (n_out, n_in)) if W is None else W
        b = self._bias_init(rng, (n_out,)) if b is None else b

        self.W = theano.shared(value=W, name='W_' + str(id(self)), borrow=True)
        self.b = theano.shared(value=b, name='b_' + str(id(self)), borrow=True)
        self.params = [self.W, self.b]

        linear = T.dot(self.W, self.ins) + self.b
        self.outs = linear if activ is None else activ(linear)

    @staticmethod
    def _weight_init(rng, shape):
        """generates an initialization for the weights

        :param rng: random number generator
        :type rng: numpy.random.RandomState

        :param shape: shape of the return initialization
        :type shape: tuple of int

        :returns: weight initialization of size shape
        :rtype: numpy.ndarray
        """

        weight_scale = 10 ** -2

        return np.asarray(
            weight_scale * rng.normal(size=shape),
            dtype=theano.config.floatX)

    @staticmethod
    def _bias_init(rng, shape):
        """generates an initialization for the biases

        :param rng: random number generator
        :type rng: numpy.random.RandomState

        :param shape: shape of the return initialization
        :type shape: tuple of int

        :returns: bias initialization of size shape
        :rtype: numpy.ndarray
        """

        return np.zeros(shape, dtype=theano.config.floatX)


class ReLULayer(DenseLayer):
    """class implementing a layer with rectified linear activation"""

    def __init__(self, rng, ins, n_in, n_out, W=None, b=None):
        return super(ReLULayer, self).__init__(
            rng, ins, n_in, n_out, W, b, T.nnet.relu)


class LinearLayer(DenseLayer):
    """class implementing a linear output layer"""

    def __init__(self, rng, ins, n_in, n_out, W=None, b=None):
        return super(LinearLayer, self).__init__(rng, ins, n_in, n_out, W, b)
