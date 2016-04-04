import unittest

import numpy as np
import theano
import theano.tensor as T

from nnet.layers import DenseLayer


class TestDenseLayer(unittest.TestCase):
    """tests a LinearLayer"""

    def setUp(self):
        self.rng = np.random.RandomState()
        self.n_in = 10
        self.n_out = 20
        self.ins = T.dvector()

        self.W = (np
            .arange(self.n_in * self.n_out)
            .reshape((self.n_out, self.n_in)))
        self.b = np.arange(self.n_out)

        self.model = DenseLayer(
            self.rng, self.ins, self.n_in, self.n_out, self.W, self.b)

    def test_weight_bias_initialization(self):
        """tests initialization of weights and bias in the layer"""

        self.assertTrue(np.all(self.model.W.get_value() == self.W))

    def test_activation(self):
        """tests initialization of weights and bias in the layer"""

        W = np.arange(self.n_in * self.n_out).reshape((self.n_out, self.n_in))
        b = np.arange(self.n_out)

        model = DenseLayer(self.rng, self.ins, self.n_in, self.n_out, W, b)
        self.assertTrue(np.all(model.W.get_value() == W))
