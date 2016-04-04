"""module for a glimpse network"""

from glimpse.sensor import glimpse_sensor
from nnet.layers import ReLULayer, LinearLayer


class GlimpseNetwork(object):
    """implements a glimpse network

    a glimpse network takes an image and a location, calls a glimpse sensor
    for a glimpse of that image, and outputs a weighted combination of the
    glimpse and location.

    weights theta consist of (theta0, theta1, theta2). The output of the
    network for a glimpse p and location l is:
        theta2 * (relu(theta0 * p) + relu(theta1 * l))

    loc (0,0) is the center of an image and (-1, -1) is the top left
    anything outside the bounds of img is padded with 0s

    :param size: side length of each glimpse patch
    :type size: int

    :param scale: scaling factor for the patches. must be at least 1
    :type scale: float

    :param depth: number of patches in the glimpse
    :type depth: int

    :attr glimpse_ins: input variable for the glimpse layer
    :type glimpse_ins: theano.tensor.var.TensorVariable

    :attr glimpse_layer: glimpse layer for the network
    :type glimpse_layer: nnet.ReLULayer

    :attr loc_ins: input variable for the loc layer
    :type loc_ins: theano.tensor.var.TensorVariable

    :attr loc_layer: loc layer for the network
    :type loc_layer: nnet.ReLULayer

    :attr last_layer: last layer of the network
    :type last_layer: nnet.LinearLayer

    :attr params: parameters of each of the parts of the network
    :type params: list of theano.tensor.var.TensorVariable

    :attr outs: output of the network
    :type outs: theano.tensor.var.TensorVariable
    """

    def __init__(
            self, rng, glimpse_ins, loc_ins, n_hidden, n_out, size, scale,
            depth, theta=None, loc_size=2):
        """creates a GlimpseNetwork

        :param rng: random number generator
        :type rng: numpy.random.RandomState

        :param glimpse_ins: input variable for the glimpse
        :type glimpse_ins: thenao.tensor.dvector

        :param loc_ins: input variable for the location
        :type loc_ins: thenao.tensor.dvector

        :param n_hidden: length of the input vector to theta2
        :type n_hidden: int

        :param n_out: length of the output vector of the network
        :type n_out: int

        :param size: side length of each glimpse patch
        :type size: int

        :param scale: scaling factor for the patches. must be at least 1
        :type scale: float

        :param depth: number of patches in the glimpse
        :type depth: int

        :param theta: optional weight initializations (theta0, theta1, theta2)
        :type theta: 3-tuple of numpy.ndarray or None
        """

        theta0 = theta1 = theta2 = None

        self.size = size
        self.scale = scale
        self.depth = depth
        self.glimpse_ins = glimpse_ins
        self.loc_ins = loc_ins

        if theta:
            theta0, theta1, theta2 = theta

        self.glimpse_layer = ReLULayer(
            rng,
            ins=self.glimpse_ins,
            n_in=(size ** 2) * depth,
            n_out=n_hidden,
            W=theta0)

        self.loc_layer = ReLULayer(
            rng,
            ins=self.loc_ins,
            n_in=loc_size,
            n_out=n_hidden,
            W=theta1)

        last_layer_ins = self.glimpse_layer.outs + self.loc_layer.outs
        self.last_layer = LinearLayer(
            rng,
            ins=last_layer_ins,
            n_in=n_hidden,
            n_out=n_out,
            W=theta2)

        self.params = (
            self.glimpse_layer.params +
            self.loc_layer.params +
            self.last_layer.params
        )

        self.outs = self.last_layer.outs

    def get_glimpse(self, img, loc):
        """gets a glimpse of img at loc

        :param img: input image
        :type img: numpy.ndarray

        :param loc: location to look at as specified above
        :type loc: (int, int)

        :returns: glimpse of shape (self.depth, self.size, self.size)
        :rtype: numpy.ndarray
        """

        return glimpse_sensor(img, loc, self.size, self.scale, self.depth)
