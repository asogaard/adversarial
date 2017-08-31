# -*- coding: utf-8 -*-

"""Custom layers used in the adversarial training of de-correlated jet taggers.

Adapted from: https://github.com/asogaard/AdversarialSubstructure/blob/master/layers.py
"""

# Numpy import(s)
import numpy as np

# Keras import(s)
from keras import backend as K_
from keras.engine.topology import Layer


def gaussian (x, coeff, mean, width):
    """ Utility function, wrapping the computation of a unit gaussian using Keras-backend methods. """ 
    return coeff * K_.exp( - K_.square(x - mean) / 2. / K_.square(width)) / K_.sqrt( 2. * K_.square(width) * np.pi) 

class Posterior(Layer):
    """ Custom layer, modelling the posterior probability distribution for the jet mass using a gaussian mixture model (GMM) """

    def __init__(self, num_components, num_dimensions, **kwargs):
        self.output_dim = 1 # Probability
        self.num_components = num_components
        self.num_dimensions = num_dimensions
        super(Posterior, self).__init__(**kwargs)

    def call(self, x, mask=None):
        """ Main call-method of the layer. 
        The GMM needs to be implemented (1) within this method and (2) using Keras backend functions in order for the error back-propagation to work properly 
        """

        # Unpack list of inputs
        coeffs = x[0]
        means  = x[1:                            1 + 1 * self.num_dimensions]
        widths = x[1 + 1 * self.num_dimensions : 1 + 2 * self.num_dimensions]
        inputs = x[-1]

        # Compute the pdf from the GMM
        pdf = gaussian(inputs[:,0], coeffs[:,0], means[0][:,0], widths[0][:,0])
        for d in range(1, self.num_dimensions):
            pdf *= gaussian(inputs[:,d], 1, means[d][:,0], widths[d][:,0])
            pass
        for c in range(1, self.num_components):
            this_pdf = gaussian(inputs[:,0], coeffs[:,c], means[0][:,c], widths[0][:,c])
            for d in range(1, self.num_dimensions):
                this_pdf *= gaussian(inputs[:,d], 1, means[d][:,c], widths[d][:,c])
                pass
            pdf += this_pdf
            pass
        
        return K_.reshape(pdf, (K_.shape(pdf)[0], 1))


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    pass  


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# From [https://github.com/fchollet/keras/issues/3119]

import theano

class ReverseGradient(theano.Op):
    """ theano operation to reverse the gradients
    Introduced in http://arxiv.org/pdf/1409.7495.pdf
    """

    view_map = {0: [0]}

    __props__ = ('hp_lambda', )

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    pass


class GradientReversalLayer(Layer):
    """ Reverse a gradient 
    <feedforward> return input x
    <backward> return -lambda * delta
    """
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return self.gr_op(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "lambda": self.hp_lambda}
        base_config = super(GradientReversalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    pass
