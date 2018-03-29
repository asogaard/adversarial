# -*- coding: utf-8 -*-

"""Custom layers used in the adversarial training of de-correlated jet taggers.

Adapted from: https://github.com/asogaard/AdversarialSubstructure/blob/master/layers.py
"""

# Basic import(s)
import logging as log

# Numpy import(s)
import numpy as np
from scipy.stats import norm

# Keras import(s)
from keras import backend as K
from keras.engine.topology import Layer

# Project import(s)
from .utils.ops import *
from .utils.math import *


class DecorrelationLayer (Layer):
    """
    Custom Keras layer, outputting the linear correlation coefficient for the
    outputs from the previous layers.
    """
    def __init__ (self, **kwargs):
        super(DecorrelationLayer, self).__init__(**kwargs)
        pass

    def build (self, input_shape):
        return

    def call (self, x, mask=None):
        assert isinstance(x, list)
        assert len(x) == 3
        return K.tile(K.expand_dims(K.flatten(wcorr(*x)), axis=1), K.shape(x[1]))

    def compute_output_shape (self, input_shape):
        return (None, 1)

    pass


class MinibatchDiscrimination (Layer):
    """Custom layer to implement minibatch discrimination.
    Cf. [https://arxiv.org/pdf/1606.03498.pdf]

    Learnable tensor has dimensions [A x B x C] with
        A: `input_dim`. Number of (intput)features in layer on which minibatch
            discrimination is performed.
        B: `output_dim`. Number of (output) minibatch discrimination features.
        C: `latent_dim`. Size of "latent space".

    Use as:
        # ...
        layer = ... # Preceeding dense Keras layer
        minibatch = MinibatchDiscrimination(20, 10)(layer)
        layer = keras.layers.Concatenate()([layer, minibatch])
        # ...
    """

    def __init__ (self, output_dim, latent_dim, **kwargs):
        """Constructor"""

        # Member variable(s)
        self.input_dim  = None
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # Base class call
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        return


    def build (self, input_shape):
        """Construct learnable tensor"""

        # Check(s)
        assert len(input_shape) == 2
        self.input_dim = input_shape[1]

        # Create a trainable tensor for minibatch discrimination
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.input_dim, self.latent_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)

        # Base class call
        super(MinibatchDiscrimination, self).build(input_shape)  # Be sure to call this somewhere!
        return


    def call (self, x):
        """Compute minibatch discrimination features.
        Cf. [https://github.com/hep-lbdl/adversarial-jets/blob/master/models/networks/ops.py#L14-L19]
        """
        M = K.reshape(K.dot(x, self.kernel), (-1, self.output_dim, self.latent_dim))
        diffs = K.expand_dims(M, 3) - \
            K.expand_dims(K.permute_dimensions(M, [1, 2, 0]), 0)
        L1_norm = K.sum(K.abs(diffs), axis=2)
        return K.mean(K.exp(-L1_norm), axis=2)


    def compute_output_shape (self, input_shape):
        return (input_shape[0], self.output_dim)

    pass


class PosteriorLayer (Layer):
    """Custom layer, modelling the posterior probability distribution for the jet mass using a gaussian mixture model (GMM)"""

    # @TODO:
    # - Check that K.sum((x < 0) || (x > 1)) == 0

    def __init__ (self, gmm_components=np.nan, gmm_dimensions=np.nan, **kwargs):
        self.output_dim = 1 # Probability
        self.gmm_components = gmm_components
        self.gmm_dimensions = gmm_dimensions
        super(PosteriorLayer, self).__init__(**kwargs)

    def call (self, x, mask=None):
        """Main call-method of the layer.

        The GMM needs to be implemented (1) within this method and (2) using
        Keras backend functions in order for the error back-propagation to work
        properly.
        """

        # Check(s)
        #mask = (x < 0) | (x > 1)
        #assert K.sum(mask) == 0, "Received input to PosteriorLayer outside of [0,1]: " + str(K.eval(x[mask]))

        # Unpack list of inputs
        coeffs = x[0]
        means  = x[1:                            1 + 1 * self.gmm_dimensions]
        widths = x[1 + 1 * self.gmm_dimensions : 1 + 2 * self.gmm_dimensions]
        inputs = x[-1]

        # Compute the pdf from the GMM
        pdf  = gaussian(inputs[:,0], coeffs[:,0], means[0][:,0], widths[0][:,0])
        pdf /= gaussian_integral_on_unit_interval(means[0][:,0], widths[0][:,0])
        for d in range(1, self.gmm_dimensions):
            pdf *= gaussian(inputs[:,d], 1,           means[d][:,0], widths[d][:,0])
            pdf /= gaussian_integral_on_unit_interval(means[d][:,0], widths[d][:,0])
            pass
        for c in range(1, self.gmm_components):
            this_pdf  = gaussian(inputs[:,0], coeffs[:,c], means[0][:,c], widths[0][:,c])
            this_pdf /= gaussian_integral_on_unit_interval(means[0][:,c], widths[0][:,c])
            for d in range(1, self.gmm_dimensions):
                this_pdf *= gaussian(inputs[:,d], 1,           means[d][:,c], widths[d][:,c])
                this_pdf /= gaussian_integral_on_unit_interval(means[d][:,c], widths[d][:,c])
                pass
            pdf += this_pdf
            pass

        return K.reshape(pdf, (K.shape(pdf)[0],))

    def compute_output_shape (self, input_shape):
        return (input_shape[0][0], self.output_dim)

    def get_config (self):
        config = {"name": self.__class__.__name__,
                  "gmm_components": self.gmm_components,
                  "gmm_dimensions": self.gmm_dimensions}
        base_config = super(PosteriorLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    pass


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Implementing gradient reversal layer
NUM_GRADIENT_REVERSALS = 0
if K.backend() == 'tensorflow':
    # Tensorflow implementation based on
    # [https://stackoverflow.com/questions/45099737/implement-theano-operation-in-tensorflow]
    log.info("Implementing gradient reversal layer in TensorFlow")

    import tensorflow as tf
    def ReverseGradient (hp_lambda):
        """Function factory for gradient reversal."""

        def reverse_gradient_function (X, hp_lambda=hp_lambda):
            """Flips the sign of the incoming gradient during training."""
            global NUM_GRADIENT_REVERSALS
            NUM_GRADIENT_REVERSALS += 1

            grad_name = "GradientReversal%d" % NUM_GRADIENT_REVERSALS

            @tf.RegisterGradient(grad_name)
            def _flip_gradients(op, grad):
                return [tf.negative(grad) * hp_lambda]

            g = K.get_session().graph
            with g.gradient_override_map({'Identity': grad_name}):
                y = tf.identity(X)
                pass

            return y

        return reverse_gradient_function

else:
    # Theano implementation based on
    # [https://github.com/fchollet/keras/issues/3119#issuecomment-230289301]
    log.info("Implementing gradient reversal layer in Theano")

    import theano
    class ReverseGradient (theano.Op):
        """Theano operation to reverse the gradients
        Introduced in http://arxiv.org/pdf/1409.7495.pdf
        """

        view_map = {0: [0]}

        __props__ = ('hp_lambda', )

        def __init__ (self, hp_lambda):
            super(ReverseGradient, self).__init__()
            self.hp_lambda = hp_lambda

        def make_node (self, x):
            assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
            x = theano.tensor.as_tensor_variable(x)
            return theano.Apply(self, [x], [x.type()])

        def perform (self, node, inputs, output_storage):
            xin, = inputs
            xout, = output_storage
            xout[0] = xin

        def grad (self, input, output_gradients):
            return [-self.hp_lambda * output_gradients[0]]

        def infer_shape (self, node, i0_shapes):
            return i0_shapes

        pass

    pass # end: tensorflow/theano


# Layer implementation based on
# [https://github.com/fchollet/keras/issues/3119#issuecomment-230289301]
class GradientReversalLayer (Layer):
    """Reverse a gradient
    <feedforward> return input x
    <backward> return -lambda * delta
    """
    def __init__ (self, hp_lambda=np.nan, **kwargs):
        # @NOTE: Default (dummy) value for `hp_lambda` is set to allow loading
        # from file, where attribute value is overwritten by saved value.
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)
        pass

    def build (self, input_shape):
        self.trainable_weights = []
        return

    def call (self, x, mask=None):
        return self.gr_op(x)

    def compute_output_shape (self, input_shape):
        return input_shape

    def get_config (self):
        config = {"name":      self.__class__.__name__,
                  "hp_lambda": self.hp_lambda}
        base_config = super(GradientReversalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    pass
