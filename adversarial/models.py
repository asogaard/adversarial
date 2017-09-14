# -*- coding: utf-8 -*-

"""Methods for constructing the neural networks used for the adversarial
training of de-correlated jet taggers.

Adapted from https://github.com/asogaard/AdversarialSubstructure/blob/master/models.py
"""

# Basic import(s)
import re

# Keras import(s)
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers.normalization import BatchNormalization

# Project import(s)
from .layers import *

# @TODO:
# - Factorise `adversary_model` and `combined_model`?


def snake_case (string):
    """ ... """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def layer_name_factory (scope):
    """ ... """
    def layer_name (name):
        if scope:
            return '{}/{}'.format(scope, name)
        return name
    return layer_name


def keras_layer_name_factory (scope):
    """ ... """
    layer_name = layer_name_factory(scope)
    def keras_layer_name (cls):
        return layer_name('{}_{}'.format(snake_case(cls), K.get_uid(cls)))
    return keras_layer_name


def stack_layers (input_layer, architecture, default, scope=None):
    """Unified utility method to stack intermediate layers.

    This avoids code duplication between the classifier- and adversary model
    factory methods, and is based on the format in the configuration- and patch
    files, allowing for easy configuration of models e.g. for hyperparameter
    optimisation.

    Args:
        input_layer: The layer on which to stack batch normalisation-, dense-,
            and dropout regularisation layers.
        architecture: List of dictionaries, each one specifying the properties
            of a hidden, densely connected layer, optionally preceeded by a
            batch normalisation layer and succeeded by a dropout regularisation
            layer.
        default: Dictionary specifying the default configuration for all layers,
            optionally overwritten by each of the dicts in `architecture`.
        scope: Name of scope in which the layers should be created.

    Returns:
        The last Keras layer in the stack.
    """

    # Method(s) to get name of layers
    keras_layer_name = keras_layer_name_factory(scope)

    # Prepare first layer
    l = input_layer

    # Loop layer specifications
    for spec in architecture:
        
        # Update the specifications of the current layer to include any defaults
        opts = dict(**default)
        opts.update(spec)
        
        # Extract non-standard keyword arguments
        batchnorm = opts.pop('batchnorm', False)
        dropout   = opts.pop('dropout',   None)
        
        # 1: (Opt.) Add batch normalisation layer before dense layer
        if batchnorm:
            l = BatchNormalization(name=keras_layer_name('BatchNormalization'))(l)
            pass
        
        # 2: Add dense layer according to specifications
        l = Dense(name=keras_layer_name('Dense'), **opts)(l)
        
        # 3: (Opt.) Add dropout regularisation layer after dense layer
        if dropout:
            l = Dropout(dropout, name=keras_layer_name('Dropout'))(l)
            pass
        
        pass
    
    return l


def adversary_model (classifier, gmm_dimensions, gmm_components=None, lambda_reg=None, lr_ratio=None, architecture=[], default=dict(), scope='adversary'):
    """Combined adversarial network model.

    This method creates an adversarial network model based on the provided
    `classifier`, taking as inputs the classifier input and the variables from
    which to de-correlation and outputting the classifer output as well as the
    posterior probability assigned to the input de-correlation variable
    configuration by the adversary's Gaussian mixture model (GMM) p.d.f.

    Args:
        classifier: Keras model, assumed to be sequential N -> 1, to be pitted
            against the adversary.
        gmm_dimensions: The number variables from which to de-correlated,
            corresponding to the number of dimensions in which the adversary's
            p.d.f. exists.
        gmm_components: The number of components in the adversary's GMM.
        lambda_reg: The regularisation parameter $\lambda$, controlling the
            weight on the adversary cost to the combined classifier and
            adversary objective function. This parameter controls the trade-off
            between powerful classification (`lambda_reg` low) and
            de-correlation (`lambda_reg` high).
        lr_ratio: The ratio of the adversary's learning rate to the classifier's.
            This should be much larger than 1, to let the adversary adapt more
            quickly than the classifier, to ensure stability of the final result.
        architecture: List of dicts specifying the architecture of the deep,
            sequential section of the adversary's network. See `stack_layers`.
        default: Default configuration of each layer in the deep, sequential
            section of the adversary's network. See `stack_layers`.
        scope: Name of scope in which the layers should be created.

    Returns:
        Keras model of the combined adversarial network.
    """

    # Method(s) to get name of layers
    keras_layer_name = keras_layer_name_factory(scope)
    layer_name       = layer_name_factory(scope)

    # Classifier
    classifier.trainable = True

    # Gradient reversal layer
    gradient_reversal = GradientReversalLayer(lambda_reg / float(lr_ratio), name=keras_layer_name('GradientReversalLayer'))(classifier.outputs[0])

    ## Intermediate layer(s)
    adversary_stack = stack_layers(gradient_reversal, architecture, default, scope=scope)

    # Posterior p.d.f. parameters
    r_coeffs = Dense(gmm_components, name=layer_name('coeffs'), activation='softmax')(adversary_stack)
    r_means  = list()
    r_widths = list()
    for i in xrange(1, gmm_dimensions + 1):
        r_means .append( Dense(gmm_components, name=layer_name('means_{}'.format(i)))(adversary_stack) )
        pass
    for i in xrange(1, gmm_dimensions + 1):
        r_widths.append( Dense(gmm_components, name=layer_name('widths_{}'.format(i)), activation='softplus')(adversary_stack) )
        pass

    # De-correlation inputs (only used as input to GMM evaluation)
    adversary_input = Input(shape=(gmm_dimensions,), name=layer_name('input'))
   
    # Posterior probability layer
    adversary_output = PosteriorLayer(gmm_components, gmm_dimensions, name=layer_name('output'))([r_coeffs] + r_means + r_widths + [adversary_input])

    # Build model
    model = Model(inputs= classifier.inputs  + [adversary_input],
                  outputs=classifier.outputs + [adversary_output],
                  name=scope)

    # Return
    return model


def classifier_model (num_params, architecture=[], default=dict(), scope='classifier'):
    """Network model used for classifier/tagger.

    Args:
        num_params: Number of input features to the classifier.
        architecture: List of dicts specifying the architecture of the deep,
            sequential section of the adversary's network. See `stack_layers`.
        defalult: Default configuration of each layer in the deep, sequential
            section of the adversary's network. See `stack_layers`.
        scope: Name of scope in which the layers should be created.
        
    Returns:
        Keras model of the classifier network.
    """    

    # Method(s) to get name of layers
    keras_layer_name = keras_layer_name_factory(scope)
    layer_name       = layer_name_factory(scope)

    # Input(s)
    classifier_input = Input(shape=(num_params,), name=layer_name('input'))

    # Layer(s)
    classifier_stack = stack_layers(classifier_input, architecture, default, scope=scope)

    # Output(s)
    classifier_output = Dense(1, activation='sigmoid', name=layer_name('output'))(classifier_stack)

    # Build model
    model = Model(inputs=classifier_input, outputs=classifier_output, name=scope)

    # Return
    return model
