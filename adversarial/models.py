# -*- coding: utf-8 -*-

"""Methods for constructing the neural networks used for the adversarial training of de-correlated jet taggers.

Adapted from https://github.com/asogaard/AdversarialSubstructure/blob/master/models.py
"""

# Keras import(s)
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam

# Project import(s)
from layers import *

# Learning configurations
params = {
    'lambda':    1000.,
    'lr':        1E-03, # Learning rate (LR) in the classifier model
    'lr_ratio':  1E+03, # Ratio of the LR in adversarial model to that in the classifier
}

# Create optimisers for each main model
clf_optim = Adam(lr=params['lr'], decay=1E-03)
adv_optim = Adam(lr=params['lr'], decay=1E-03)

# Define compiler options
compiler_options = {

    # Classifier
    'classifier' : {
        'loss': 'binary_crossentropy',
        'optimizer': clf_optim, #'SGD',
    },

    # Adversarial (combined)
    'adversarial' : {
        'loss': ['binary_crossentropy', 'binary_crossentropy'],
        'optimizer': adv_optim,
        'loss_weights': [1, params['lr_ratio']],
    },

}


def adversarial_model (classifier, architecture, num_posterior_components, num_posterior_dimensions):
    """Combined (classifier and adversary) network model used for the training of de-correlated jet taggers."""

    # Classifier
    classifier.trainable = True
   
    # Adversary
    # -- Gradient reversal
    l = GradientReversalLayer(params['lambda'] / float(params['lr_ratio']))(classifier.output)

    # -- De-correlation inputs
    input_decorrelation = Input(shape=(num_posterior_dimensions,), name='input_adversary')

    # -- Intermediate layer(s)
    for ilayer, (nodes, activation) in enumerate(architecture):
        l = Dense(nodes, activation=activation, name='dense_adversary_%d' % ilayer)(l)
        pass

    # -- Posterior p.d.f. parameters
    r_coeffs = Dense(num_posterior_components, name='coeffs_adversary', activation='softmax')(l)
    r_means  = list()
    r_widths = list()
    for i in xrange(num_posterior_dimensions):
        r_means .append( Dense(num_posterior_components, name='mean_adversary_P%d'  % i)(l) )
        pass
    for i in xrange(num_posterior_dimensions):
        r_widths.append( Dense(num_posterior_components, name='width_adversary_P%d' % i, activation='softplus')(l) )
        pass

    # -- Posterior probability layer
    output_adversary = Posterior(num_posterior_components, num_posterior_dimensions, name='adversary')([r_coeffs] + r_means + r_widths + [input_decorrelation])
    
    return Model(input=[classifier.input] + [input_decorrelation], output=[classifier.output, output_adversary], name='combined')


def classifier_model (num_params, architecture=[], default=dict(), name='classifier'):
    """Network model used for classifier/tagger."""
    
    # Input(s)
    classifier_input = Input(shape=(num_params,), name='{}__input'.format(name))

    # Layer(s)
    l = classifier_input
    for ilayer, spec in enumerate(architecture):

        # -- Update the specifications of the current layer to include any defaults
        opts = dict(**default)
        opts.update(spec)

        # -- Extract non-standard keyword arguments
        batchnorm = opts.pop('batchnorm', False)
        dropout   = opts.pop('dropout',   None)

        # -- (Opt.) Add batch normalisation layer
        if batchnorm:
            l = BatchNormalization(name='{}__batch_normalisation_{}'.format(name, ilayer))(l)
            pass

        # -- Add dense layer according to optsifications
        l = Dense(name='{}__dense_{}'.format(name, ilayer), **opts)(l)

        # -- (Opt.) Add dropout regularisation layer
        if dropout:
            l = Dropout(dropout, name='{}__dropout_{}'.format(name, ilayer))(l)
            pass
        pass

    # Output(s)
    classifier_output = Dense(1, activation='sigmoid', name='{}__output'.format(name))(l)

    # Model
    return Model(inputs=classifier_input, outputs=classifier_output, name=name)
