# -*- coding: utf-8 -*-

"""Common methods for training and testing neural network classifiers."""

def parallelise_model (model, args):
    """
    Parallelise model on GPUs. Requires TensorFlow backend, GPU-enabled running
    environment, and more than once devices requested/available.

    Arguments:
        model: Keras model to be parallelised.
        args: Namespace holding commandline arguments, for configuration.

    Returns:
        Parallelised, multi-GPU Keras model if possible; otherwise the unaltered
        input model.
    """
    # Import(s) -- done here to ensure that Keras background has been configured.
    from keras.utils import multi_gpu_model

    # Parallelise on GPUs
    if (not args.theano) and args.gpu and args.devices > 1:
        parallelised = multi_gpu_model(model, args.devices)
    else:
        parallelised = classifier
        pass

    return parallelised
