#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom Keras callbacks."""

# Keras import(s)
from keras.callbacks import Callback

# Project import(s)
# ...


class LossCallback (Callback):
    """Report the losses for certain datasets under identical circumstances."""
    def __init__ (self, **kwargs):
        # Member variable(s)
        self.datasets = dict(**kwargs)

        # Check(s)
        for name, dataset in self.datasets.iteritems():
            assert isinstance(dataset, (list, tuple)), "LossCallback: Argument {} should be list-type.".format(name)
            assert len(dataset) >= 2, "LossCallback: Argument {} need at least 2 entries: inputs, targets, and optionally weights. Only {} were provided.".format(name, len(dataset))
            pass
        return

    def on_epoch_end (self, epoch, logs={}):
        losses = dict()
        for name, dataset in self.datasets.iteritems():
            x = dataset[0]
            y = dataset[1]
            if len(dataset) >= 2:
                w = dataset[2]
            else:
                w = None
                pass

            losses[name] = self.model.evaluate(x, y, sample_weight=w, batch_size=self.params['batch_size'], verbose=0)
            pass

        print "\n" + " | ".join(["{} loss: {: 7.4f}".format(name,loss) for (name,loss) in losses.iteritems()])
        return

    pass


# @TODO:
# - Implement `plot_*` methods, accommodating Pandas.DataFrame inputs.
"""
class PosteriorCallback (Callback):
    """Plot adversary posterior p.d.f. during training."""
    def __init__ (self, data, args, adversary):
        self.opts = dict(data=data, args=args, adversary=adversary)
        return

    def on_train_begin (self, logs={}):
        plot_posterior(name='posterior_begin', title="Beginning of training", **self.opts)
        return

    def on_epoch_end (self, epoch, logs={}):
        plot_posterior(name='posterior_epoch_{:03d}'.format(epoch + 1), title="Epoch {}".format(epoch + 1), **self.opts)
        return
    pass


class ProfilesCallback (Callback):
    """Plot classifier profiles during training."""
    def __init__ (self, data, args, var):
        self.opts = dict(data=data, args=args, var=var)
        return

    def on_train_begin (self, logs={}):
        plot_profiles(name='profiles_begin', title="Beginning of training", **self.opts)
        return

    def on_epoch_end (self, epoch, logs={}):
        plot_profiles(name='profiles_epoch_{:03d}'.format(epoch + 1), title="Epoch {}".format(epoch + 1), **self.opts)
        return
    pass
"""
