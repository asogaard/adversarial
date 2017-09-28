#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility class to store and manipulate datasets."""

# Basic import(s)
import h5py
from functools import wraps

# Scientific import(s)
import numpy as np
from numpy.lib.recfunctions import append_fields
from sklearn.preprocessing import StandardScaler

# Project import(s)
from .profile import Profile, profile


# Convenience class for storing data
class Data (object):

    def __init__ (self, data, **kwargs):
        """..."""
        # Check(s)
        assert isinstance(data, np.ndarray), \
        "Base data container type ({}) is not supported.".format(str(type(data)))

        # Store base data container
        self.__data = data

        # Initialise list of fields to be sliced
        self.__fields = ['_{}__data'.format(self.__class__.__name__)]

        # Store number of examples, for consistency checks
        self.__num_samples = data.shape[0]

        # Loop manual fields
        for field, array in kwargs.iteritems():

            # Check(s)
            assert isinstance(array, np.ndarray), \
            "Field '{}' type ({}) is not supported.".format(field, str(type(array)))
            
            assert array.shape[0] == self.__num_samples, \
            ("Array in field '{}' does not have the same number of examples as the base data\n"
            "container ({} vs. {}).".format(field, array.shape[0], num_samples))
            
            # Add attribute for the current field
            setattr(self, field, array)

            # Add field to list of manually added attributes
            self.__fields.append(field)
            pass

        # Data train-, test-, and validation split fraction
        self.__split_fractions = None

        # Data splits
        self.__background = None
        self.__signal     = None
        
        self.__train      = None
        self.__test       = None
        self.__validation = None

        # Random seed
        self.__seed = None
        return
    
    
    def __getitem__ (self, key):
        """Define accessor to re-direct to underlying data container."""
        return self.__data[key]


    def __repr__ (self):
        """..."""
        return ("adversarial.data.{}:".format(self.__class__.__name__) + "\n"
                "Fields: {}"          .format(self.__data.dtype.names) + "\n"
                "Contents: {}"        .format(self.__data))


    # Public methods
    # --------------------------------------------------------------------------

    def concatenate (self, other):
        for field in self.__fields:
            setattr(self, field, np.concatenate((
                getattr(self, field), getattr(other, field)
                )))
            pass
        # self.inputs        = np.concatenate((self.inputs,        other.inputs))
        # self.targets       = np.concatenate((self.targets,       other.targets))
        # self.weights       = np.concatenate((self.weights,       other.weights))
        # self.weights_flat  = np.concatenate((self.weights_flat,  other.weights_flat))
        # self.decorrelation = np.concatenate((self.decorrelation, other.decorrelation))
        # self.__data        = np.concatenate((self.__data,        other._Data__data))
        return


    @profile
    def shuffle (self, seed=None):
        """..."""
        
        num_samples = self.inputs.shape[0]
        indices = np.arange(num_samples)
        
        if seed is None:
            self.__seed = int(np.random.rand() * 1E+09)
        else:
            self.__seed = seed
            pass
        
        np.random.seed(self.__seed)
        np.random.shuffle(indices)
        for field in self.__fields:
            setattr(self, field, getattr(self, field)[indices])
            pass
        
        self._clean()
        return

    
    @profile
    def split (self, train=0, test=0, validation=0, seed=None, signal_value=1):
        """..."""
        
        # Check(s)
        assert sum([train, test, validation]) == 1, "Splits do not sum to one ({}, {}, {})".format(train, test, validation)
        self.__split_fractions = dict(train=train, test=test, validation=validation)
        
        if self.__seed is None:
            print "WARNING: Splitting data without shuffling first."
            pass
        
        # Call internal method
        self._split(signal_value)
        return


    def add_field (self, name, array):
        """..."""
        self.__data = append_fields(self.__data, name, array)

        # Clean and re-create train-, test-, validation-, signal-, and
        # background splits.
        self._clean()
        self._split()
        return


    def slice (self, mask):
        """..."""
        # Manually removing (possible) mangling of field names passed to `Data`
        # contructor.
        return Data(**{field.replace('_{}__'.format(self.__class__.__name__), ''): getattr(self, field)[mask] for field in self.__fields})
    
    
    @property
    def signal (self):
        return self.__signal
    
    
    @property
    def background (self):
        return self.__background
    
    
    @property
    def train (self):
        return self.__train
    
    
    @property
    def test (self):
        return self.__test
    

    @property
    def validation (self):
        return self.__validation
    
    
    # Protected methods
    # --------------------------------------------------------------------------
    def _clean (self):
        """..."""
        del self.__background
        del self.__signal
        del self.__train
        del self.__test
        del self.__validation
        
        self.__background = None
        self.__signal     = None
        self.__train      = None
        self.__test       = None
        self.__validation = None
        return
    
        
    def _split (self, signal_value=1):
        """..."""
        
        # Train-, test-, and validation split
        if self.__split_fractions:
            # @TODO: Create _views_ or masks(?).

            # ...
            
            num_samples = self.inputs.shape[0]
            indices = np.arange(num_samples)
            
            # (1) Get training indices
            idx1, idx2 = 0, int(self.__split_fractions['train'] * num_samples)
            indices_train = indices[idx1:idx2]
            
            # (2) Get test indices
            idx1, idx2 = idx2, idx2 + int(self.__split_fractions['test'] * num_samples)
            indices_test = indices[idx1:idx2]
            
            # (3) Get validationing indices (remainder)
            idx1, idx2 = idx2, num_samples-1
            indices_validation = indices[idx1:idx2]
            
            # Store arrays
            self.__train      = self.slice(indices_train)
            self.__test       = self.slice(indices_test)
            self.__validation = self.slice(indices_validation)

            # Split these to allow for e.g. `data.train.signal`
            self.__train     ._split()
            self.__test      ._split()
            self.__validation._split()
        else:
            self.__train      = None
            self.__test       = None
            self.__validation = None
            pass

        # Signal- and background masks
        msk_signal     = (self.targets == signal_value)
        msk_background = (self.targets != signal_value)
        
        # Store arrays
        self.__signal     = self.slice(msk_signal)
        self.__background = self.slice(msk_background)
        return
    
    pass



# Utility methods for reading in and preparing data containers.
# ------------------------------------------------------------------------------

@profile
def load_data (path, dataset_name='dataset'):
    """Method to load dataset from HDF5 file."""
    with h5py.File(path, 'r') as hf:
        data = hf['dataset'][:]
        pass
    return data


@profile
def prepare_data (data):
    """..."""
    
    # Local Keras import
    import keras.backend as K
    
    # Restrict phasespace
    msk  = (data['m']  >  40.) & (data['m']  <  300.)
    msk &= (data['pt'] > 200.) & (data['pt'] < 2000.)
    data = data[msk]
    
    # Get input feeatures
    # -- Extended
    exclude = ['m', 'pt', 'phi', 'eta', 'EventInfo_NPV', 'nthLeading', 'weight', 'weight_flat', 'signal']
    features_ext = [name for name in data.dtype.names if name not in exclude]
    # -- Compatible
    features_Wtop = ['Tau21', 'C2', 'D2', 'Angularity', 'Aplanarity', 'FoxWolfram20', 'KtDR', 'PlanarFlow', 'Split12', 'ZCut12']
    
    features = features_Wtop

    # Re-scale, to let signal and background have same sum of weights
    msk_sig = (data['signal'] == 1)
    num_signal = np.sum(msk_sig)

    data['weight'][ msk_sig] /= np.sum(data['weight'][ msk_sig]) 
    data['weight'][ msk_sig] *= num_signal

    data['weight'][~msk_sig] /= np.sum(data['weight'][~msk_sig]) 
    data['weight'][~msk_sig] *= num_signal

    data['weight_flat'][ msk_sig] /= np.sum(data['weight_flat'][ msk_sig]) 
    data['weight_flat'][ msk_sig] *= num_signal
    
    data['weight_flat'][~msk_sig] /= np.sum(data['weight_flat'][~msk_sig]) 
    data['weight_flat'][~msk_sig] *= num_signal

    # De-correlation features
    decorrelation_features = ['m'] # ..., 'pt']
    
    print "All features:", list(data.dtype.names)
    print "Input features:", features
    print "De-correlation features:", decorrelation_features
    
    # Create structure arrays for each field
    inputs        = data[features].copy()
    targets       = data[['signal']].copy()
    weights       = data[['weight']].copy()
    weights_flat  = data[['weight_flat']].copy()
    decorrelation = data[decorrelation_features].copy()
    
    # Convert structure arrays to regular arrays
    # -- 1D arrays
    targets      = np.array(targets,      dtype=K.floatx())
    weights      = np.array(weights,      dtype=K.floatx())
    weights_flat = np.array(weights_flat, dtype=K.floatx())
    # -- 2D arrays
    inputs        = np.vstack(tuple(inputs[name]        for name in inputs.dtype.names))       .T.astype(K.floatx())
    decorrelation = np.vstack(tuple(decorrelation[name] for name in decorrelation.dtype.names)).T.astype(K.floatx())
    
    # Standard-scale inputs
    substructure_scaler = StandardScaler().fit(inputs)
    inputs = substructure_scaler.transform(inputs)
    
    # Log-scale and normalise decorrelation variables
    for col in range(decorrelation.shape[1]):
        decorrelation[:,col]  = np.log(decorrelation[:,col])
        decorrelation[:,col] -= np.min(decorrelation[:,col])
        decorrelation[:,col] /= np.max(decorrelation[:,col])
        pass

    num_signal     = np.sum(targets == 1)
    num_background = np.sum(targets != 1)

    print "Found {} signal and {} background events, for {} events in total.".format(num_signal, num_background, num_signal + num_background)
    
    return Data(inputs=inputs,
                targets=targets,
                weights=weights,
                weights_flat=weights_flat,
                decorrelation=decorrelation,
                data=data)
