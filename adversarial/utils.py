#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for training and evaluating adversarial neural networks for de-correlated jet tagging."""

# Basic import(s)
import os
import psutil
from .profiler import profile


def print_memory ():
    """Utility method for logging the current CPU memory usage."""
    mem = psutil.virtual_memory()
    print "CPU memory: {:.1f}GB / {:.1f}GB".format(mem.available * 1.0E-09,
                                                   mem.total     * 1.0E-09)
    return


@profile
def initialise_backend (args):
    """Method to initialise the chosen Keras backend according to the settings
    specified in the command-line arguments `args`."""
    
    # Specify Keras backend and import module
    os.environ['KERAS_BACKEND'] = "tensorflow" if args.tensorflow else "theano"
    
    # Configure backends
    if args.tensorflow:
        
        # Set print level to avoid unecessary warnings, e.g.
        #  $ The TensorFlow library wasn't compiled to use <SSE4.1, ...>
        #  $ instructions, but these are available on your machine and could
        #  $ speed up CPU computations.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Switch: CPU/GPU
        if args.gpu:
            
            # Set this environment variable to "0,1,...", to make Tensorflow
            # use the first N available GPUs
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,range(args.threads)))

        else:
            # Setting this enviorment variable to "" makes all GPUs
            # invisible to tensorflow, thus forcing it to run on CPU (on as
            # many cores as possible)
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            pass
        
        # Load the tensorflow module here to make sure only the correct
        # GPU devices are set up
        import tensorflow as tf

        # @TODO: Some smart selection of GPUs to used based on actual
        # utilisation?
        
        # Manually configure Tensorflow session
        config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1,
                                allow_soft_placement=True,
                                device_count = {args.mode.upper(): args.threads},
                                #log_device_placement=True,
                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,
                                                          allow_growth=True
                                                          ),
                                )
        session = tf.Session(config=config)
        
    else:
        
        if args.gpu:
            if args.threads > 1:
                log.warning("Currently it is not possible to specify more than one GPU thread for \
                Theano backend.")
                pass
        else:
            # Set number of OpenMP threads to use; even if 1, set to force
            # use of OpenMP which doesn't happen otherwise, for some
            # reason. Gives speed-up of factor of ca. 6-7. (60 sec./epoch ->
            # 9 sec./epoch)
            os.environ['OMP_NUM_THREADS'] = str(args.threads)
            pass

        
        # Switch: CPU/GPU
        cuda_version = '8.0.61'
        standard_flags = [
            'device={}'.format('cuda' if args.gpu else 'cpu'),
            'openmp=True',
            ]
        dnn_flags = [
            'dnn.enabled=True',
            'dnn.include_path=/exports/applications/apps/SL7/cuda/{}/include/'.format(cuda_version),
            'dnn.library_path=/exports/applications/apps/SL7/cuda/{}/lib64/'  .format(cuda_version),
            ]
        os.environ["THEANO_FLAGS"] = ','.join(standard_flags + (dnn_flags if args.gpu else []))
        pass
    
    # Import Keras backend
    import keras.backend as K
    K.set_floatx('float32')
    
    if args.tensorflow:
        # Set global Tensorflow session
        K.set_session(session)
        pass
    
    return
