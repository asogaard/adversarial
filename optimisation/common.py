# -*- coding: utf-8 -*-

# Basic import(s)
import os
import json

# Project import(s)
from adversarial.utils import mkdir


class cd:
    """
    Context manager for changing the current working directory.
    From: [https://stackoverflow.com/a/13197763]
    """
    
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
        return
    
    def __enter__(self):
        self.savedPath = os.getcwd()
        print "cd: Going to {}".format(self.newPath)
        os.chdir(self.newPath)
        return
    
    def __exit__(self, etype, value, traceback):
        print "cd: Going back to {}".format(self.savedPath)
        os.chdir(self.savedPath)
        return
    pass


def create_patch (params):
    """
    Create temporary patch dict to use in optimisation.
    
    The method traverses the entries in dict `params`, which have a hierarchial
    structure
    params = { ..., 'first-level/second-level/third-level': value, ... }
    
    The patch is returned as a python dict with structure
        d = { ...
              'first-level' : {
                  'second-level' : {
                      'third-level' : value,
                      ...
                  }
                  ...
              }
              ...

    This dict can be saved to JSON file as a patch file.
    """
    
    # Initialise output dictionary
    patch = dict()
    
    # Loop parameters, establish dictionary structure
    for param in params:
        print "param: {}".format(param)
        entry = patch
        
        # Traverse hierarchy
        keys = param.split('/')
        for key in keys[:-1]:
            if key not in entry:
                entry[key] = dict()
                pass
            entry = entry[key]
            pass
        
        # Set deepest-level entry by assignment
        entry[keys[-1]] = params[param][0]
        pass
    
    return patch


def save_patch (patch, filename):
    """
    ...
    
    Arguments:
        ...
    """
    
    # @TEMP: Debug
    print "- " * 40
    print "Saving the following patch to '{}':".format(filename)
    print patch
    print "- " * 40
    
    # Make sure target directory exists
    directory = '/'.join(filename.split('/')[:-1])
    mkdir(directory)

    # Dump patch to JSONo file
    with open(filename, 'w') as f:
        json.dump(patch, f, indent=4, sort_keys=True)
        pass

    return
    
