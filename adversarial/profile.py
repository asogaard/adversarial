#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for training and evaluating adversarial neural networks for de-correlated jet tagging."""

# Basic import(s)
import re
import sys
import time
import inspect
from inspect import currentframe, getframeinfo, getouterframes
from functools import wraps


class Profile:
    """Profile class, for measuring time elapsed running sections of code."""

    # Static member keeping track of the number of class instances.
    instances = 0
    
    def __init__ (self, title=None, depth=2):
        """Constructor"""
        self.__title = title
        self.__depth = depth
        self.__start = None
        self.__end   = None
        self.__filename  = None
        self.__startline = None
        self.__endline   = None

        self.__prefix = '\033[38;2;74;176;245m\033[1m{title}\033[0m {icon} '.format(title=str(self.__class__).split('.')[-1], icon='⏱ ')
        self.__width = 80

        Profile.instances += 1
        pass

    
    def __del__ (self):
        """Destructor"""
        Profile.instances -= 1
        return


    def get_calling_frameinfo (self):
        return getframeinfo(getouterframes(currentframe())[self.__depth][0])

    
    def indent (self):
        return (Profile.instances - 1) * 2

    
    def indent_string (self, delim='·'):
        return delim * self.indent() + (' ' if self.indent() > 0 else '')

    
    def prefix (self, **kwargs):
        return self.__prefix + self.indent_string(**kwargs)

    
    def length (self, string):
        """Get length of string, manually excluding formatted substrings."""
        regex = re.compile(r"\x1b.*?m")
        return len(regex.sub("", string))


    def __enter__ (self):
        # Extract information
        frameinfo = self.get_calling_frameinfo()
        self.__start = time.time()
        self.__startline = frameinfo.lineno
        self.__filename  = frameinfo.filename

        # Print notice of new, named profiling block
        if self.__title is not None:
            print self.prefix() + 'Starting \033[1m{title}\033[0m'.format(title=self.__title)
            pass
        return


    def __exit__ (self, *args):
        # Extract information
        frameinfo = self.get_calling_frameinfo()
        self.__end = time.time()
        self.__endline = frameinfo.lineno
        assert self.__filename == frameinfo.filename, "Discrepancy in deduced file names"
        duration = self.__end - self.__start

        # Print summary        
        title = self.__title or "%s:L%d-%d" % (self.__filename, self.__startline, self.__endline)
        left  = self.prefix() + "Time elapsed in \033[1m{title}\033[0m: ".format(title=title)
        right = '\033[1m{:.1f}s\033[0m'.format(duration)
        
        # '-2' due to icon in 'prefix'
        # '+indent' due to choice of non-standard '·' character
        # Last term takes formatting of duration into accound
        width = self.__width \
                - (self.length(left) - 2) \
                + self.indent() \
                + (len(right) - self.length(right))

        print "{:s}{:.>{width}s}".format(left, ' ' + right, width=width)
        return

    pass


def profile (fn):
    """Implement Profile class as function decorator."""
    @wraps(fn)
    def wrapper (*args, **kwargs):
        spec  = inspect.getargspec(fn)
        name  = (args[0].__class__.__name__ + '.') if (spec.args and spec.args[0] == 'self') else ''
        name += fn.__name__
        with Profile(title='@' + name, depth=3):
            result = fn(*args, **kwargs)
            pass
        return result
    return wrapper
