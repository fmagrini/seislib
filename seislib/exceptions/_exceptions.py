#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Fabrizio Magrini
@email1: fabrizio.magrini@uniroma3.it
@email2: fabrizio.magrini90@gmail.com
"""



class DispersionCurveException(Exception):
    """
    Exception raised when a dispersion curve could not be extracted from the 
    data
    """

    def __init__(self):
        self.message = 'It was not possible to retrieve a dispersion curve'
        super().__init__(self.message)

    def __str__(self):
        return self.message


class TimeSpanException(Exception):
    """
    Exception raised when no common time span is found in two obspy traces or 
    streams.
    """

    def __init__(self, *args, message=None):
        if message is not None:
            self.message = message
        else:
            self.message = 'No common time span found.'
        for arg in args:
            self.message += '\n%s'%arg
        super().__init__(self.message)

    def __str__(self):
        return self.message


class NonFiniteDataException(Exception):
    """
    Exception raised when the data should be strictly finite but contain
    infinite or nan values.
    """

    def __init__(self, *args):
        self.message = 'The data should be strictly finite, but contain either'
        self.message += ' infinite or nan values.'
        for arg in args:
            self.message += '\n%s'%arg
        super().__init__(self.message)

    def __str__(self):
        return self.message



