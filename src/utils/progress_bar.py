# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:46:55 2021

@author: FVS
"""

import math
import sys

TOOLBAR_WIDTH = 100

def start():
    '''
    Starts with the progress bar on the standard output.
    
    Source
    ------
    https://stackoverflow.com/questions/3160699/python-progress-bar
    
    Returns
    -------
    None.
    '''
    sys.stdout.write('[%s]' % (' '*TOOLBAR_WIDTH))
    sys.stdout.flush()
    sys.stdout.write('\b' * (TOOLBAR_WIDTH+1))
    
def progress(count, total, latest_bar):
    '''
    Computes and writes the progress bar onto the standard output.
    
    Source
    ------
    https://stackoverflow.com/questions/3160699/python-progress-bar

    Parameters
    ----------
    count : int
        The current steps.
    total : int
        The total number of steps.
    latest_bar : int
        The latest_bar count in the progress bar. It is relative to the TOOLBAR_WIDTH.

    Returns
    -------
    latest_bar : int
        The updated count.
    '''
    should_bar = TOOLBAR_WIDTH*(count/total)
    if math.floor(should_bar) >= latest_bar:
        latest_bar += 1
        sys.stdout.write('#')
        sys.stdout.flush()
    return latest_bar

def end():
    '''
    Ends the progress bar.
    
    Source
    ------
    https://stackoverflow.com/questions/3160699/python-progress-bar

    Returns
    -------
    None.
    '''
    sys.stdout.write('\n')
    