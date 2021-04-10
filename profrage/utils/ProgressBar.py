# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:46:55 2021

@author: FVS
"""

import math
import sys

class ProgressBar:
    '''
    Writes a progress bar on the standard output.
    
    Source
    ------
    https://stackoverflow.com/questions/3160699/python-progress-bar
    
    Attributes
    ----------
    current : int
        The current progression of the progress.
    BAR_WIDTH : int
        The maximum width of the progress bar.
    '''
    
    def __init__(self, current=1):
        '''
        Initializes the class.

        Parameters
        ----------
        current : int, optional
            The current value for the progression. The default is 1.

        Returns
        -------
        None.
        '''
        self.current = current
        self.BAR_WIDTH = 100

    def start(self):
        '''
        Starts with the progress bar on the standard output.
    
        Returns
        -------
        None.
        '''
        sys.stdout.write('[%s]' % (' '*self.BAR_WIDTH))
        sys.stdout.flush()
        sys.stdout.write('\b' * (self.BAR_WIDTH+1))
    
    def step(self, count, total):
        '''
        Computes and writes the progress bar onto the standard output.

        Parameters
        ----------
        count : int
            The current number steps.
        total : int
            The total number of steps.

        Returns
        -------
        None.
        '''
        should_bar = math.floor(self.BAR_WIDTH*(count/total))
        # print(self.current, should_bar)
        if should_bar >= self.current:
            delta = should_bar - self.current
            for i in range(delta):
                sys.stdout.write('#')
            self.current += delta
            sys.stdout.flush()
            
    def end(self):
        '''
        Ends the progress bar.

        Returns
        -------
        None.
        '''
        sys.stdout.write('#')
        sys.stdout.write('\n')
    