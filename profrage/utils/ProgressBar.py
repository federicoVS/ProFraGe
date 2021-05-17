# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 22:46:55 2021

@author: Federico van Swaaij
"""

import math
import sys

class ProgressBar:
    """
    Write a progress bar on the standard output.
    
    Source
    ------
    https://stackoverflow.com/questions/3160699/python-progress-bar
    
    Attributes
    ----------
    n_steps : int
        The total number of steps to be made.
    steps_made : int
        The number of steps made.
    current : int
        The current progression of the progress.
    BAR_WIDTH : int
        The maximum width of the progress bar.
    """
    
    def __init__(self, n_steps, steps_made=1, current=1):
        """
        Initialize the class.

        Parameters
        ----------
        n_steps : int
            The total number of steps to be made.
        steps_made : int, optional
            The number of steps made. The default is 1.
        current : int, optional
            The current value for the progression. The default is 1.

        Returns
        -------
        None.
        """
        self.n_steps = n_steps
        self.steps_made = steps_made
        self.current = current
        self.BAR_WIDTH = 100

    def start(self):
        """
        Start with the progress bar on the standard output.
    
        Returns
        -------
        None.
        """
        sys.stdout.write('[%s]' % (' '*self.BAR_WIDTH))
        sys.stdout.flush()
        sys.stdout.write('\b' * (self.BAR_WIDTH+1))
    
    def step(self):
        """
        Compute and write the progress bar onto the standard output.

        Returns
        -------
        None.
        """
        should_bar = math.floor(self.BAR_WIDTH*(self.steps_made/self.n_steps))
        self.steps_made += 1
        # print(self.current, should_bar)
        if should_bar >= self.current:
            delta = should_bar - self.current
            for i in range(delta):
                sys.stdout.write('#')
            self.current += delta
            sys.stdout.flush()
            
    def end(self):
        """
        End the progress bar.

        Returns
        -------
        None.
        """
        sys.stdout.write('#')
        sys.stdout.write('\n')
    