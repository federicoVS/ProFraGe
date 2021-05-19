#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:35:34 2021

@author: FVS
"""

class Loss:
    """
    A general class implementing a loss function.
    
    Attributes
    ----------
    y : Any
        The real value.
    y_hat : Any
        The predicted value.
    """
    
    def __init__(self, y, y_hat):
        """
        Initialize the class.

        Parameters
        ----------
        y : Any
            The real value.
        y_hat : Any
            The predicted value.

        Returns
        -------
        None.
        """
        self.y = y
        self.y_hat = y_hat
        
    def get_loss(self):
        """
        Compute the loss. This method is meant to be overridden by subclasses.

        Returns
        -------
        None.
        """
        pass