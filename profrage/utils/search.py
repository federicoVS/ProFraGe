# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:28:27 2021

@author: Federico van Swaaij
"""

def binary_search(target, lst):
    """
    Perform binary search over the specified list with the specified target.
    
    The type of the target should match the types of the elements of the list and the list should be sorted.

    Parameters
    ----------
    target : Any
        The element to search for in the list.
    lst : list of Any
        The list.

    Returns
    -------
    bool
        A boolean value indicating whether the target is found in the list.
    int
        The index in the list matching the target.
    """
    low = 0
    high = len(lst) - 1
    while low < high:
        mid = low + (high - low)//2
        if lst[mid] == target:
            return True, mid
        elif lst[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return False, -1
    