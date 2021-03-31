# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:28:27 2021

@author: FVS
"""

def binary_search(target, lst, sorted=False):
    '''
    Performs binary search over the specified list with the specified target. The type of the target should
    match the types of the elements of the list.

    Parameters
    ----------
    target : Any
        The element to search for in the list.
    lst : list of Any
        The list.
    sorted : bool, optional
        Whether the list is alredy sorted. The default is False.

    Returns
    -------
    bool
        A boolean value indicating whether the target is found in the list.
    '''
    if not sorted:
        lst.sort()
    low = 0
    high = len(lst) - 1
    while low < high:
        mid = low + (high - low)//2
        if lst[mid] == target:
            return True
        elif lst[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return False
    