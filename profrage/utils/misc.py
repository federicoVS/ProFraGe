# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 15:46:37 2021

@author: FVS
"""

import os

def get_files(data_dir, ext='.pdb'):
    '''
    Returns a list of files with the desired extension from the specified directory.
    
    Parameters
    ----------
    data_dir : str
        The name of the directory.
    ext : str, optional
        The file extension. The default is '.pdb'
    
    Returns
    -------
    files : list of str
        The list containing the files.
    '''
    files = []
    for file in os.listdir(data_dir):
        if file.endswith(ext):
            files.append(data_dir+file)
    return files