# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 00:41:51 2021

@author: FVS
"""

import os
from scipy.sparse import csr_matrix
from utils.ProgressBar import ProgressBar
from utils.misc import to_pdb

def tm_align(fragments, tm_align_dir, out_dir='./', verbose=False):
    '''
    Call the TMalign structural comparison for all fragments.

    Parameters
    ----------
    fragments : fragments.Fragment
        The fragments to compare.
    tm_align_dir : str
        The directory where the TMalign tool is located.
    out_dir : str
        The location of where the distance matrix is saved. By default './' (current directory).
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    distance_matrix : scipy.sparse.csr_matrix
        The (symmetric) distance matrix.

    '''
    n = len(fragments)
    distance_matrix = csr_matrix(shape=(n,n))
    progress_bar = ProgressBar()
    if verbose:
        print('Computing distance matrix...')
        progress_bar.start()
    for i in range(n-1):
        if verbose:
            progress_bar.step(i, n)
        for j in range(1, n):
            pdb_i = fragments[i].get_name() + '.pdb'
            pdb_j = fragments[j].get_name() + '.pdb'
            to_pdb(fragments[i].get_fragment(), pdb_i)
            to_pdb(fragments[j].get_fragment(), pdb_j)
            command = './TMalign ' + pdb_i + ' ' + pdb_j + ' a'
            process = os.popen(command)
            output = process.read()
            # TODO parse output and update distance matrix
            os.remove(pdb_i)
            os.remove(pdb_j)
    return distance_matrix