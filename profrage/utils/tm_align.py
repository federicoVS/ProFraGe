# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 00:41:51 2021

@author: FVS
"""

import os
import subprocess
import numpy as np
from utils.ProgressBar import ProgressBar
from utils.misc import to_pdb

def tm_align(fragments, tm_align_dir, outfile, out_dir='./', save=True, verbose=False):
    '''
    Call the TM-align structural comparison for all fragments.

    Parameters
    ----------
    fragments : fragments.Fragment
        The fragments to compare.
    tm_align_dir : str
        The directory where the TMalign tool is located.
    outfile : str
        The name of the output file holding the TM-score matrix. It should contain no extension.
    out_dir : str, optional
        The location of where the distance matrix is saved. By default './' (current directory).
    save : bool, optional
        Whether to save the distance matrix as a .npy file. The default is False.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    tm_score_matrix : numpy.ndarray
        The (symmetric) distance matrix.

    '''
    n = len(fragments)
    tm_score_matrix = np.zeros((n,n))
    progress_bar = ProgressBar()
    if verbose:
        print('Computing TM-score matrix...')
        progress_bar.start()
    for i in range(n-1):
        if verbose:
            progress_bar.step(i, n-1)
        for j in range(i+1, n):
            pdb_i = fragments[i].get_name() + '.pdb'
            pdb_j = fragments[j].get_name() + '.pdb'
            to_pdb(fragments[i].get_fragment(), pdb_i[:-4])
            to_pdb(fragments[j].get_fragment(), pdb_j[:-4])
            command = [tm_align_dir + 'TMalignMac', pdb_i, pdb_j, '-a', 'T']
            command = tm_align_dir + 'TMalignMac ' + pdb_i + ' ' + pdb_j + ' -a T'
            ps = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            output = ps.communicate()[0]
            lines = output.split(b'\n')
            score_line = lines[15] # it is line 16 in the output
            score_str = np.array([score_line.split(b' ')[1]])
            tm_score = score_str.astype(np.float64())[0]
            tm_score_matrix[i,j] = tm_score
            tm_score_matrix[j,i] = tm_score
            os.remove(pdb_i)
            os.remove(pdb_j)
    if verbose:
        progress_bar.end()
    if save:
        np.save(out_dir + outfile, tm_score_matrix)
    return tm_score_matrix