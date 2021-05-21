# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 00:41:51 2021

@author: Federico van Swaaij
"""

import subprocess
import numpy as np

def run_tm_align(tm_align_dir, pdb_1, pdb_2):
    """
    Run the TM-Align tool to compare the specified proteins.

    Parameters
    ----------
    tm_align_dir : str
        The directory holding the TM-Align tool.
    pdb_1 : str
        The first PDB file.
    pdb_2 : str
        The second PDB file.

    Returns
    -------
    tm_score : float in [0,1]
        The TM-Align score.
    """
    command = tm_align_dir + 'TMalignMac ' + pdb_1 + ' ' + pdb_2 + ' -a T'
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    lines = output.split(b'\n')
    score_line = lines[15] # it is line 16 in the output
    score_str = np.array([score_line.split(b' ')[1]])
    tm_score = score_str.astype(np.float64())[0]
    return tm_score