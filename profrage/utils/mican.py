# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:22:11 2021

@author: Federico van Swaaij
"""

import subprocess

def run_mican(mican_dir, pdb_1, pdb_2):
    """
    Run the MICAN tool to compare the specified proteins.

    Parameters
    ----------
    mican_dir : str
        The directory holding the MICAN tool.
    pdb_1 : str
        The first PDB file.
    pdb_2 : str
        The second PDB file.

    Returns
    -------
    float in [0,1]
        The TM-Align score.
    """
    command = mican_dir + 'mican ' + pdb_1 + ' ' + pdb_2
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    lines = output.split(b'\n')
    if len(lines) < 9:
        return 0
    score_lines = lines[8].decode('utf-8').split()
    if len(score_lines) == 0:
        return 0
    if score_lines[0] == '1': # only insterested in rank 1 (i.e. the best)
        tm_score = float(score_lines[2])
        return tm_score
    return 0