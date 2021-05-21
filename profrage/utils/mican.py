#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:22:11 2021

@author: FVS
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
    s_tm_score : float in [0,1]
        The sTM-Align score.
    """
    command = mican_dir + 'mican ' + pdb_1 + ' ' + pdb_2
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    lines = output.split(b'\n')
    if len(lines) < 18:
        return 0
    score_line = lines[17] # it is line 18 in the output
    score_line = score_line.decode('utf-8')
    scores = score_line.split('=')[1]
    s_tm_score = float(scores.split()[0][:-1]) # take out the comma at the end
    return s_tm_score