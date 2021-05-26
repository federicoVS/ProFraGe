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
    float in [0,1]
        The TM-Align score.
    """
    command = mican_dir + 'mican ' + pdb_1 + ' ' + pdb_2
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    lines = output.split(b'\n')
    for line in lines:
        line = line.decode('utf-8')
        if line.startswith('TM-score'):
            scores = line.split('=')[1]
            tm_score = float(scores.split()[0][:-1]) # take out the comma at the end
            return tm_score
    return 0