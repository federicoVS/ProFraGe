# -*- coding: utf-8 -*-
"""
Created on Fri May 21 00:01:13 2021

@author: Federico van Swaaij
"""

import os
import subprocess
import pickle

from utils.io import get_files
from utils.ProgressBar import ProgressBar

def multi_stride(stride_dir, pdb_dir, out_dir='./', save=False, verbose=False):
    """
    Compute the secondary structures for multiple proteins.

    Parameters
    ----------
    stride_dir : str
        The directory holding the Stride tool.
    pdb_dir : str
        The directory holding the PDB files.
    out_dir : str, optional
        The output directory. The default is './'.
    save : bool, optional
        Whether to save the output. The default is False.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    pdbs = get_files(pdb_dir)
    progress_bar = ProgressBar(len(pdbs))
    if verbose:
        progress_bar.start()
    for pdb in pdbs:
        if verbose:
            progress_bar.step()
        single_stride(stride_dir, pdb, out_dir=out_dir, save=save)
    if verbose:
        progress_bar.end()

def single_stride(stride_dir, pdb, out_dir='./', save=False):
    """
    Compute the secondary structure of the given protein using the Stride tool.

    Parameters
    ----------
    stride_dir : str
        The directory holding the Stride tool.
    pdb : str
        The PDB file to analyze.
    out_dir : str, optional
        The output directory. The default is './'.
    save : bool, optional
        Whether to save the output. The default is False.

    Returns
    -------
    code_dict : dict of str -> int
        A dictionary mapping the secondary structure ID to its count within the protein.
    """
    command = stride_dir + 'stride ' + pdb
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    lines = output.split(b'~~~~')
    code_dict = {}
    for line in lines:
        line = line.decode('utf-8')
        if line[0:4].strip() == 'ASG':
            code = line[25].strip()
            if code not in code_dict:
                code_dict[code] = 1
            else:
                code_dict[code] += 1
    if save:
        pdb_id = os.path.basename(pdb)[:-4]
        file_name = out_dir + pdb_id + '.pkl'
        file = open(file_name, 'wb')
        pickle.dump(code_dict, file)
        file.close()
    return code_dict

def get_composition(code_dict, pct_thr=0.6, min_strands=4):
    """
    Compute the main composition of the protein.
    
    The returned key comprises the top P% of secondary structures that make the protein.

    Parameters
    ----------
    code_dict : dict of str -> int
        The dictionary of secondary structures.
    pct_thr : float in [0,1], optional
        The percentage threshold that is needed to reach for secondary structures to be the main
        components of the protein. The default is 0.6.
    min_strands : int, optional
        The minimum number of strands a protein must have. The default is 4.

    Returns
    -------
    keys : str
        The secondary structures better representing the protein.
    """
    total = 0
    for key in code_dict:
        total += code_dict[key]
    code_dict = dict(sorted(code_dict.items(), key=lambda x: x[1], reverse=True))
    # Strands are a little bit weird, so they should be their own cluster
    e_count = 0
    for key in code_dict:
        if key == 'E':
            e_count += 1
    if e_count >= min_strands:
        return 'E'
    keys, total_probs = '', 0
    for key in code_dict:
        prob = code_dict[key]/total
        total_probs += prob
        keys += key
        if total_probs >= pct_thr:
            keys = ''.join(sorted(keys))
            return keys
        