# -*- coding: utf-8 -*-
"""
Created on Fri May 21 00:01:13 2021

@author: Federico van Swaaij
"""

import os
import subprocess
import pickle
import numpy as np

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

def single_stride(stride_dir, pdb):
    """
    Compute the secondary structure of the given protein using the Stride tool.

    Parameters
    ----------
    stride_dir : str
        The directory holding the Stride tool.
    pdb : str
        The PDB file to analyze.

    Returns
    -------
    sequence : list of str
        The sequence of the secondary structure.
    code_dict : dict of str -> int
        A dictionary mapping the secondary structure ID to its count within the protein.
    """
    command = stride_dir + 'stride ' + pdb
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0]
    lines = output.split(b'~~~~')
    code_dict = {}
    sequence = []
    for line in lines:
        line = line.decode('utf-8')
        if line[0:4].strip() == 'ASG':
            code = line[25].strip()
            sequence.append(code)
            if code not in code_dict:
                code_dict[code] = 1
            else:
                code_dict[code] += 1
    return sequence, code_dict

def get_secondary_ratios(stride_dict):
    """
    Compute the ratios of each secondary structure within the structure.

    Parameters
    ----------
    stride_dict : dict of str -> int
        The dictionary built by the Stride tool.

    Returns
    -------
    ratios : numpy.ndarray
        The array containing the ratios. The order is H, G, I, E, B, T, C.
    """
    ratios = np.zeros(shape=(7,))
    total = 0
    for key in stride_dict:
        total +=stride_dict[key]
    if 'H' in stride_dict:
        ratios[0] = stride_dict['H']/total
    else:
        ratios[0] = 0
    if 'G' in stride_dict:
        ratios[1] = stride_dict['G']/total
    else:
        ratios[1] = 0
    if 'I' in stride_dict:
        ratios[2] = stride_dict['I']/total
    else:
        ratios[2] = 0
    if 'E' in stride_dict:
        ratios[3] = stride_dict['E']/total
    else:
        ratios[3] = 0
    if 'B' in stride_dict:
        ratios[4] = stride_dict['B']/total
    else:
        ratios[4] = 0
    if 'b' in stride_dict:
        ratios[4] = stride_dict['b']/total
    else:
        ratios[4] = 0
    if 'T' in stride_dict:
        ratios[5] = stride_dict['T']/total
    else:
        ratios[5] = 0
    if 'C' in stride_dict:
        ratios[6] = stride_dict['C']/total
    else:
        ratios[6] = 0
    return ratios

def get_eht_sequence(sequence):
    """
    Compute whether there exists a HTH, ETE, HTE, ETH composition.

    Parameters
    ----------
    sequence : list of str
        The sequence of the secondary structure.

    Returns
    -------
    eht_seq : str
        The EHT sequence configuration.
    """
    main_seq = ''
    current = ''
    for s in sequence:
        if current == '' or current[-1] == s:
            current += s
        else:
            last = current[-1]
            if last == 'E' and len(current) >= 3:
                main_seq += 'E'
            elif last == 'H' and len(current) >= 2:
                main_seq += 'H'
            elif last == 'T' and len(current) >= 2:
                main_seq += 'T'
            current = ''
        if len(main_seq) >= 3:
            return main_seq
    if len(current) == 0 and len(main_seq) == 3:
        return main_seq
    elif len(current) == 0 and len(main_seq) < 3:
        return None
    last = current[-1]
    if last == 'E' and len(current) >= 3:
        main_seq += 'E'
    elif last == 'H' and len(current) >= 2:
        main_seq += 'H'
    elif last == 'T' and len(current) >= 2:
        main_seq += 'T'
    if len(main_seq) >= 3:
        return main_seq
    else:
        return None

def get_simple_composition(code_dict):
    """
    Compute the main composition of the protein in a very simplistic way.
    
    It limits itself to cluster: helices, strands, strands+helices, coils, turns.

    Parameters
    ----------
    code_dict : dict of str -> int
        The dictionary of secondary structures.

    Returns
    -------
    keys : str
        The secondary structures better representing the protein.
    """
    if 'E' in code_dict and 'H' in code_dict:
        return 'EH'
    elif 'E' in code_dict and 'G' in code_dict:
        return 'EG'
    elif 'E' in code_dict and 'I' in code_dict:
        return 'EI'
    elif 'E' in code_dict:
        return 'E'
    elif 'H' in code_dict:
        return 'H'
    elif 'T' in code_dict:
        return 'T'
    elif 'G' in code_dict:
        return 'G'
    elif 'I' in code_dict:
        return 'I'
    elif 'C' in code_dict:
        return 'C'

def get_composition(code_dict, pct_thr=0.6):
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

    Returns
    -------
    keys : str
        The secondary structures better representing the protein.
    """
    total = 0
    for key in code_dict:
        total += code_dict[key]
    code_dict = dict(sorted(code_dict.items(), key=lambda x: x[1], reverse=True))
    keys, total_probs = '', 0
    for key in code_dict:
        prob = code_dict[key]/total
        total_probs += prob
        keys += key
        if total_probs >= pct_thr:
            keys = ''.join(sorted(keys))
            return keys
        