# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:30:55 2021

@author: FVS
"""

import os
from pdb_file import merge_pdbs, get_files
import progress_bar

def merge_fragments(frag_dir, pdb_ch_ids, k=2, out_dir='./', verbose=False):
    '''
    Merge individual fragments into non-redundant super-fragments. The super-fragments are composed by the
    specified number of indivdual fragments.

    Parameters
    ----------
    frag_dir : str
        The directory holding the fragments.
    pdb_ch_ids : list of (str, str)
        A list of tuples, where each tuple is of the form (protein_id, chain_id). This parameter is necessary
        because of where TERMANAL stores the fragments: <protein_id>_<chain_id>/fragments/.
    k : int, optional
        The number of individual fragments making up a super-fragment. The default is 2.
    out_dir : str, optional:
        THe directory where to write the super-fragments. The default is './' (current directory).
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    '''
    # Iterate over the PDB, chain IDs
    count = 1
    latest_bar = 1
    if verbose:
        print('Merging fragments...')
        progress_bar.start()
    for pdb_ch_id in pdb_ch_ids:
        if verbose:
            latest_bar = progress_bar.progress(count, len(pdb_ch_ids), latest_bar)
            count += 1
        p_id, c_id = pdb_ch_id
        dir_name = frag_dir + p_id + '_' + c_id + '/fragments/'
        pdb_frags = get_files(dir_name)
        # Sort the list to get the correct order of the fragments
        pdb_frags.sort()
        i = 0
        while i+k < len(pdb_frags):
            pdbs = pdb_frags[i:i+k]
            merge_pdbs(pdbs, out_dir=out_dir)
            i += k
        # Check if there are extra fragments left 
        if i < len(pdb_frags):
            if len(pdb_frags) - i == 1:
                a=1
            else:
                a=1