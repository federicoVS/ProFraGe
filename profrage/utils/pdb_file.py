# -*- coding: utf-8 -*-
"""
@author: FVS
"""

import os
from prody.proteins.pdbfile import fetchPDB, parsePDB, writePDB

def read_pdb_ids_file(file_name):
    '''
    Reads the content of the file holding the IDs of the proteins. Such file is downloaded from
    https://www.rcsb.org/search, after having specified criteria for the proteins to download via the
    Advance Search settings.
    
    Parameters
    ----------
    file_name : str
        The name of the file to read.
    
    Returns
    -------
    pdb_ids : list of str
        The list holding the protein IDs.
    '''
    pdb_ids = open(file_name, 'r').read().split(',')
    return pdb_ids
        
def merge_pdbs(pdbs, out_dir='./', sep='+'):
    '''
    Merges the specified PDB files into the specified output directory.
    
    TODO: See how the mmtf_file merging go, if it works then remove this.
    
    Parameters
    ----------
    pdbs : list of str
        The PDB files to be merged.
    out_dir : str, optional
        The output directory. The default is the './' (current directory).
    sep : str, optional
        The separator used in the filename to explicitly distinguish between the proteins in the newly
        created PDB file. The default is +.
        
    Returns
    -------
    None.
    '''
    prots = []
    for pdb in pdbs:
        prots.append(parsePDB(pdb))
    merged_prots = prots[0]
    for i in range(1, len(prots)):
        merged_prots += prots[i]
    merged_prot_name = os.path.basename(pdbs[0])[:-4] + sep
    for i in range(1, len(pdbs)-1):
        merged_prot_name += os.path.basename(pdbs[i]).split('_')[1][:-4] + sep
    merged_prot_name += os.path.basename(pdbs[len(pdbs)-1]).split('_')[1]
    file_name = out_dir + merged_prot_name
    writePDB(file_name, prots)

def fetch_pdb(pdb_ch_id, pdb_gz_dir=None, out_dir='./', remove_pdb_gz=False):
    '''
    Fetches the specified chains of the specified proteins from the PDB using the ProDy API. The fetched
    PDB file <pdb_id>_<chain_id>.pdb is then written to the specified location.
    
    Parameters
    ----------
    pdb_ch_id : (str, str)
        The ID the protein to download and which chain to save.
    pdb_dir : None
        The directory holding the protein file (compressed, i.e. .pdb.gz). The default is None, in which
        case ProDy will download it. Since the ProDy download time is very slow, it is encouraged to
        download all the desired proteins beforehand.
    out_dir : str, optional
        The directory where to save the file. The default is './' (current directory).
    remove_pdb_gz : bool, optional
        Whether to remove the original compressed PDB file. The default is False.

    Returns
    -------
    None.
    '''
    p_id, c_id = pdb_ch_id
    pdb_file = None
    if pdb_gz_dir is None:
        pdb_file = fetchPDB(p_id, chain=c_id)
    else:
        pdb_file = pdb_gz_dir + p_id + '.pdb.gz'
    if pdb_file is not None and os.path.isfile(pdb_file):
        ag = parsePDB(pdb_file, chain=c_id)
        if ag is not None:
            file_name = out_dir + p_id + '_' + c_id + '.pdb'
            writePDB(file_name, ag)
        if remove_pdb_gz:
            if os.path.isfile(pdb_file):
                os.remove(pdb_file)
