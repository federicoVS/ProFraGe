# -*- coding: utf-8 -*-
"""
@author: FVS
"""

import os
from prody.proteins.pdbfile import fetchPDB, parsePDB, writePDB
from prody.proteins.functions import showProtein
import progress_bar

def get_files(data_dir, ext='.pdb'):
    '''
    Returns a list of files with the desired extension from the specified directory.
    
    Parameters
    ----------
    data_dir : str
        The name of the directory.
    ext : str, optional
        The file extension. The default is '.pdb'
    
    Returns
    -------
    files : list of str
        The list containing the files.
    '''
    files = []
    for file in os.listdir(data_dir):
        if file.endswith(ext):
            files.append(data_dir+file)
    return files

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
        
def merge_pdbs(pdb_1, pdb_2, out_dir='./', show=False):
    '''
    Merges two PDB files into the specified output directory.
    
    Parameters
    ----------
    pdb_1 : str
        The first PDB file.
    pdb_2 : str
        The second PDB file.
    out_dir : str, optional
        The output directory. The default is the './' (current directory).
    show : bool, optional
        Whether to plot the two proteins. The default is False.
        
    Returns
    -------
    None.
    '''
    prot_1 = parsePDB(pdb_1)
    prot_2 = parsePDB(pdb_2)
    prots = prot_1 + prot_2
    if show :
        showProtein(prot_1, prot_2)
    file_name = out_dir + os.path.basename(pdb_1)[:-4] + '_' + os.path.basename(pdb_2).split('_')[1]
    writePDB(file_name, prots)

def fetch_pdb(pdb_ch_id, out_dir='./'):
    '''
    Fetches the specified chains of the specified proteins from the PDB using the ProDy API. The fetched
    PDB file <pdb_id>_<chain_id>.pdb is then written to the specified location.
    
    Parameters
    ----------
    pdb_ch_id : (str, str)
        The ID the protein to download and which chain to save.
    out_dir : str, optional
        The directory where to save the file. The default is './' (current directory).

    Returns
    -------
    None.
    '''
    p_id, c_id = pdb_ch_id
    pdb_file = fetchPDB(p_id, chain=c_id)
    if pdb_file is not None:
        ag = parsePDB(pdb_file)
        if ag is not None:
            file_name = out_dir + p_id + '_' + c_id + '.pdb'
            writePDB(file_name, ag)
        # Remove compressed file
        if os.path.isfile(pdb_file):
            os.popen('rm ' + pdb_file)
