# -*- coding: utf-8 -*-

import os
from prody.proteins.pdbfile import fetchPDB, parsePDB, writePDB
from prody.proteins.functions import showProtein

def get_files(data_dir, ext='.pdb'):
    '''
    Returns a list of files with the desired extension from the specified directory.
    
    Parameters
    ----------
    data_dir : string
        The name of the directory.
    ext : string, optional
        The file extension. The default is '.pdb'
    
    Returns
    -------
    files : list<string>
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
    file_name : string
        The name of the file to read.
    
    Returns
    -------
    pdb_ids : list<string>
        The list holding the protein IDs.
    '''
    pdb_ids = open(file_name, 'r').split(',')
    return pdb_ids
        
def merge_pdbs(pdb_1, pdb_2, out_dir='./', show=False):
    '''
    Merges two PDB files into the specified output directory.
    
    Parameters
    ----------
    pdb_1 : string
        The first PDB file.
    pdb_2 : string
        The second PDB file.
    out_dir : string, optional
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

def fetch_pdbs(pdb_ch_ids, out_dir='./', verbose=False):
    '''
    Fetches the specified chains of the specified proteins from the PDB using the ProDy API. The fetched
    PDB file <p_id>_<c_id>.pdb is then written to the specified location.
    
    Parameters
    ----------
    pdb_ch_ids : list<(string, string)>
        The list of the IDs of the (p_id, c_id) tuples to download. This means that chain c_id of protein
        p_id will be downloaded.
    out_dir : string, optional
        The directory where to save the files. The default is './' (current directory).
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    '''
    count = 1
    for pdb_ch_id in pdb_ch_ids:
        p_id, c_id = pdb_ch_id
        if verbose:
            print(f'Fetching {p_id}, {count}/{len(pdb_ch_ids)}')
            count += 1
        pdb_file = fetchPDB(p_id)
        if pdb_file is not None:
            ag = parsePDB(pdb_file, chain=c_id)
            if ag is not None:
                file_name = out_dir + p_id + '_' + c_id + '.pdb'
                writePDB(file_name, ag)
    
# file = '../../../../../../data/pdb/raw/3DSD.pdb'
# data_path = '../../../../../../data/pdb/raw/'

# fragment_1 = '1A31_A474.pdb'
# fragment_2 = '1A31_A475.pdb'
