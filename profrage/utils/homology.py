import os

import shutil

from prody.proteins.blastpdb import blastPDB

from utils.structure import get_sequences
from utils.io import from_pdb, get_files
from utils.ProgressBar import ProgressBar

def filter_homologues(pdb_dir, pct_identity=0.3, pct_coverage=0.9, out_dir='homologues/', verbose=False):
    """
    Filter the given structures for homologs.
    
    Each structure sequence is used as the base for a Blast search, and it is clustered with all
    matching sturctures: such clustering depends on the provided parameters.
    
    The representatives of each cluster are then written in the specified output directory.

    Parameters
    ----------
    pdb_dir : str
        The PDB dataset.
    pct_identity : float in [0,1], optional
        The minimum percentage sequence identity. The default is 0.3.
    pct_coverage : float in [0,1], optional
        The minimum percent coverage. The default is 0.9.
    out_dir : str, optional
        The output directory. The default is 'homologues/'.
    verbose : bool, optional
        Whether to print progess information. The default is False.

    Returns
    -------
    None.
    """
    # Create output directory if it does not exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # Get PDB files
    pdbfs = get_files(pdb_dir)
    # Define data structures
    n = len(pdbfs)
    assigned = {}
    homologues_dict = {}
    progress_bar = ProgressBar(n)
    if verbose:
        progress_bar.start()
    # Iterate over each structure, and perform a Blast search over each one, if its not already selected
    for i in range(n):
        if verbose:
            progress_bar.step()
        p_id, ch_id = os.path.basename(pdbfs[i])[:-4].split('_')
        structure = from_pdb(p_id+'_'+ch_id, pdbfs[i], quiet=True)
        if p_id not in assigned:
            seq = get_sequences(structure)[0] # take the first sequence
            try:
                blast_record = blastPDB(str(seq))
                hits = blast_record.getHits(percent_identity=pct_identity, percent_overlap=pct_coverage)
            except:
                continue # skip iteration
            assigned[p_id] = True
            homologues_dict[p_id] = ch_id
            hits_list = list(hits)
            for hit in hits_list:
                assigned[hit] = True
    if verbose:
        progress_bar.end()
    # Iterate over the keys, and select the first component of each cluster
    if verbose:
        print('Writing filtered proteins...')
    for key in homologues_dict.keys():
        p_id, ch_id = key, homologues_dict[key]
        pdb_id = p_id + '_' + ch_id
        shutil.copy(pdb_dir + pdb_id + '.pdb', out_dir + pdb_id + '.pdb')