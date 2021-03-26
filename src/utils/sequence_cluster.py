# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:09:40 2021

@author: FVS
"""

import pickle
import matplotlib.pyplot as plt
from prody.proteins.pdbclusters import fetchPDBClusters, loadPDBClusters
from search import binary_search
import progress_bar

def match_clusters(pdb_ids, clusters=None, sqid=30, out_file=None, verbose=False):
    '''
    For each cluster, take the all the chains of the first protein, if it is matched to any in the database.
    If the first is not is the database, take the second, third, and so on.
    The need for this function is that the proteins contained in the cluster may not match the proteins
    found at https://www.rcsb.org/search using the Advance Search settings.
    
    
    Parameters
    ----------
    pdb_ids : list of str
        Alist holding the IDs of the proteins in the dataset.
    clusters : list of object, optional
        The clusters, if loaded. The default is None (not loaded).
    sqid : int, optional
        The sequence identity for the sequence clustering. The default is 30.
    out_file : str, optional
        The file where to write the resulting list. The default is None (not write).
    verbose : bool, optional
        Whether to print progress information. The default is False.
    
    Returns
    -------
    m_pdb_ch_ids : list of (string, string)
        A list of tuples of string, each holding the ID of the matching protein in the dataset and the ID of
        one of its chains.
    '''    
    if clusters is None:
        if verbose:
            print('Fetching clusters...')
        fetchPDBClusters(sqid=sqid)
        clusters = loadPDBClusters()
    # Sort list of PDB IDs for binary search
    pdb_ids.sort()
    m_pdb_ch_ids = []
    count = 1
    latest_bar = 1
    if verbose:
        print('Matching clusters with protein chains...')
        progress_bar.start()
    for cluster in clusters:
        if verbose:
            latest_bar = progress_bar.progress(count, len(clusters), latest_bar)
            count += 1
        last_p = None
        for i in range(len(cluster)):
            p_id, c_id = cluster[i] # protein id, chain id
            if last_p is not None and p_id == last_p:
                m_pdb_ch_ids.append((p_id, c_id))
            elif last_p is not None and p_id != last_p:
                break
            else:
                if binary_search(p_id, pdb_ids, sorted=True):
                    m_pdb_ch_ids.append((p_id, c_id))
                    last_p = p_id
    if out_file is not None:
        with open(out_file, 'wb') as f:
            pickle.dump(m_pdb_ch_ids, f)
    if verbose:
        progress_bar.end()
        print(f'Found {len(m_pdb_ch_ids)} matching protein chains.')
    return m_pdb_ch_ids

def get_clusters(sqid=30, loaded=False, show=False, verbose=False):
    '''
    Get the sequence clusters resulting from the weekly BLAST clustering.
    
    Parameters
    ----------
    sqid : int, optional
        The sequence identity for the sequence clustering. The default is 30.
    loaded : bool, optional
        Whether the clusters have already been loaded. The default is False.
    show : bool, optional
        Whether to show an histogram of the clusters densities. The default is False.
    verbose : bool, optional
        Whether to print the number of clusters found. The default is False.
    
    Returns
    -------
    clusters : list of object
        The list of clusters.
    '''
    if verbose:
        print('Fetching clusters...')
    if not loaded:
        fetchPDBClusters(sqid=sqid)
    clusters = loadPDBClusters()
    if verbose:
        print(f'Found {len(clusters)} clusters.')
    if show:
        freqs = {}
        for cluster in clusters:
            size = len(cluster)
            if size in freqs:
                freqs[size] += 1
            else:
                freqs[size] = 1
        plt.bar(freqs.keys(), freqs.values(), 1.0, color='b')
        plt.show()
    return clusters