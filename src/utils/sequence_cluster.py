# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:09:40 2021

@author: FVS
"""

import pickle
import matplotlib.pyplot as plt
from prody.proteins.pdbclusters import fetchPDBClusters, loadPDBClusters

def match_clusters(pdb_ids, clusters=None, sqid=30, out_file=None, verbose=False):
    '''
    Matches the proteins in the current dataset against elements of a cluster. The need for this function is
    that the proteins contained in the cluster may not match the proteins found at https://www.rcsb.org/search
    using the Advance Search settings.
    
    Parameters
    ----------
    pdb_ids : list<string>
        Alist holding the IDs of the proteins in the dataset.
    clusters : list<?>, optional
        The clusters, if loaded. The default is None (not loaded).
    sqid : int, optional
        The sequence identity for the sequence clustering. The default is 30.
    out_file : string, optional
        The file where to write the resulting list. The default is None (not write).
    verbose : bool, optional
        Whether to print progress information. The default is False.
    
    Returns
    -------
    m_clusters : list<(string, string)>
        A list of tuples of the form (p_id, c_id), matching with the proteins in the dataset.
    '''
    if clusters is not None:
        fetchPDBClusters(sqid=sqid)
        clusters = loadPDBClusters()
    count = 1
    m_clusters = []
    for cluster in clusters:
        if verbose:
            print(f'Processing cluster {count}/{len(clusters)}')
            count += 1
        for i in range(len(cluster)):
            p_id, c_id = cluster[i]
            for pdb_id in pdb_ids:
                if pdb_id == p_id:
                    m_clusters.append((p_id, c_id))
                    break
    if out_file is not None:
        with open(out_file, 'wb') as f:
            pickle.dump(m_clusters, f)
    return m_clusters

def get_clusters(sqid=30, show=False, verbose=False):
    '''
    Get the sequence clusters resulting from the weekly BLAST clustering.
    
    Parameters
    ----------
    sqid : int, optional
        The sequence identity for the sequence clustering. The default is 30.
    show : bool, optional
        Whether to show an histogram of the clusters densities. The default is False.
    verbose : bool, optional
        Whether to print the number of clusters found. The default is False.
    
    Returns
    -------
    clusters : list<?>
        The list of clusters.
    '''
    fetchPDBClusters(sqid=sqid)
    clusters = loadPDBClusters()
    if verbose:
        print(f'Found {len(clusters)} clusters')
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