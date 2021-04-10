# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:01:08 2021

@author: FVS
"""

import os
import argparse
from fragments.generate import CCG, SeqG
from utils.io import get_files, from_mmtf, from_pdb
from utils.ProgressBar import ProgressBar

def pipeline(method, data_dir, out_dir, radius, k, ext, verbose=False):
    '''
    This pipeline generates fragments from the specified proteins with the specified generation
    method.

    Parameters
    ----------
    method : str
        The method to use to generate the proteins.
    data_dir : str
        The directory holding the protein data in MMTF format.
    out_dir : str
        The directory where the fragments will be written.
    radius : float
        The cutoff radius in Angstroms.
    k : int
        The number of spectral partitions.
    ext : str
        The extension of the input files.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    '''
    s_files = get_files(data_dir, ext=ext)
    progress_bar = ProgressBar()
    count = 1
    if verbose:
        print('Generating fragments...')
        progress_bar.start()
    for s_file in s_files:
        if verbose:
            progress_bar.step(count, len(s_files))
            count += 1
        structure = None
        if ext == '.mmtf':
            structure = from_mmtf(s_file)
        elif ext == '.pdb':
            structure = from_pdb(os.path.basename(s_file)[:-4], s_file, quiet=True)
        else:
            print('Invalid file extension.')
            return
        generator = None
        if method == 'ccg':
            generator = CCG(structure, radius)
        elif method == 'seq':
            generator = SeqG(structure, k)
        else:
            print('Invaid generation method')
            return
        # Generate the fragments
        generator.generate()
        # Save the fragments in the specified directory
        generator.save(out_dir)
    if verbose:
        progress_bar.end()


if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Generation of fragments.')
    arg_parser.add_argument('method', type=str, help='The method to be used to the generate the fragments. The options are [ccg, seq].')
    arg_parser.add_argument('out_dir', type=str, help='The directory where the fragments will be saved.')
    arg_parser.add_argument('--data_dir', type=str, default='../pdb/data/', help='The directory containing the proteins in MMTF format. The default is ../pdb/data/')
    arg_parser.add_argument('--radius', type=float, default=3, help='The cutoff radius in Angstroms. This option is only effective when method `seq` is chosen. The default is 3.')
    arg_parser.add_argument('--k', type=int, default=10, help='The number of spectral partitions. This option is only effective when method `knn` is chosen. The default is 10.')
    arg_parser.add_argument('--ext', type=str, default='.mmtf', help='The format of the input files. The deafult is .mmtf.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Begin pipeline
    pipeline(args.method, args.data_dir, args.out_dir, args.radius, args.k, args.ext, verbose=args.verbose)