# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:01:08 2021

@author: FVS
"""

import os
import argparse
from fragments.generate import CCGen, SeqGen, FuzzleGen
from utils.io import get_files, from_mmtf, from_pdb
from utils.ProgressBar import ProgressBar

def pipeline(method, data_dir, out_dir, radius, k, fuzzle_json, ext, verbose=False):
    """
    Generate fragments from the specified proteins with the specified generation method.

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
    fuzzle_json : str
        The JSON file describing the Fuzzle fragment network.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    if ext is not None:
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
            # Get generation method
            generator = None
            if method == 'ccg':
                generator = CCGen(structure, radius)
            elif method == 'seq':
                generator = SeqGen(structure, k)
            else:
                print('Invaid generation method')
                return
            # Generate the fragments
            generator.generate()
            # Build proper output directory
            dir_name = method + '/'
            if not os.path.exists(out_dir+dir_name):
                os.makedirs(out_dir+dir_name)
            # Save the fragments in the specified directory
            generator.save(out_dir+dir_name)
        if verbose:
            progress_bar.end()
    else:
        # Get generation method
        generator = None
        if method == 'fuzzle':
            generator = FuzzleGen(fuzzle_json, verbose=verbose)
        else:
            print('Invaid generation method')
            return
        # Generate the fragments
        generator.generate()
        # Build proper output directory
        dir_name = method + '/'
        if not os.path.exists(out_dir+dir_name):
            os.makedirs(out_dir+dir_name)
        # Save the fragments in the specified directory
        generator.save(out_dir+dir_name)


if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Generation of fragments.')
    arg_parser.add_argument('method', type=str, help='The method to be used to the generate the fragments. The options are [ccg, seq, fuzzle, terms].')
    arg_parser.add_argument('--out_dir', type=str, default='../pdb/fragments/', help='The base directory where the fragments will be saved. the default is ../pdb/fragments/.')
    arg_parser.add_argument('--data_dir', type=str, default='../pdb/data/', help='The directory containing the proteins. The default is ../pdb/data/')
    arg_parser.add_argument('--radius', type=float, default=3, help='The cutoff radius in Angstroms. This option is only effective when method `seq` is chosen. The default is 3.')
    arg_parser.add_argument('--k', type=int, default=10, help='The number of connected components/contiguous sequences. The default is 10.')
    arg_parser.add_argument('--fuzzle_json', type=str, default='../pfg-fuzzle/psiblast2.06.1.25-3.json', help='The JSON file describing the Fuzzle fragment network. This option is only effective when method `fuzzle` is chosen. The default is ../pfg-fuzzle/psiblast2.06.1.25-3.json.')
    arg_parser.add_argument('--ext', type=str, default=None, help='The format of the input files. The options are [.mmtf, .pdb]. Note that methods `ccg` and `seq` need one of them. The deafult is None.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Begin pipeline
    pipeline(args.method, args.data_dir, args.out_dir, args.radius, args.k, args.fuzzle_json, args.ext, verbose=args.verbose)