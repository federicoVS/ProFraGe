# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:01:08 2021

@author: Federico van Swaaij
"""

import os
import argparse

from fragment.mine import KSeqMiner, KSeqTerMiner, KTerCloseMiner, LeidenMiner, HierarchyMiner, FuzzleMiner
from fragment.builders import Neighborhoods, MITStructure
from structure.representation import USR, MITResidue
from utils.io import get_files, from_mmtf, from_pdb
from utils.ProgressBar import ProgressBar

def pipeline(method, build, represent, data_dir, out_dir, cmaps, fuzzle_json, k, max_inters, max_size, score_thr, f_thr, ext, verbose=False):
    """
    Mine fragments from the specified proteins with the specified generation method.

    Parameters
    ----------
    method : str
        The method to use to mine the fragments from the proteins.
    build : str
        The builder class to use to build the fragments.
    represent : str
        The representation to use on the structures.
    data_dir : str
        The directory holding the protein data in MMTF format.
    out_dir : str
        The directory where the fragments will be written.
    cmaps : str
        The directory holding the CMAP files.
    fuzzle_json : str
        The JSON file describing the Fuzzle fragment network.
    k : int
        The number of residues to the left and right of the centroid in a neighborhood.
    max_inters : int
        The maximum number of interactions a neighborhood can have.
    max_size : int
        The maximum number of neighborhoods per fragment.
    score_thr : float in [0,1]
        The cosine similarity threshold. A higher threshold implies a higher similitude is needed.
    f_thr : float in [0,1]
        The interaction threshold applied to CMAP interaction. A higher threshold implies more interaction is needed.
    ext : str
        The extension of the input files.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    if method != 'fuzzle':
        # Get CMAP entries in a dictionary
        cmap_dict = {}
        cmapsf = get_files(cmaps, ext='.cmap')
        for cmapf in cmapsf:
            cmap_id = os.path.basename(cmapf)[:-5]
            cmap_dict[cmap_id] = cmapf
        # Get proteins files
        s_files = get_files(data_dir, ext=ext)
        progress_bar = ProgressBar()
        count = 1
        if verbose:
            print('Mining fragments...')
            progress_bar.start()
        for s_file in s_files:
            if verbose:
                progress_bar.step(count, len(s_files))
                count += 1
            # Get pdb ID
            pdb_id = ''
            structure = None
            if ext == '.mmtf':
                structure = from_mmtf(s_file)
                pdb_id = os.path.basename(s_file)[:-5]
            elif ext == '.pdb':
                structure = from_pdb(os.path.basename(s_file)[:-4], s_file, quiet=True)
                pdb_id = os.path.basename(s_file)[:-4]
            # Check if the structure is valid
            if structure is None:
                print('Invalid protein')
                return
            # Get representation
            Rep = None
            if represent == 'usr':
                Rep = USR
            elif represent == 'mitr':
                Rep = MITResidue
            else:
                print('Invaid representation')
                return
            # Get builder
            builder = None
            if build == 'neighs':
                # Check CMAP validity
                if pdb_id not in cmap_dict:
                    print('Invalid CMAP')
                    return
                builder = Neighborhoods(structure, Rep, cmap_dict[pdb_id], k)
            elif build == 'mits':
                builder = MITStructure(structure)
            else:
                print('Invaid builder')
                return
            # Get mining method
            miner = None
            if method == 'kseq':
                miner = KSeqMiner(structure, Rep, k=k,score_thr=score_thr, max_inters=max_inters)
            elif method == 'kseqter':
                miner = KSeqTerMiner(structure, Rep, cmap_dict[pdb_id], k=k, score_thr=score_thr, f_thr=f_thr, max_inters=max_inters, max_size=max_size) # CMAP validity already checked
            elif method == 'kterclose':
                miner = KTerCloseMiner(structure, Rep, cmap_dict[pdb_id], k=k, score_thr=score_thr, f_thr=f_thr, max_inters=max_inters)
            elif method == 'leiden':
                miner = LeidenMiner(structure, cmap_dict[pdb_id], f_thr=f_thr)
            elif method == 'hierarchy':
                miner = HierarchyMiner(structure, builder)
            else:
                print('Invaid mining method')
                return
            # Generate the fragments
            miner.mine()
            # Build proper output directory
            dir_name = method + '/'
            if not os.path.exists(out_dir+dir_name):
                os.makedirs(out_dir+dir_name)
            # Save the fragments in the specified directory
            miner.save(out_dir+dir_name)
        if verbose:
            progress_bar.end()
    else:
        # Get generation method
        miner = None
        if method == 'fuzzle':
            miner = FuzzleMiner(fuzzle_json, verbose=verbose)
        else:
            print('Invaid mining method')
            return
        # Generate the fragments
        miner.mine()
        # Build proper output directory
        dir_name = method + '/'
        if not os.path.exists(out_dir+dir_name):
            os.makedirs(out_dir+dir_name)
        # Save the fragments in the specified directory
        miner.save(out_dir+dir_name)


if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Mining of fragments.')
    arg_parser.add_argument('method', type=str, help='The method to be used to the mine the fragments. The options are [kseq, kseqter, kterclose, leiden, hierarchy, fuzzle].')
    arg_parser.add_argument('build', type=str, help='The builder class to be used represent a fragment. The options are [neighs, mits].')
    arg_parser.add_argument('represent', type=str, help='The representation of a protein. The options are [usr, mitr].')
    arg_parser.add_argument('--data_dir', type=str, default='../pdb/data/', help='The directory containing the proteins. The default is ../pdb/data/')
    arg_parser.add_argument('--out_dir', type=str, default='../pdb/fragments/', help='The base directory where the fragments will be saved. the default is ../pdb/fragments/.')
    arg_parser.add_argument('--cmaps', type=str, default='../pdb/cmaps/', help='The directory holding the CMAP files generated by ConFind. This option has effect when methods `kseq`, `kseqter`, `kterclose`, `leiden`, and `hierarchy` are chosen. The default is ../pdb/cmaps/.')
    arg_parser.add_argument('--fuzzle_json', type=str, default='../pfg-fuzzle/psiblast2.06.1.25-3.json', help='The JSON file describing the Fuzzle fragment network. This option has effect when method `fuzzle` is chosen. The default is ../pfg-fuzzle/psiblast2.06.1.25-3.json.')
    arg_parser.add_argument('--k', type=int, default=3, help='The number of residues to the left or right of a centroid in a neighborhood. This option has effect when methods `kseq`, `kseqter`, `kterclose`, `leiden` and `hierarchy` are chosen. The default is 3.')
    arg_parser.add_argument('--max_inters', type=int, default=1, help='The maximum number of interactions a neighborhood can have. This option has effect when methods `kseq`, `kseqter`, and `kterclose`are chosen. The default is 1.')
    arg_parser.add_argument('--max_size', type=int, default=4, help='The maximum number of neighborhoods per fragment. This option has effect when methods `kseq`, `kseqter`, `kterclose`, `leiden` and `hierarchy` are chosen. The default is 4.')
    arg_parser.add_argument('--score_thr', type=float, default=0.4, help='The cosine threshold. The smaller the more similar. This option has effect when methods `kseq`, `kseqter`, and `kterclose` are chosen. The default is 0.4.')
    arg_parser.add_argument('--f_thr', type=float, default=0.1, help='The residue-residue interaction score threshold. The larger the more similar. This option has effect when methods `kseq`, `kseqter`, `kterclose`, `leiden`, and `hierarchy` are chosen. The default is 0.1.')
    arg_parser.add_argument('--ext', type=str, default='.pdb', help='The format of the input files. The options are [.mmtf, .pdb]. This option has effect when methods `kseq`, `kseqter`, `kterclose`, `leiden`, and `hierarchy` are chosen. The deafult is .pdb.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Begin pipeline
    pipeline(args.method, args.build, args.represent, args.data_dir, args.out_dir, args.cmaps, args.fuzzle_json, args.k, args.max_inters, args.max_size, args.score_thr, args.f_thr, args.ext, verbose=args.verbose)