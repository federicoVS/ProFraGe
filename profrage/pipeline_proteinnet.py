# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:17:17 2021

@author: Federico van Swaaij
"""

import os
import argparse

from utils.protein_net import process_testing_set, process_training_set, process_validation_set
from utils.io import parse_protein_net

def pipeline(casp_dir, testing_dir, sqid=30, pct_thr=0.9, out_dir='protein-net/', do_test=True, do_validation=True, do_train=True, verbose=False):
    """
    Perform a full ProteinNet pipeline.
    
    The pipeline takes care in handling multiple chains. If a structure has multiple models, only the
    first one is selected. If multiple chains are detected, all are selected.
    
    The results are then filtered by removing duplicate/similar chains.

    Parameters
    ----------
    casp_dir : str
        The directory holding the ProteinNet splits.
    testing_dir : str
        The directory holding the PDB files for the testing set.
    sqid : int, optional
        the sequence identity percentage, which should be the same as the PDB dataset. The default is 30.
    pct_thr : float in [0,1], optional
        The percentage threshold above which two sequences are similar. The default is 0.9.
    out_dir : str, optional
        The output directory. The default is 'protein-net/'.
    do_test : bool, optional
        Whether to process the testing split. The default is True.
    do_validation : bool, optional
        Whether to process the validation split. The default is True.
    do_train : bool, optional
        Whether to process the training split. The default is True.
    verbose : bool, optional
        Whether to print progress information. The default is False.

    Returns
    -------
    None.
    """
    # Create out directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # First process all directories
    # Process testing set
    if do_test:
        process_testing_set(testing_dir, out_dir, verbose=verbose)
    # Process validation set
    if do_validation:
        pn_dict = parse_protein_net(casp_dir + 'validation')
        process_validation_set(pn_dict, out_dir, sqid, verbose=verbose)
    # Process training set
    if do_train:
        pn_dict = parse_protein_net(casp_dir + 'training_' + str(sqid))
        process_training_set(pn_dict, out_dir, verbose=verbose)
        
if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full ProteinNet dataset pipeline.')
    arg_parser.add_argument('casp_dir', type=str, help='The directory containing the ProteinNet splits.')
    arg_parser.add_argument('testing_dir', type=str, help='The directory containing the testing PDB files.')
    arg_parser.add_argument('--sqid', type=int, default=30, help='The sequence identity percentage. The default is 30.')
    arg_parser.add_argument('--pct_thr', type=float, default=0.9, help='The percentage threshold above which two sequences are equal. The default is 0.9.')
    arg_parser.add_argument('--out_dir', type=str, default='protein-net/', help='The output directory. The default is protein-net/.')
    arg_parser.add_argument('--do_test', type=bool, default=True, help='Whether to process the testing split. The default is True.')
    arg_parser.add_argument('--do_validation', type=bool, default=True, help='Whether to process the validation split. The default is True.')
    arg_parser.add_argument('--do_train', type=bool, default=True, help='Whether to process the training split. The default is True.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args = arg_parser.parse_args()
    # Begin pipeline
    pipeline(args.casp_dir, args.testing_dir, sqid=args.sqid, pct_thr=args.pct_thr, out_dir=args.out_dir, do_test=args.do_test, do_validation=args.do_validation, do_train=args.do_train, verbose=args.verbose)
    