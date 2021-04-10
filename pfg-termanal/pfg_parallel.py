# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 01:21:46 2021

@author: FVS
"""

import os, sys, argparse
import subprocess as sub
import multiprocessing as mp
from prody import parsePDB
from pfg_functions import makeFragment, changeExt, contactList

def get_pdb_files(pdb_dir):
    files = []
    for file in os.listdir(pdb_dir):
        if file.endswith('.pdb'):
            files.append(pdb_dir + file)
    files.sort()
    return files

def run_termanal(SELF_BIN, out_dir, pdbf, ROTLIB):
    pdb_id = os.path.basename(pdbf)[:-4]
    
    term_support = 'termanal_support' + '_' + pdb_id
    if not os.path.exists(term_support):
        os.system('ln -s ' + SELF_BIN + '/support.default' + ' ' + term_support)
    else:
        print ('\'termanal_support\' exists in current directory, causing collision of file names. Exiting...')
        sys.exit(0)
        
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    if os.path.isdir(out_dir + '/' + pdb_id + '/fragments'):
        os.unlink(term_support)
        return # directory already exists so no need to run everything
        
    FRAGMENTS_OUT = out_dir + '/' + pdb_id + '/fragments'
    os.makedirs(FRAGMENTS_OUT, mode=0o777)
    
    # ConFind
    cmap_file = pdb_id + '.cmap'
    if not os.path.isfile(cmap_file):
        cmd_confind = [CONFIND, '--p', pdbf, '--o', cmap_file, '--rLib', ROTLIB]
        sub.call(cmd_confind)
    
    # Read PDB file
    protein = parsePDB(pdbf).select('protein').copy()
    residues = []
    for res in protein.iterResidues():
        residues.append(res)
        
    # TERMs
    for res in residues:
        cid, resnum = res.getChid(), res.getResnum()
        fragment_pdb = ''
        # make a list file that saves the information of contacts
        if cid == ' ':
            listname = pdb_id + '_' + str(resnum) + '.list'
        else:
            listname = pdb_id + '_' + cid + str(resnum) + '.list'
        fragment_pdb = changeExt(listname, 'pdb')
        if not os.path.isfile(FRAGMENTS_OUT + '/' + fragment_pdb):
            seeds = contactList(cmap_file, cid, resnum, FRAGMENTS_OUT + '/' + listname, dcut = 0.1)
            seeds.insert(0, cid + ',' + str(resnum))
            makeFragment(pdbf, seeds, FRAGMENTS_OUT + '/' + fragment_pdb)
            
    os.unlink(term_support)
    
def parallel_process(SELF_BIN, o, pdbf, ROTLIB, from_id):
    pdb_id = os.path.basename(pdbf)[:-4]
    if pdb_id[0] <= from_id:
        return
    run_termanal(SELF_BIN, o, pdbf, ROTLIB)
    
    
if __name__ == '__main__':
    SELF_BIN = os.path.dirname(os.path.realpath(sys.argv[0]))
    sys.path.insert(0, SELF_BIN + '/support.default')

    # parse arguments
    par = argparse.ArgumentParser()
    par.add_argument('p', type=str, help='The input PDB directory.')
    par.add_argument('o', type=str, help='The output directory.')
    par.add_argument('--cpu', type=int, default=8, help='The number of CPUs to use. The default is 8.')
    par.add_argument('--from_id', type=str, default=0, help='The digit from which the process should start. For example, if `s` is 1, then all files with the form 1xxx will be ignored. The default is 0 (i.e. nothing to ignore.')
    args = par.parse_args()

    SEARCHDB = SELF_BIN + '/support.default' + '/database/bc-30-sc-20141022'
    ROTLIB = SELF_BIN + '/support.default' + '/rotlib/RR2000.rotlib'
    MASTER = SELF_BIN
    CONFIND = SELF_BIN + '/confind'
    
    pdb_dir = args.p
    from_id = args.from_id
    
    pdb_files = get_pdb_files(pdb_dir)
    
    pool = mp.Pool(args.cpu)
    _ = [pool.apply(parallel_process, args=(SELF_BIN, args.o, pdbf, ROTLIB, from_id)) for pdbf in pdb_files]
    pool.close()
    
    