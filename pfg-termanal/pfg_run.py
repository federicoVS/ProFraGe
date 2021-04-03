'''
----------------------------------------------------------------------------
This file is part of TERMANAL.

TERMANAL is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

TERMANAL is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with TERMANAL.  If not, see <http://www.gnu.org/licenses/>.

Copyright (C) 2015 Fan Zheng, Gevorg Grigoryan
----------------------------------------------------------------------------
'''

import os, sys, glob, shutil, math, argparse
import subprocess as sub
from time import sleep
from prody import *

# set up path to support files
SELF_BIN = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.insert(0, SELF_BIN + '/support.default')
from pfg_functions import makeFragment, changeExt, contactList

# parse arguments
par = argparse.ArgumentParser()
par.add_argument('--p', required = True, help = 'input PDB file')
par.add_argument('--m', default = SELF_BIN, help = 'path to MASTER binary. By default, assumes MASTER resides in the same directory as this script.')
par.add_argument('--c', default = SELF_BIN + '/confind', help = 'path to ConFind binary. By default, assumes ConFind resides in the same directory as this script.')
par.add_argument('--d', default = SELF_BIN + '/support.default', help = '''path to support directory, which must contain directories database/ (with the MASTER database) and rotlib/ (with the rotamer library).
By default, assumes the that support directory is in support.default/ under the directory of this script.''')
par.add_argument('--v', action = 'store_true', help = 'if set to be true, keep the intermediate files and data')
par.add_argument('--o', help = 'specify another location of the output')
par.add_argument('--dontuse', help = 'a file with some strings, and if the path of database targets contains these strings, they are excluded from the result')
args = par.parse_args()

SEARCHDB = args.d + '/database/bc-30-sc-20141022'
ROTLIB = args.d + '/rotlib/RR2000.rotlib'
MASTER = args.m
CONFIND = args.c
pdbf = args.p
dir = os.path.dirname(os.path.realpath(pdbf))
base = os.path.splitext(os.path.basename(pdbf))[0]

# no matter what is the current directory, create a simlink called 'termnal_support' to 'support.default'.
# if 'termanal_support already exists in current directory, output an error
term_support = 'termanal_support'
if not os.path.exists(term_support):
    os.system('ln -s ' + args.d + ' ' + term_support)
else:
    print ('\'termanal_support\' exists in current directory, causing collision of file names. Exiting...')
    exit(0)

# if set another location for output
if not args.o == None:
    dir = args.o
    if not os.path.isdir(dir):
        os.makedirs(dir)
        
if os.path.isdir(dir + '/' + base + '/fragments'):
    sys.exit() # directory already exists so no need to run everything

# set up and create working directory paths
FRAGMENTS_OUT = dir + '/' + base + '/fragments'
SCORES_OUT = dir
os.makedirs(FRAGMENTS_OUT, mode=0o777)

# run confind to get contacts
# print('Running ConFind...')
cmap_file = base + '.cmap'
if not os.path.isfile(cmap_file):
    cmd_confind = [CONFIND, '--p', pdbf, '--o', cmap_file, '--rLib', ROTLIB]
    sub.call(cmd_confind)
# print('Done with ConFind...')

# read in PDB file
protein = parsePDB(pdbf).select('protein').copy()
residues = []
for res in protein.iterResidues():
    residues.append(res)
head = ['t1k', 'uniq_t1k', 'hit1', 't1k_hit1', 'uniq_t1k_hit1']

# # generate TERMs
# print('Generating TERMs...')
for res in residues:
    cid, resnum = res.getChid(), res.getResnum()
    fragment_pdb = ''
    # make a list file that saves the information of contacts
    if cid == ' ':
        listname = base+ '_' + str(resnum) + '.list'
    else:
        listname = base+ '_' + cid + str(resnum) + '.list'
    fragment_pdb = changeExt(listname, 'pdb')
    if not os.path.isfile(FRAGMENTS_OUT + '/' + fragment_pdb):
        seeds = contactList(cmap_file, cid, resnum, FRAGMENTS_OUT + '/' + listname, dcut = 0.1)
        seeds.insert(0, cid + ',' + str(resnum))
        makeFragment(pdbf, seeds, FRAGMENTS_OUT + '/' + fragment_pdb)
# print('Done generating TERMs...')
