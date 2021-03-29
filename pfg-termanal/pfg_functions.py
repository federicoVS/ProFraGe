import os, re, time
import subprocess as sub
from time import sleep
from Bio.PDB.mmtf.mmtfio import MMTFIO
from Bio.PDB import PDBParser
from prody import *
confProDy(verbosity='critical')

## added ##
from getpass import getuser
##

a2aaa = {
'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 
'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 
'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 
'Y': 'TYR', 'V': 'VAL'
}
aaa2a = {aaa:a for a, aaa in a2aaa.items()}
# add unnatural amino acids (the most common ones)
aaa2a['ASX'] = aaa2a['ASN']
aaa2a['CSO'] = aaa2a['CYS']
aaa2a['GLX'] = aaa2a['GLU'] # or GLN
aaa2a['HIP'] = aaa2a['HIS']
aaa2a['HSC'] = aaa2a['HIS']
aaa2a['HSD'] = aaa2a['HIS']
aaa2a['HSE'] = aaa2a['HIS']
aaa2a['HSP'] = aaa2a['HIS']
aaa2a['MSE'] = aaa2a['MET']
aaa2a['SEC'] = aaa2a['CYS']
aaa2a['SEP'] = aaa2a['SER']
aaa2a['TPO'] = aaa2a['THR']
aaa2a['PTR'] = aaa2a['TYR']
aaa2a['XLE'] = aaa2a['LEU'] # or ILE

def changeExt(file, ext):
    return file.rpartition('.')[0] + '.' + ext

def ConResDict(pdbf):
    mol = parsePDB(pdbf)
    residues = {}
    for res in mol.iterResidues():
        if res.getResname() in aaa2a:
            cid, resnum = res.getChid(), res.getResnum()
            residues[cid+','+str(resnum)] = res
    return residues

def contactList(cmap, chain, resnum, outFile = None, dcut = 0.1):
    mapfile = open(cmap)
    if outFile != None:
        out = open(outFile, 'w')
    pid = os.path.basename(cmap).rpartition('.')[0]
    center = chain +','+ str(resnum)
    cons = []
    N = 0
    for line in mapfile:
        if not re.match('contact', line):
            continue 
        if line.find(center+'\t') >= 0:
            larr = line.split()
        else:
            continue
        if chain == ' ':
            larr[1] = ' ' + larr[1]
            larr[2] = ' ' + larr[2]
        if larr[1] == center:
            [contact, cenres, conres] = [larr[2], larr[4], larr[5]]
        else:
            [contact, cenres, conres] = [larr[1], larr[5], larr[4]]
        cond = float(larr[3])
        if cond < dcut:
            continue
        elif re.search('\D$', contact): # sometimes residue number are not interger, like '31E', just ignore
            continue
        else:
            cons.append(contact)
            if outFile != None:
                out.write('\t'.join(map(str, [N, pid, center, contact, cond, cenres, conres]))+'\n')
            N += 1
    return cons

def makeFragment(pdb, seeds, outFile, flank = 2):
    resiDict = ConResDict(pdb)
    frag = None
    fragres = []
    for seed in seeds:
        if not seed in resiDict:
            print('Cannot find residue ', seed)
            return -1
        res = resiDict[seed]
        cid, resnum = res.getChid(), res.getResnum()

        if frag == None:
            frag = res
            fragres.append(seed)
        elif not seed in fragres:
            frag = frag | res
            fragres.append(seed)
        for i in range(1, flank+1):
            lkey = cid + ',' + str(resnum-i)
            hkey = cid + ',' + str(resnum+i)
            if lkey in resiDict and (not lkey in fragres):
                frag = frag | resiDict[lkey]
                fragres.append(lkey)
            if hkey in resiDict and (not hkey in fragres):
                frag = frag | resiDict[hkey]
                fragres.append(hkey)
    if outFile != None:
        writePDB(outFile, frag)
        pdb_id = os.path.basename(outFile)[:-4]
        p = PDBParser()
        structure = p.get_structure(pdb_id, outFile)
        io = MMTFIO()
        io.set_structure(structure)
        io.save(outFile[:-4] + '.mmtf')
        os.remove(outFile)
    fragres = sorted(fragres, key = lambda x : (x.split(',')[0], x.split(',')[1]))
    return fragres
