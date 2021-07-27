import numpy as np

from Bio import pairwise2
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Polypeptide import PPBuilder

AA_DICT = {'A': 'ALA',
           'R': 'ARG',
           'N': 'ASN',
           'D': 'ASP',
           'C': 'CYS',
           'Q': 'GLN',
           'G': 'GLY',
           'E': 'GLU',
           'H': 'HIS',
           'I': 'ILE',
           'L': 'LEU',
           'K': 'LYS',
           'M': 'MET',
           'F': 'PHE',
           'P': 'PRO',
           'S': 'SER',
           'T': 'THR',
           'W': 'TRP',
           'Y': 'TYR',
           'V': 'VAL'}

AA_TO_INT = {'ALA': 1,
             'ARG': 2,
             'ASN': 3,
             'ASP': 4,
             'CYS': 5,
             'GLN': 6,
             'GLY': 7,
             'GLU': 8,
             'HIS': 9,
             'ILE': 10,
             'LEU': 11,
             'LYS': 12,
             'MET': 13,
             'PHE': 14,
             'PRO': 15,
             'SER': 16,
             'THR': 17,
             'TRP': 18,
             'TYR': 19,
             'VAL': 20}

INT_TO_AA = {1: 'ALA',
             2: 'ARG',
             3: 'ASN',
             4: 'ASP',
             5: 'CYS',
             6: 'GLN',
             7: 'GLY',
             8: 'GLU',
             9: 'HIS',
             10: 'ILE',
             11: 'LEU',
             12: 'LYS',
             13: 'MET',
             14: 'PHE',
             15: 'PRO',
             16: 'SER',
             17: 'THR',
             18: 'TRP',
             19: 'TYR',
             20: 'VAL'}

def is_complete(structure):
    """
    Check whether the given structure is complete. A structure is considered complete if all residues contain the carbon-alpha atom.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to check.

    Returns
    -------
    bool
        Whether the structure is complete.
    """
    for residue in structure.get_residues():
        if 'CA' not in residue or residue.get_resname() not in AA_TO_INT:
            return False
    return True

def structure_length(structure):
    """
    Return the length of the specified structure in terms of its number of residues.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to compute the length of.

    Returns
    -------
    count : int
        The length of the structure measured in number of residues.
    """
    count = 0
    for residue in structure.get_residues():
        count += 1
    return count

def get_backbone_atoms(structure):
    """
    Return the backbone atoms of the specified structure.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure.

    Returns
    -------
    list of Bio.PDB.Atom
        The backbone atoms.
    """
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    atoms.append(residue['CA'])
                if 'C' in residue:
                    atoms.append(residue['C'])
                if 'N' in residue:
                    atoms.append(residue['N'])
    return atoms

def get_bb_atoms_coords(structure):
    """
    Return the coordinates of the backbone atoms of the given structure.
    
    The output has shape Nx3, where N is the total number of atoms.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to get the atoms coordinates.

    Returns
    -------
    numpy.ndarray
        The atoms coordinates.
    """
    atoms_coords = []
    count = 0
    for atom in structure.get_atoms():
        if atom.get_name() == 'CA' or atom.get_name() == 'C' or atom.get_name() == 'N':
            coords = atom.get_coord()
            atoms_coords.append([coords[0], coords[1], coords[2]])
            count += 1
    return np.array(atoms_coords).reshape((count,3))

def get_atoms_coords(structure):
    """
    Return the coordinates of the atoms of the given structure.
    
    The output has shape Nx3, where N is the total number of atoms.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure of which to get the atoms coordinates.

    Returns
    -------
    numpy.ndarray
        The atoms coordinates.
    """
    atoms_coords = []
    count = 0
    for atom in structure.get_atoms():
        coords = atom.get_coord()
        atoms_coords.append([coords[0], coords[1], coords[2]])
        count += 1
    return np.array(atoms_coords).reshape((count,3))

def get_residue_center(residue):
    """
    Compute the coordinates of the center of the given residue.

    Parameters
    ----------
    residue : Bio.PDB.Residue
        The residue.

    Returns
    -------
    numpy.ndarray
        The ccoordinates of the center of the residue.
    """
    coords = []
    for atom in residue:
        coords.append(atom.get_coord())
    coords = np.array(coords)
    return np.mean(coords, axis=0)

def lengths_within(structure_1, structure_2, ptc_thr):
    """
    Return whether the specified structures have comparable length.

    Parameters
    ----------
    structure_1 : Bio.PDB.Structure
        The first structure.
    structure_2 : Bio.PDB.Structure
    The first structure.

    Returns
    -------
    bool
        Whether the structures have comparable length.
    """
    small = structure_length(structure_1)
    large = structure_length(structure_2)
    if small > large:
        temp = large
        large = small
        small = temp
    return (small/large) >= ptc_thr

def get_sequences(structure):
    """
    Compute the sequences of all the chains of the specified structure.
    
    The sequences are already filtered as not to include duplicates.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure from which to compute the sequences.

    Returns
    -------
    sequences : list of Bio.Seq.Seq
        The list of sequences, each sequence matching with a chain of the class.
    """
    ppb = PPBuilder()
    sequences = []
    for pp in ppb.build_peptides(structure):
        seq = pp.get_sequence()
        if seq not in sequences:
            sequences.append(pp.get_sequence())
    return sequences

def align_sequences(seq_1, seq_2):
    """
    Align the given sequences.
    
    In this setting, there are no gap penalties.

    Parameters
    ----------
    seq_1 : Bio.Seq.Seq
        The first sequence.
    seq_2 : Bio.Seq.Seq
        The second sequence.

    Returns
    -------
    alignments : list of (str, str, float, int, int)
        The alignements.
    """
    alignments = pairwise2.align.globalxx(seq_1, seq_2)
    return alignments

def get_model_residues(structure, m_id=0):
    """
    Return the residues of the specified sturcture belonging to the specified model.
    
    Using this method may be beneficial when dealing with proteins coming from NMR.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The original structure.
    m_id : int, optional
        The model ID. The default is 0.

    Returns
    -------
    m_residues : list of Bio.PDB.Residues
        The residues belonging to the specified model.
    """
    m_residues = []
    for model in structure:
        if model.get_id() == m_id:
            for chain in model:
                for residue in chain:
                    m_residues.append(residue)
    return m_residues

def contains_dna_rna(structure):
    """
    Check whether the speficied structure contains DNA and/or RNA.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to check.

    Returns
    -------
    bool
        Whether the stucture contains DNA and/or RNA.
    """
    for model in structure:
        for chain in model:
            for residue in chain:
                rn = residue.get_resname().strip() # necessary since name are all 3 characters long
                if rn == 'DA' or rn == 'DC' or rn == 'DG' or rn == 'DT' or rn == 'DI' or rn == 'A' or rn == 'C' or rn == 'G' or rn == 'U' or rn == 'I':
                    return True
    return False

def break_chains(structure, m_id=0):
    """
    Break the specified structure into its chains.
    
    In case the specified structure has multiple models, only the first one is returned.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure to be broken into its chaines.
    m_id : int, optional
        The model ID. The default is 0.

    Returns
    -------
    structures_dict : dict of str -> list of Bio.PDB.Residue
        The dictionary mapping the chain ID to the residues belonging to that chain.
    """
    structures_dict = {}
    for model in structure:
        if model.get_id() == m_id:
            for chain in model:
                ch_id = chain.get_id().upper()
                if ch_id not in structures_dict:
                    structures_dict[ch_id] = []
                for residue in chain:
                    structures_dict[ch_id].append(residue)
    return structures_dict

def filter_chains(chains, pct_thr=0.9):
    """
    Filter the chains belonging to the same structure to remove similar and/or identical chains.

    Parameters
    ----------
    chains : list of Bio.PDB.Structure
        The list of structures, each representing a chain of the original structure.
    pct_thr : float in [0,1], optional
        The score percentage threshold above which two chains are considered to be similar. The score is
        computed as the best alignement score divided by the length of the smallest chain.
        The default is 0.9.

    Returns
    -------
    uniques : list of Bio.PDB.Structures
        The list of unique chains.
    """
    chains_dict = {}
    seq_dict = {}
    uniques = []
    assigned = {}
    # Get the sequence for each chain
    for chain in chains:
        p_id = chain.get_id()
        chains_dict[p_id] = chain
        sequences = get_sequences(chain)
        if len(sequences) == 0:
            continue
        seq_dict[p_id] = sequences[0] # since there is only one chain, there is only one sequence
    # Iterate over the dictionaries to cluster the chains based on sequence alignement
    for i in seq_dict:
        if i not in assigned:
            assigned[i] = True
            uniques.append(chains_dict[i])
            for j in seq_dict:
                if j != i and j not in assigned:
                    # Get alignments
                    seq_1, seq_2 = seq_dict[i], seq_dict[j]
                    alignements = align_sequences(seq_1, seq_2)
                    best_alignment = sorted(alignements, key=lambda x: x[2], reverse=True)[0]
                    score = best_alignment[2]/(min(len(seq_1),len(seq_2)))
                    if score >= pct_thr:
                        assigned[j] = True
    return uniques

def build_structure(s_id, residues, header):
    """
    Generate a structure based on a collection of residues.

    Parameters
    ----------
    s_id : str
        The ID of the structure to build.
    residues : list of Bio.PDB.Residue
        The list of residue from which to build the structure.
    header : dict of str -> Any
        The header of the structure. See the documentation of Bio.PDB.Structure for more information.

    Returns
    -------
    structure : Bio.PDB.Structure
        The generate structure.
    """
    structure = Structure(s_id)
    # Sort residues to ensure correct sequence order
    residues = sorted(residues, key=lambda x: x.get_id()[1])
    for residue in residues:
        r_full_id = residue.get_full_id()
        # Check if residue model exists, if not add it
        if not structure.has_id(r_full_id[1]):
            structure.add(Model(r_full_id[1]))
        # Get correct model for the residue
        for model in structure:
            been_added = False
            if model.get_id() == r_full_id[1]:
                # Check if model has the chain, if not add it
                if not model.has_id(r_full_id[2]):
                    model.add(Chain(r_full_id[2]))
                for chain in model:
                    if chain.get_id() == r_full_id[2]:
                        r_id = r_full_id[3]
                        if not chain.has_id(r_id):
                            r = Residue(r_id, residue.get_resname(), residue.get_segid())
                            for atom in residue:
                                r.add(atom)
                            chain.add(r)
                            been_added = True
                            break
                # If residue has been added then we can exit the loop
                if been_added:
                    break
    # Add stucture header
    structure.header = header
    # Return created structure
    return structure
