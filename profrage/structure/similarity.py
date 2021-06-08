"""
@author: Federico van Swaaij
"""

from Bio.PDB.QCPSuperimposer import QCPSuperimposer

from structure.representation import USR, FullStride
from utils.structure import get_backbone_atoms
from utils.tm_align import run_tm_align

class QCPSimilarity:

    def __init__(self, structure_1, structure_2):
        self.structure_1 = structure_1
        self.structure_2 = structure_2

    def _get_atoms(self, structure):
        pass