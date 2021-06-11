import numpy as np

from Bio.PDB.QCPSuperimposer import QCPSuperimposer

from utils.structure import get_backbone_atoms

class QCPSimilarity:
    """
    A similarity Measure based on the QCPSuperimposer.

    The structures are compared based on the superimposition of their backbone atoms.

    Attributes
    ----------
    structure_1 : Bio.PDB.Structure
        The first structure.
    structure_2 : Bio.PDB.Structure
        The second structure.
    """

    def __init__(self, structure_1, structure_2):
        """
        Initialize the class.

        Parameters
        ----------
        structure_1 : Bio.PDB.Structure
            The first structure.
        structure_2 : Bio.PDB.Structure
            The second structure.
        """
        self.structure_1 = structure_1
        self.structure_2 = structure_2

    def _get_shifted_atoms(self, atoms_long, atoms_short):
        shifted_atoms, i = [], 0
        while i + len(atoms_short) <= len(atoms_long):
            shifted_atoms.append(atoms_long[i:i+len(atoms_short)])
            i += 1
        return shifted_atoms

    def _superimpose(self, atoms_fixed, atoms_moving):
        # Prepare the coordinates
        fixed, moving = np.zeros(shape=(len(atoms_fixed),3)), np.zeros(shape=(len(atoms_fixed),3))
        for i in range(len(atoms_fixed)):
            fixed[i,:], moving[i,:] = atoms_fixed[i].get_coord(), atoms_moving[i].get_coord()
        # Superimpose
        qcpsi = QCPSuperimposer()
        qcpsi.set(fixed, moving)
        qcpsi.run()
        return qcpsi.get_rms()

    def compare(self):
        """
        Perform the comparison.

        Returns
        -------
        float
            The RMSD super-imposition score.
        """
        atoms_1 = get_backbone_atoms(self.structure_1)
        atoms_2 = get_backbone_atoms(self.structure_2)
        if len(atoms_1) == len(atoms_2):
            return self._superimpose(atoms_1, atoms_2)
        elif len(atoms_1) > len(atoms_2):
            shifted_atoms = self._get_shifted_atoms(atoms_1, atoms_2)
            rmsds = []
            for sas in shifted_atoms:
                rmsds.append(self._superimpose(sas, atoms_2))
            rmsds = np.array(rmsds)
            return np.min(rmsds)
        else:
            shifted_atoms = self._get_shifted_atoms(atoms_2, atoms_1)
            rmsds = []
            for sas in shifted_atoms:
                rmsds.append(self._superimpose(atoms_1, sas))
            rmsds = np.array(rmsds)
            return np.min(rmsds)