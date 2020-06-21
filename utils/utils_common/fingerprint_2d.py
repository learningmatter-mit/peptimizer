from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import numpy as np
import json

class Fingerprint_Generation:
    def __init__(self, smiles_file, radius, nbits, newsmiles=None):
        with open(smiles_file) as json_file:
            smiles = json.load(json_file)
        
        self.lookupfps = {}
        
        if newsmiles != None:
            smiles.update(newsmiles)
        
        for key, value in smiles.items():
            mol = Chem.MolFromSmiles(value)
            fp = np.array(Chem.GetMorganFingerprintAsBitVect(mol,radius,nbits))
            self.lookupfps[key] = fp
        self.lookupfps[' '] = np.zeros(self.lookupfps['A'].shape, dtype=int)
        self.lookupfps['0'] = np.zeros(self.lookupfps['A'].shape, dtype=int)
    
    def seq(self, seq):
        fp = np.asarray([self.lookupfps[seq[i]] for i in range(len(seq))])
        return fp