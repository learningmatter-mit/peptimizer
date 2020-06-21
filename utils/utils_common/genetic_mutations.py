import random
import pandas as pd
import json

class Genetic_Mutations:
    def __init__(self, **kwargs):
        
        '''
        Initialize a Genetic_Mutations class to perform random single and multi residue genetic mutations
        
        Example:
        genetic_mutation = genetic_mutations.Genetic_Mutations(
            data_path = self.__data_path, 
            smiles_path = self.__smiles_path,
            seq_max = self.__seq_max
        )

        genetic_mutation.mutate('ABCDEFGHI')
        
        
        Parameters
        ----------
        data_path:      str
                        filepath of current dataset, to be used when training
                        Custom dataset as a *.csv file may be used. The dataset should contain a list of sequences and activity
                        The column headers should be 'sequences' and 'intensity'
                    
        smiles_path:    str
                        filepath of monomer structures
                        *.json file may be used. 
                        Keys are monomers in the same notation as the dataset, preferably single letter.
                        Values are SMILES strings.

        seq_max:    int
                    maximum feature length
        )
        
        '''
        
        self.__data_path = kwargs.get('data_path')
        self.__smiles_path = kwargs.get('smiles_path')
        self.__seq_max = kwargs.get('seq_max')
        
        self.seq_list = pd.read_csv(self.__data_path)['sequences'].tolist()
        
        with open(self.__smiles_path) as f:
            lookupsmiles = json.load(f)
        
        self.residues = [i for i in lookupsmiles.keys() if i not in '@#1234567890']

    def __deletion(self, str_sequence):
        
        '''
        Utility function to mutate a sequence by deleting a random residue

        Parameters
        ----------
        sequence:   str
                    peptide sequence

        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        str_sequence = str_sequence.split(" ")[0]
        toremove = random.randint(0, len(str_sequence) - 1)
        new_str = str_sequence[:toremove] + str_sequence[toremove+1:] + " "
        
        return new_str

    def __insertion(self, str_sequence):
        
        '''
        Utility function to mutate a sequence by inserting a random residue

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        str_sequence = str_sequence.split(" ")[0]
        toinsert = random.randint(0, len(str_sequence))
        new_str = str_sequence[:toinsert] + random.choice(self.residues) + str_sequence[toinsert:]
        return new_str[:self.__seq_max]

    def __swap(self, str_sequence):
        
        '''
        Utility function to mutate a sequence by swapping a random residue

        Parameters
        ----------
        sequence:   str
                    peptide sequence

        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        existingaas = [i for i in set(str_sequence) if i != " "]
        aatoreplace = random.choice(existingaas)
        aaindices = [index for index, value in enumerate(str_sequence) if value == aatoreplace]
        indextoreplace = random.choice(aaindices)
        new_str = str_sequence[:indextoreplace] + random.choice(self.residues) + str_sequence[indextoreplace+1:]
        return new_str

    def __hybrid(self, str_sequence):
        
        '''
        Utility function to mutate a sequence by replacing a random motif  with 
        another random motif from a random sequence 

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        constant = 5 # 1/constant(th) of the sequence to be considered for hybridization
        str_sequence = str_sequence.split(" ")[0]
        tohybrid = random.randint(0, len(str_sequence)-1)
        hybridlen = random.randint(0, int((len(str_sequence) - tohybrid)/constant))+1
        hybrid_from = random.choice(self.seq_list)
        leastlen = hybridlen

        while leastlen != hybridlen:
            hybrid_from = random.choice(self.seq_list)
            leastlen = len(hybrid_from)

        index_hybrid_max = leastlen-hybridlen
        if index_hybrid_max > 0:
            index_hybrid_max = index_hybrid_max - 1
        index_hybrid = random.randint(0, index_hybrid_max)
        hybrid_from = hybrid_from[index_hybrid:index_hybrid+hybridlen]
        new_str = str_sequence[:tohybrid] + hybrid_from + str_sequence[tohybrid+hybridlen:] + " "

        return new_str        
    
    
    '''
    ----------------------------------------------------------------
                           PUBLIC FUNCTIONS
    ----------------------------------------------------------------
    '''

    def mutate(self, sequence):
        
        '''
        Public function to mutate a sequence

        Parameters
        ----------
        sequence:   str
                    peptide sequence

        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        mutation_list = [self.__insertion, self.__deletion, self.__swap, self.__hybrid]
        mutation = random.choice(mutation_list)
        
        return mutation(sequence)