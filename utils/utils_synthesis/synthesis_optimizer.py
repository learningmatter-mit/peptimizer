import pandas as pd
import numpy as np
import json

from utils.utils_synthesis import synthesis_feature_transformation, synthesis_predictor

class Optimization:
    def __init__(self, **kwargs):
        '''
        Initializes an instance for Optimizer

        Example:
        optimizer = Optimization(
            model_path = /path/to/minimal_model,
            fp_radius = 3,
            fp_bits = 128,
            seq_max = 50,
            pre_chain_smiles_path = /path/to/pre_chain_smiles,
            amino_acid_smiles_path = /path/to/amino_acid_smiles,
            transformation_functions_path = /path/to/transformation_functions,
            scaling_functions_path = /path/to/scaling_functions
        )
        
        optimizer.optimize('RQIKIWFQNRRMKWK')
        
        Args:
        model_path: str, filepath of pre-trained model, to be used for optimization
        fp_radius: int, radius of topological exploration for 2d fingerprint
        fp_bits: int, size of bit-vector
        seq_max: int, maximum permissible length of sequence in predictor

        pre_chain_smiles_path: str, filepath of monomer structures for pre-chain in synthesis
        amino_acid_smiles_path: str, filepath of monomer structures for next amino acids in synthesis

        transformation_functions_path: str, path for feature transformation functions in synthesis
        scaling_functions_path: str, path for label scaling functions in synthesis

        '''
        
        self.__model_path = kwargs.get('model_path')
        self.__fp_radius = kwargs.get('fp_radius')
        self.__fp_bits = kwargs.get('fp_bits')
        self.__seq_max = kwargs.get('seq_max')
        
        amino_acid_smiles_path = kwargs.get('amino_acid_smiles_path')

        self._feature_transformation = synthesis_feature_transformation.FeatureTransformation(
                    pre_chain_smiles_path = kwargs.get('pre_chain_smiles_path'),
                    amino_acid_smiles_path = amino_acid_smiles_path,
                    mode = 'predict', model_type = 'minimal',
                    fp_radius = self.__fp_radius,
                    fp_bits = self.__fp_bits,
                    seq_max = self.__seq_max,
                    transformation_functions_path = kwargs.get('transformation_functions_path')
                )

        self.predictor = synthesis_predictor.Predictor(
            model_path = self.__model_path,
            scaling_functions_path = kwargs.get('scaling_functions_path'),
            model_type = 'minimal'
        )

        with open(amino_acid_smiles_path, 'r') as f:
            self.__amino_acids = list(json.load(f).keys())
        
    def __mutations_all(self, original_seq):
        '''
        Generates all possible mutations and names for a given sequence
        
        Arg:
        original_seq: str, peptide sequence
        
        '''
        pre_chain = original_seq[::-1]
        len_original = len(original_seq)
        list_mutations = []
        name_mutation = []

        for position_temp in range(len(pre_chain)-1):
            seq_temp = list(pre_chain)
            for aa_temp in self.__amino_acids:
                seq_temp[position_temp] = aa_temp
                list_mutations += [''.join(seq_temp)]
                name_mutation += [pre_chain[position_temp] + str(len_original - position_temp) + aa_temp]

        list_mutations += [pre_chain]
        name_mutation += ['WT']

        df_temp = pd.DataFrame({'Mutant C-> N': list_mutations, 'Mutation': name_mutation})   
        df_temp.drop_duplicates(subset=['Mutation'], inplace=True)

        return df_temp
    
    def __predict_diff(self, sequence):
        ''' 
        Returns predicted difference using pre-trained model 
        
        Arg:
        sequence: str, peptide sequence from C-terminus side with pre-chain and next amino acid
        
        '''
        feature, _ = self._feature_transformation.scale_transform(pd.DataFrame({
                'pre-chain': [sequence[:-1]], 'amino_acid': [sequence[-1]]
            }))
        return self.predictor.model.predict(feature)[0][0]
    
    def optimize(self, sequence):
        ''' 
        Optimizes a given pre-chain 
        
        Arg:
        sequence: str, peptide sequence from C-terminus side with pre-chain and next amino acid
        '''
        
        df_mutants = self.__mutations_all(sequence[::-1])
        df_mutants['Difference'] = df_mutants['Mutant C-> N'].apply(self.__predict_diff)
        df_mutants.sort_values(['Difference'], ascending=True, inplace=True)
        df_mutants.reset_index(drop=True, inplace=True)
        
        return df_mutants