import os
app_dir = os.getcwd()

import sys
sys.path.append(app_dir + '/utils/utils_common')

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=FutureWarning)

import activator
import plots
from utils.utils_cpp import cpp_predictor

class Activation:
    def __init__(self, **kwargs):
        '''
        Initialize an Activator object -
        
        Examples:
        activator = cpp_activator.Activation(
            model_path = '/path/to/model.hdf5',
            data_path = '/path/to/dataset.csv',
            smiles_path = '/path/to/smiles.json',
            stats_path = '/path/to/stats.json',
            fp_radius = 3,
            fp_bits = 1024,
            seq_max = 108
        )
        
        activator.analyze('ABCDEFGHI')

        Parameters
        ----------
        
        model_path:     str
                        filepath of pre-trained model, to be used when using for prediction
                    
        smiles_path:    str
                        filepath of monomer structures
                        *.json file may be used. 
                        Keys are monomers in the same notation as the dataset, preferably single letter.
                        Values are SMILES strings.
                        
        stats_path:     str
                        filepath of training dataset statistics
                    
        fp_radius:      int
                        radius of topological exploration for 2d fingerprint
                    
        fp_bits:        int
                        size of bit-vector
                    
        seq_max:        int
                        maximum permissible length of sequence in predictor
         
        '''

        self.__model_path = kwargs.get('model_path')
        self.__smiles_path = kwargs.get('smiles_path')
        self.__stats_path = kwargs.get('stats_path')
        self.__fp_radius = kwargs.get('fp_radius')
        self.__fp_bits = kwargs.get('fp_bits')
        self.__seq_max = kwargs.get('seq_max')
        
        self.activator = activator.Activation(
            mode = 'cpp',
            model_path = self.__model_path, 
            smiles_path = self.__smiles_path,
            stats_path = self.__stats_path,
            fp_radius = self.__fp_radius,
            fp_bits = self.__fp_bits,
            seq_max = self.__seq_max
        )


    '''
    ----------------------------------------------------------------
                           PUBLIC FUNCTIONS
    ----------------------------------------------------------------
    '''

    def analyze(self, sequence):        
        
        output = self.activator.calc_activation(sequence)
        plots.positive_activation_feature_map(sequence, output)
        plots.positive_activation_avg_residues(sequence, output)
        plots.positive_activation_avg_fingerprint(sequence, output)