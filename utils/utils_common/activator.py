import os
app_dir = os.getcwd()

import sys
sys.path.append(app_dir + '/utils/utils_common')

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=FutureWarning)

import plots

import pandas as pd
import numpy as np
import random
import json
import h5py
import pickle

from matplotlib import pyplot as plt

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

from utils.utils_cpp import cpp_predictor
from utils.utils_synthesis import synthesis_feature_transformation, synthesis_predictor

class Activation:
    def __init__(self, **kwargs):
        '''
        Initializes an Activator object -

        Examples:
        
        1. For Synthesis - 
        activator = Activation(
            mode = 'synthesis',
            model_path = /path/to/minimal_model,
            fp_radius = 3,
            fp_bits = 128,
            seq_max = 50,
            pre_chain_smiles_path = /path/to/pre_chain_smiles,
            amino_acid_smiles_path = /path/to/amino_acid_smiles,
            transformation_functions_path = /path/to/transformation_functions,
            scaling_functions_path = /path/to/scaling_functions
        )
        
        activator.analyze('NEWSEQ') # 'NEWSE' is pre-chain from C-terminus, 'Q' is next amino acid
        
        2. For CPP -
        activator = Activation(
            mode = 'cpp',
            model_path = /path/to/model,
            fp_radius = 3,
            fp_bits = 128,
            seq_max = 108,
            smiles_path = /path/to/smiles,
            stats_path = /path/to/stats
        )

        activator.analyze('NEWSEQ')

        Args:
        mode: str, mode of usage - synthesis or cpp
        model_path: str, filepath of pre-trained model, to be used when using for prediction
        fp_radius: int, radius of topological exploration for 2d fingerprint
        fp_bits: int, size of bit-vector
        seq_max: int, maximum permissible length of sequence in predictor
        
        pre_chain_smiles_path: str, filepath of monomer structures for pre-chain in synthesis
        amino_acid_smiles_path: str, filepath of monomer structures for next amino acids in synthesis
        smiles_path: str, filepath of monomer structures for CPP
        
        transformation_functions_path: str, path for feature transformation functions in synthesis
        scaling_functions_path: str, path for label scaling functions in synthesis
        stats_path: str, path for mean,std stats in CPP

        '''

        self.__model_path = kwargs.get('model_path')
        self.__mode = kwargs.get('mode')
        self.__fp_radius = kwargs.get('fp_radius')
        self.__fp_bits = kwargs.get('fp_bits')
        self.__seq_max = kwargs.get('seq_max')
        
        if self.__mode == 'cpp':
            self.predictor = cpp_predictor.Predictor(
                model_path = self.__model_path, 
                smiles_path = kwargs.get('smiles_path'),
                stats_path = kwargs.get('stats_path'),
                fp_radius = self.__fp_radius,
                fp_bits = self.__fp_bits,
                seq_max = self.__seq_max
                )
        
        elif self.__mode == 'synthesis':
            self._feature_transformation = synthesis_feature_transformation.FeatureTransformation(
                pre_chain_smiles_path = kwargs.get('pre_chain_smiles_path'),
                amino_acid_smiles_path = kwargs.get('amino_acid_smiles_path'),
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
            
        
    def __target_category_loss_output_shape(self, input_shape):
        ''' Utility function for shape of output '''
        return input_shape-1

    def __target_category_loss(self, x, category_index, nb_classes):
        ''' Utility function for calculation of loss for regression '''
        return tf.multiply(x, K.one_hot([category_index], nb_classes))

    def __normalize(self, x):
        ''' Utility function to normalize a tensor by its L2 norm '''
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    def _calc_activation(self, sequence):
        '''
        Calculates the activation for a given sequence
        Protected function to allow access for optimization using gradient activation

        Args:
        sequence: str, 
                  pre-chain followed by next amino acid in synthesis
                  peptide sequence in cpp
        '''
        category_index = [0]
        nb_classes = 1

        target_layer = lambda x: self.__target_category_loss(x, category_index, nb_classes)

        self.predictor.model.add(Lambda(target_layer, output_shape = self.__target_category_loss_output_shape))

        loss = K.sum(self.predictor.model.layers[-1].output)

        conv_output =  self.predictor.model.layers[0].output
        conv_input =  self.predictor.model.layers[0].input

        grads = self.__normalize(K.gradients(loss, conv_input)[0])
        gradient_function = K.function([self.predictor.model.layers[0].input], [grads])

        if self.__mode == 'cpp':
            feature = self.predictor.nn_feature(sequence)
            length = len(sequence)
            grads_val = gradient_function([np.array([feature])])
        
        elif self.__mode == 'synthesis':
            feature, _ = self._feature_transformation.scale_transform(pd.DataFrame({
                'pre-chain': [sequence[:-1]], 'amino_acid': [sequence[-1]]
            }))
            length = len(sequence) - 1
            grads_val = gradient_function(feature)

        grads_val = np.maximum(grads_val[0][0], 0)
        
        if self.__mode == 'synthesis':
            diff = self.predictor.predict(feature)
            output = np.multiply(grads_val, feature)*diff
            return output[0][:-1][50-len(sequence)+1:,:]
        
        elif self.__mode == 'cpp':
            return np.multiply(grads_val, feature)[:length,:]

        return output
    
        ''' PUBLIC FUNCTION '''
    
    def analyze(self, sequence):
        '''
        Analyzes activation for given sequence and displays activated feature maps

        Args:
        sequence: str, 
                pre-chain followed by next amino acid in synthesis
                CPP sequence in cpp
        '''
        output = self._calc_activation(sequence)
        
        if self.__mode == 'synthesis':
            sequence = sequence[:-1]
        plots.positive_activation_feature_map(sequence, output)
        plots.positive_activation_avg_residues(sequence, output)
        plots.positive_activation_avg_fingerprint(sequence, output)
