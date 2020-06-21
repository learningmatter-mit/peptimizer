import os
app_dir = os.getcwd()

import sys
sys.path.append(app_dir + '/utils/utils_common')

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import fingerprint_2d as fingerprint
import plots

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import random
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import json
import h5py
import pickle

class FeatureTransformation:
    def __init__(
        self,
        pre_chain_smiles_path, 
        amino_acid_smiles_path, 
        mode='train',**kwargs):
        
        '''
        Initializes an instance of FeatureTransformation
        
        Examples:
        1. For training, validation - 
        feature_transformation = FeatureTransformation(
            fp_radius = 3,
            fp_bits = 128,
            seq_max = 50,
            pre_chain_smiles_path = /path/to/pre_chain_smiles,
            amino_acid_smiles_path = /path/to/amino_acid_smiles,
            transformation_functions_path = /path/to/transformation_functions
        )
        
        nnX, nnY = feature_transformation.scale_transform(df)
        
        2. For prediction - 
        feature_transformation = FeatureTransformation(
            fp_radius = 3,
            fp_bits = 128,
            seq_max = 50,
            pre_chain_smiles_path = /path/to/pre_chain_smiles,
            amino_acid_smiles_path = /path/to/amino_acid_smiles,
            transformation_functions_path = /path/to/transformation_functions,
            scaling_functions_path = /path/to/scaling_functions
        )
        
        nnX, _ = feature_transformation.scale_transform(df)
        
        Args:
        model_type: str, complete or minimal
        mode: str, either one of train, predict or validate
        
        fp_radius: int, radius of topological exploration for 2d fingerprint
        fp_bits: int, size of bit-vector
        seq_max: int, maximum permissible length of sequence in predictor
        pre_chain_smiles_path: str, filepath of monomer structures for pre-chain in synthesis
        amino_acid_smiles_path: str, filepath of monomer structures for next amino acids in synthesis
        
        feature_types: dict of lists (keys: continuous, categorical), datatype for each feature
        labels: list, labels to be trained over
        
        transformation_functions_path: str, path for feature transformation functions to load/save
        '''
        
        self.__model_type = kwargs.get('model_type', 'complete')
        self.__mode = mode
        
        self.__fp_radius = kwargs.get('fp_radius', 3)
        self.__fp_bits = kwargs.get('fp_bits', 128)
        self.__seq_max = kwargs.get('seq_max', 50)
        self.__pre_chain_smiles_path = pre_chain_smiles_path
        self.__amino_acid_smiles_path = amino_acid_smiles_path
        
        self._feature_types = {
            'categorical': [
                'coupling_agent',
                'machine',
                'coupling_strokes',
                'deprotection_strokes',
                'flow_rate'
            ],
            
            'continuous': [
                'temp_coupling',
                'temp_reactor_1'
            ]
        }
        
        self._labels = kwargs.get(
            'labels', 
            ['first_area', 'first_height', 'first_width', 'first_diff'])
        
        if 'categorical' in kwargs:
            self._feature_types.update(
                {'categorical': kwargs.get('categorical_features')})
            
        if 'continuous' in kwargs:
            self._feature_types.update(
                {'continuous': kwargs.get('continuous_features')})
                
        self.fp_pre_chain = fingerprint.Fingerprint_Generation(smiles_file = self.__pre_chain_smiles_path, 
                                               nbits = self.__fp_bits, radius = self.__fp_bits)
        
        self.fp_amino_acid = fingerprint.Fingerprint_Generation(smiles_file = self.__amino_acid_smiles_path, 
                                               nbits = self.__fp_bits, radius = self.__fp_bits)
        
        if self.__mode == 'train':
            self._scaling_functions = {}
            self._transformation_functions = {}
        
        elif self.__mode == 'predict' or self.__mode == 'validate':
            with open(kwargs.get('transformation_functions_path'), 'rb') as f:
                self._transformation_functions = pickle.load(f)
                
        else:
            raise NameError("Invalid mode.")

    def _one_hot(self, feature_list):
        ''' Generates one-hot encodings for categorical features '''
        mlb = MultiLabelBinarizer()
        feature_list = mlb.fit_transform([[x] for x in feature_list])
        return mlb, np.array(feature_list)
    
    def _normalize(self, feature_list):
        ''' Scales continous features '''
        scaler = StandardScaler()
        feature_list = scaler.fit_transform(feature_list)
        return scaler, np.array(feature_list)
    
    ''' PUBLIC FUNCTION '''
    
    def scale_transform(self, df, **kwargs):
        '''
        Transforms raw features and scales labels, returns nnX, nnY to be used in Predictor class
        
        Args:
        df: pd dataframe, data to be transformed/scaled
        
        (optional)
        scaling_functions_path: str, path for label scaling functions to save 
        
        '''
        nnX = {}
        nnY = {}
        
        self._df_fp = pd.DataFrame()
        
        df['padded_pre-chain'] = [
            ('0' * (self.__seq_max - len(x))) + x for x in df['pre-chain'].tolist()]
        
        nnX['fp_pre_chain'] = df['padded_pre-chain'].apply(self.fp_pre_chain.seq)
        nnX['fp_pre_chain'] = np.array([np.array(x) for x in nnX['fp_pre_chain']])
        
        nnX['fp_amino_acid'] = df['amino_acid'].apply(self.fp_amino_acid.seq)
        nnX['fp_amino_acid'] = np.array([
            x[0] for x in nnX['fp_amino_acid']])
        
        if self.__mode == 'train':
            for label in self._labels:
                self._scaling_functions[label], nnY[label] = self._normalize(
                    df[[label]])
            
        if self.__model_type == 'minimal':
            nnX = np.array([
                np.append(
                    nnX['fp_pre_chain'][i], 
                    np.array([nnX['fp_amino_acid'][i]]), axis=0) 
                for i in range(df.shape[0])])
            
            if self.__mode == 'train':
                nnY = nnY[self._labels[0]]
        
        elif self.__model_type == 'complete':

            if self.__mode == 'train':
                for feature in self._feature_types['categorical']:
                    self._transformation_functions[feature], nnX[feature] = self._one_hot(
                        df[feature])

                for feature in self._feature_types['continuous']:
                    self._transformation_functions[feature], nnX[feature] = self._normalize(
                        df[[feature]])
                
                if 'transformation_functions_path' in kwargs:
                    with open(kwargs.get('transformation_functions_path'), 'wb') as f:
                        pickle.dump(self._transformation_functions, f)
                
                if 'scaling_functions_path' in kwargs:
                    with open(kwargs.get('scaling_functions_path'), 'wb') as f:
                        pickle.dump(self._scaling_functions, f)

            elif self.__mode == 'predict' or self.__mode == 'validate':
                for feature in self._feature_types['categorical_features']:
                    nnX[feature] = self._transformation_functions[feature].transform(
                        [[x] for x in df[feature].tolist()])

                for feature in self._feature_types['continuous_features']: 
                    nnX[feature] = self._transformation_functions[feature].transform(df[[feature]])

        return nnX, nnY