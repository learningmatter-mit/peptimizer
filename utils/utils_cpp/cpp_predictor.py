import os
app_dir = os.getcwd()

import sys
sys.path.append(app_dir + '/utils/utils_common')

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import fingerprint_2d as fingerprint
import plots
from calc_charge import net_charge

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import random
from sklearn.preprocessing import StandardScaler
import json
import h5py

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, LSTM, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class Predictor:
    def __init__(self, **kwargs):
        '''
        Initialize a Predictor object which can be used in both modes - train and predict
        
        Examples:
        1. Training a predictor - 
        
            predictor.Predictor(
                data_path = '/path/to/dataset.csv',
                smiles_path = '/path/to/smiles.json',
                stats_path = '/path/to/folder/',
                fp_radius = 3,
                fp_bits = 1024,
                seq_max = 108
            )
            
            predictor.train_model()
            
        2. Using a pre-trained model - 
        
            predictor.Predictor(
                model_path = '/path/to/model.hdf5',
                smiles_path = '/path/to/smiles.json',
                stats_path = '/path/to/folder/',
                fp_radius = 3,
                fp_bits = 1024,
                seq_max = 108
            )
            
            predictor.predict('NEWSEQ')

        Parameters
        ----------
        data_path:      str
                        filepath of current dataset, to be used when training
                        Custom dataset as a *.csv file may be used. The dataset should contain a list of sequences and activity
                        The column headers should be 'sequences' and 'intensity'
                    
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
        
        self.__smiles_path = kwargs.get('smiles_path')
        self.__fp_radius = kwargs.get('fp_radius', 3)
        self.__fp_bits = kwargs.get('fp_bits', 2048)
        self.__seq_max = kwargs.get('seq_max', 108)
        
        self.fp = fingerprint.Fingerprint_Generation(smiles_file = self.__smiles_path, 
                                               nbits = self.__fp_bits, radius = self.__fp_bits)
        
        if 'model_path' in kwargs:
            self.__model_path = kwargs.get('model_path')
            self.model = load_model(self.__model_path)
            
            self.__stats_path = kwargs.get('stats_path')
            with open(self.__stats_path) as json_file:
                self.dict_data = json.load(json_file)

        elif 'data_path' in kwargs:
            self.__data_path = kwargs.get('data_path')
            self.model_params = {
            'epochs': 2,
            'batch_size': 10,
            'val_split': 0.2,
            'save_checkpoint': False,
            'checkpoint_filepath': None,
            'filters': 256,
            'kernel_size': 2,
            'dropout': 0.1,
            'opt_lr': 0.0005,
            'opt_beta_1': 0.9,
            'opt_beta_2': 0.999,
            'opt_epsilon': None,
            'opt_decay': 0.0,
            'opt_amsgrad': False,
            'loss_function': 'mse'
        }

        else:
            raise NameError("Enter data_path for training, or for model_path for prediction.")
            
    def __load_data(self):
        
        '''
        Utility function to load the dataset

        Custom dataset as a *.csv file may be used. The dataset should contain a list of sequences and activity
        The column headers should be 'sequences' and 'intensity'

        Key Parameters
        -------
        self.X:  array of dimension (number of sequences, features_max, NBITS)
                 shuffled sequences used to train the predictor

        self.y:  array of dimension (number of sequences, 1)
                 shuffled (with same index as nnX) intensities used to train the predictor

        self.X_valid:    array of dimension (number of sequences, features_max, NBITS)
                         shuffled sequences used to validate the predictor while training

        self.y_valid:    array of dimension (number of sequences, 1)
                         shuffled (with same index as nnX) intensities used to validate the predictor while training

        self.dict_data:  dict
                         mean and standard deviation of intensity, arginine count, sequence length and charge
        '''

        print('Loading Data for Training of Predictor')
        
        df = pd.read_csv(self.__data_path)

        X_df = pd.DataFrame(columns=['sequence', 'feature'])
        Y_df = pd.DataFrame(columns=['intensity'])

        print ('Featurizing Data for Predictor')

        for i in range(0, df.shape[0]):
            seq = df['sequences'][i]
            X_df.at[i, 'sequence'] = seq
            X_df.at[i, 'feature'] = self.nn_feature(seq)

            Y_df.at[i, 'intensity'] = df['intensity'][i]

        self.X = np.ndarray(shape=(X_df.shape[0], self.__seq_max, self.__fp_bits), dtype=int)
        
        for i in range(0, X_df.shape[0]):
            self.X[i] = X_df.at[i, 'feature']
        
        X_df['charge'] = X_df['sequence'].apply(net_charge)
        X_df['R_count'] = X_df['sequence'].str.count('R')
        X_df['len_seq'] = X_df['sequence'].map(len)
        
        self.dict_data = {}
        self.dict_data['mean_intensity'] = Y_df['intensity'].mean()
        self.dict_data['std_intensity'] = Y_df['intensity'].std()
        
        self.dict_data['mean_charge'] = X_df['charge'].mean()
        self.dict_data['std_charge'] = X_df['charge'].std()
        
        self.dict_data['mean_R_count'] = X_df['R_count'].mean()
        self.dict_data['std_R_count'] = X_df['R_count'].std()
        
        self.dict_data['mean_len_seq'] = X_df['len_seq'].mean()
        self.dict_data['std_len_seq'] = X_df['len_seq'].std()

        scaler = StandardScaler()
        Y_df.fillna(0, inplace=True) #If there are missing values in the Spreadsheet, replacing them with 0.
        Y_df[['intensity']] = scaler.fit_transform(Y_df[['intensity']])

        self.y = np.asarray(Y_df['intensity'].values.tolist())

        indices = np.random.RandomState(seed=108).permutation(np.arange(self.X.shape[0]))

        self.X = self.X[indices]
        self.y = self.y[indices]
        
        self.X_valid = self.X[-int(len(indices)*self.model_params['val_split']):]
        self.y_valid = self.y[-int(len(indices)*self.model_params['val_split']):]
        self.y_valid = self.y_valid * self.dict_data['std_intensity'] + self.dict_data['mean_intensity']
                
    
    '''
    ----------------------------------------------------------------
                           PUBLIC FUNCTIONS
    ----------------------------------------------------------------
    '''
    def nn_feature(self, sequence):

        '''
        Utility function to generate feature map
        Needs to be public for access during activation analysis
        
        Parameters
        -------
        sequence:   str
                    peptide/polymer sequence

        Returns
        -------
        fp_seq: array of dimension (features_max, NBITS)
                feature map of sequence

        '''

        fp_seq = self.fp.seq(sequence)
        n_rows = self.__seq_max - len(sequence)
        shape_padding = (n_rows, self.__fp_bits)
        padding_array = np.zeros(shape_padding)
        fp_seq = np.concatenate((fp_seq, padding_array), axis = 0)

        return fp_seq
    
    
    def train_model(self, **kwargs):
        '''
        Public function to train the predictor.

        Parameters
        ----------
        (optional)
        
        model_path:     str
                        filepath to save model
                        
        stats_path:     str
                        filepath to save training dataset statistics
                        
        model_params:   dict
                        Parameters for the model - 
                            epochs: int
                            number of epochs to train the model

                            batch_size: int
                            batch_size for the model

                            val_split: float
                            train with (1-validation_split) of the dataset, and validate with the rest
                            
                            save_checkpoint:  bool
                            to save or not to save intermediate models

                            checkpoint_filepath: str
                            location to save intermediate models, if save_checkpoint == True
                            
                            filters: int
                            number of filters of Conv1D model
                            
                            kernel_size: int
                            kernel size of Conv1D model
                            
                            dropout: float
                            dropout to regularize training, 0.0 < value < 1.0
                            
                            opt_lr: float
                            learning rate of Adam optimizer
                            
                            opt_beta_1: float
                            parameter for Adam optimizer
                            
                            opt_beta_2: float
                            parameter for Adam optimizer
                            
                            opt_epsilon: float
                            parameter for Adam optimizer
                            
                            opt_decay: float
                            parameter for Adam optimizer
                            
                            opt_amsgrad: boolean
                            parameter for Adam optimizer
                            
                            loss_function: str
                            loss function for predictor

        '''
            
        if 'model_params' in kwargs:
            self.__model_params = (kwargs.get('model_params'))
            self.__model_params.update(kwargs.get('model_params'))
            
        self.__load_data()
        
        if 'stats_path' in kwargs:
            self.__stats_path = (kwargs.get('stats_path'))
            json.dump(self.dict_data, open(self.__stats_path, 'w'))
        
        print ('Creating Model for Predictor')
    
        model = Sequential()

        model.add(Conv1D(self.model_params['filters'], self.model_params['kernel_size'], 
                         input_shape=(self.X.shape[1], self.X.shape[2])))
        model.add(Dropout(self.model_params['dropout']))
        model.add(Conv1D(self.model_params['filters'], self.model_params['kernel_size']))
        model.add(Dropout(self.model_params['dropout']))
        model.add(Activation('relu'))
        model.add(Conv1D(self.model_params['filters'], self.model_params['kernel_size']))
        model.add(Dropout(self.model_params['dropout']))
        model.add(Flatten())
        model.add(Dense(self.model_params['filters']))
        model.add(Activation('softplus'))
        model.add(Dropout(self.model_params['dropout']))
        model.add(Dense(1, activation='linear'))

        optimizer = Adam(
            lr=self.model_params['opt_lr'], 
            beta_1=self.model_params['opt_beta_1'], 
            beta_2=self.model_params['opt_beta_2'], 
            epsilon=self.model_params['opt_epsilon'], 
            decay=self.model_params['opt_decay'], 
            amsgrad=self.model_params['opt_amsgrad']
        )

        model.compile(optimizer=optimizer,
                      loss=self.model_params['loss_function'])

        callbacks_list = []

        if self.model_params['save_checkpoint'] == True:
            checkpoint = ModelCheckpoint(self.model_params['checkpoint_filepath'] + 
                                         "predictor-epoch{epoch:02d}-loss{loss:.4f}-val_loss{val_loss:.4f}.hdf5", 
                                         monitor='val_loss', 
                                         save_best_only=True, 
                                         mode='min')
            callbacks_list = [checkpoint]
        
        model.fit(self.X, self.y, 
                  epochs=self.model_params['epochs'], 
                  batch_size=self.model_params['batch_size'], 
                  validation_split=self.model_params['val_split'], 
                  callbacks=callbacks_list
                 )
        
        self.model = model
        
        plots.model_performance(experimental = self.y_valid,
                                predicted = self.model.predict(self.X_valid)*
                                self.dict_data['std_intensity']+self.dict_data['mean_intensity']
                               )
        
        if 'model_path' in kwargs:
            self.__model_path = kwargs.get('model_path')
            model.save(filepath = self.__model_path)
    
    
    def predict(self, sequence):
        '''
        Public function to predict the activity
        
        Parameters
        -------
        sequence:   str
                    peptide/polymer sequence

        Returns
        -------
        y:  float
            predicted intensity (normalized)

        '''
        
        return (self.model.predict(np.asarray([self.nn_feature(sequence)]))[0][0]*
                self.dict_data['std_intensity']+self.dict_data['mean_intensity'])