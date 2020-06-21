import os
app_dir = os.getcwd()

import sys
sys.path.append(app_dir + '/utils/utils_common')

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

import plots

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from sklearn.preprocessing import MultiLabelBinarizer
import json
import h5py
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv1D, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

class Predictor:
    def __init__(self, model_type, **kwargs):
        '''
        Initializes a Predictor object

        Examples:

        1. For training -
        predictor = Predictor(model_type = 'complete') 
        predictor.train(nnX, nnY)

        2. For prediction - 
        predictor = Predictor(
            model_path = /path/to/pre_trained_model,
            model_type = 'complete',
            scaling_functions_path = /path/to/scaling_functions)

        predictor.predict(nnX)

        Args:
        model_type: str
                    'complete' uses all features and labels
                    'minimal' uses only pre-chain and next amino acid, and 1 label

        model_path: str, path for pre trained model for prediction
        scaling_functions_path: str, path for label scaling functions for prediction

        '''
        self.__model_type = model_type
        
        self.__HYPER_PARAMS_PATH = './dataset/data_synthesis/hyperparameters.json'
        SCALING_FUNCTIONS_PATH = './dataset/data_synthesis/scaling_function.pkl'
        
        with open(kwargs.get('scaling_functions_path', SCALING_FUNCTIONS_PATH), 'rb') as f:
                self._scaling_functions = pickle.load(f)
        
        if 'model_path' in kwargs:
            self.__model_path = kwargs.get('model_path')
            self.model = load_model(self.__model_path)

    def __create_model(self):
        ''' Utility function to create and train model '''
        
        if self.__model_type == 'complete':
             
            fp_pre_chain = keras.Input(
                shape=self._nnX['fp_pre_chain'][0].shape, 
                name='fp_pre_chain')
            
            fp_amino_acid = keras.Input(
                shape=self._nnX['fp_amino_acid'][0].shape,
                name='fp_amino_acid')

            coupling_agent = keras.Input(
                shape=self._nnX['coupling_agent'][0].shape, 
                name='coupling_agent')
            
            coupling_strokes = keras.Input(
                shape=self._nnX['coupling_strokes'][0].shape,
                name='coupling_strokes')
            
            temp_coupling = keras.Input(
                shape=self._nnX['temp_coupling'][0].shape, 
                name='temp_coupling')
            
            deprotection_strokes = keras.Input(
                shape=self._nnX['deprotection_strokes'][0].shape, 
                name='deprotection_strokes')

            flow_rate = keras.Input(
                shape=self._nnX['flow_rate'][0].shape, 
                name='flow_rate')
            
            machine = keras.Input(
                shape=self._nnX['machine'][0].shape, 
                name='machine')
            
            temp_reactor_1 = keras.Input(
                shape=self._nnX['temp_reactor_1'][0].shape, 
                name='temp_reactor_1')

            x_pre_chain = Conv1D(2**self.model_params['pre_chain_conv1_filter'], 
                                 2**self.model_params['pre_chain_conv1_kernel'])(fp_pre_chain)
            x_pre_chain = Dense(2**self.model_params['pre_chain_dense1'])(x_pre_chain)
            x_pre_chain = Dropout(self.model_params['pre_chain_dropout1'])(x_pre_chain)
            x_pre_chain = Conv1D(2**self.model_params['pre_chain_conv2_filter'], 
                                 2**self.model_params['pre_chain_conv2_kernel'])(x_pre_chain)
            x_pre_chain = Dropout(self.model_params['pre_chain_dropout2'])(x_pre_chain)
            x_pre_chain = Activation(self.model_params['pre_chain_activation1'])(x_pre_chain)
            x_pre_chain = Flatten()(x_pre_chain)
            x_pre_chain = Dense(2**self.model_params['pre_chain_amino_acid_dense_final'], 
                                activation=self.model_params['pre_chain_activation2'])(x_pre_chain)

            x_amino_acid = Dense(2**self.model_params['amino_acid_dense1'])(fp_amino_acid)
            x_amino_acid = Dense(2**self.model_params['amino_acid_dense2'], 
                                 activation=self.model_params['amino_acid_activation1'])(x_amino_acid)
            x_amino_acid = Dropout(self.model_params['amino_acid_dropout1'])(x_amino_acid)
            x_amino_acid = Dense(2**self.model_params['pre_chain_amino_acid_dense_final'], 
                                 activation=self.model_params['amino_acid_activation2'])(x_amino_acid)

            x_chemistry = concatenate([x_pre_chain, x_amino_acid])
            x_chemistry = Dense(2**self.model_params['chemistry_dense1'])(x_chemistry)
            x_chemistry = Dense(2**self.model_params['chemistry_dense2'])(x_chemistry)

            x_coupling_agent = Activation('sigmoid')(coupling_agent)
            x_coupling_strokes = Activation('sigmoid')(coupling_strokes)
            x_temp_coupling = Activation('sigmoid')(temp_coupling)
            x_deprotection_strokes = Activation('sigmoid')(deprotection_strokes)
            x_deprotection_strokes = Dense(4, activation='relu')(x_deprotection_strokes)

            x_coupling = concatenate(
                [x_coupling_agent, x_coupling_strokes, x_temp_coupling, x_deprotection_strokes])
            x_coupling = Dense(self.model_params['coupling_dense1'])(x_coupling)
            x_coupling = Dense(self.model_params['coupling_dense2'])(x_coupling)

            x_flow_rate = Activation('sigmoid')(flow_rate)
            x_machine = Activation('sigmoid')(machine)
            x_machine = Dense(3, activation='relu')(x_machine)
            x_temp_reactor_1 = Activation('sigmoid')(temp_reactor_1)

            x_machine_variables = concatenate([x_flow_rate, x_machine, x_temp_reactor_1])
            x_machine_variables = Dense(self.model_params['machine_dense1'])(x_machine_variables)
            x_machine_variables = Dense(self.model_params['machine_dense2'])(x_machine_variables)

            x = concatenate([x_chemistry, x_coupling, x_machine_variables])
            x = Dense(2**self.model_params['concat_dense1'])(x)
            x = Dense(2**self.model_params['concat_dense2'], 
                      activation=self.model_params['concat_activation2'])(x)
            x = Dropout(self.model_params['concat_dropout1'])(x)
            x = Dense(2**self.model_params['concat_dense3'], 
                      activation=self.model_params['concat_activation3'])(x)

            first_area = Dense(1,  activation='linear', name='first_area')(x)
            first_height = Dense(1,  activation='linear', name='first_height')(x)
            first_width = Dense(1,  activation='linear', name='first_width')(x)

            first_diff = Dense(1,  activation='linear', name='first_diff')(x)

            model = Model(
                inputs=[fp_pre_chain, fp_amino_acid, 
                        coupling_agent, coupling_strokes, temp_coupling, deprotection_strokes, 
                        flow_rate, machine, temp_reactor_1], 
                outputs=[first_area, first_height, first_width, first_diff]
            )

        elif self.__model_type == 'minimal':
            model = Sequential()
            model.add(Conv1D(
                2**self.model_params['pre_chain_conv1_filter'], 
                2**self.model_params['pre_chain_conv1_kernel'], 
                input_shape=(self._nnX[0].shape[0], self._nnX[0].shape[1])))
            model.add(Dense(2**self.model_params['pre_chain_dense1']))
            model.add(Dropout(self.model_params['pre_chain_dropout1']))
            model.add(Conv1D(
                2**self.model_params['pre_chain_conv2_filter'], 
                2**self.model_params['pre_chain_conv2_kernel']))
            model.add(Dropout(self.model_params['pre_chain_dropout2']))
#             model.add(Activation(self.model_params['pre_chain_activation1']))
            model.add(Flatten())
            model.add(Dense(
                2**self.model_params['pre_chain_amino_acid_dense_final'],
                activation=self.model_params['pre_chain_activation2']))
            model.add(Dense(
                2**self.model_params['concat_dense1']))
            model.add(Dense(
                2**self.model_params['concat_dense2']))
            model.add(Dropout(
                self.model_params['concat_dropout1']))
            model.add(Dense(
                2**self.model_params['concat_dense3']))
            model.add(Dense(
                1, activation='linear'))
        
        model.compile(
            optimizer = RMSprop(lr=self.model_params['opt_lr']),
            loss=mse)

        callbacks_list = []

        if self.model_params['save_checkpoint'] == True:
            checkpoint = ModelCheckpoint(
                self.model_params['checkpoint_filepath'] + 
                "predictor-epoch{epoch:02d}-loss{loss:.4f}-val_loss{val_loss:.4f}.hdf5", 
                monitor='val_loss', 
                save_best_only=True, 
                mode='min')
            callbacks_list = [checkpoint]
        
        model.fit(self._nnX, self._nnY, 
                  epochs=self.model_params['epochs'], 
                  batch_size=self.model_params['batch_size'], 
                  validation_split=self.model_params['val_split'], 
                  callbacks=callbacks_list, verbose=False
                 )
        
        self.model = model

    ''' PUBLIC FUNCTIONS '''

    def train(self, nnX, nnY, **kwargs):
        ''' 
        Trains a new model using transformed feature(s) and scaled label(s)
        
        Args:
        nnX: array, for minimal model;  dict, for complete model
        nnY: array, for minimal model;  dict, for complete model
        
        hyperparams_path: str, path for json file with hyperparameters, default: optimized hyperparameters
        custom_params: dict, custom hyperparameters to be updated over pre-loaded hyperparameters
        
        '''

        self._nnX = nnX
        self._nnY = nnY
        
        with open(kwargs.get('hyperparams_path', self.__HYPER_PARAMS_PATH), 'r') as f:
            self.model_params = json.load(f)
            
        if 'custom_params' in kwargs:
            self.model_params.update(kwargs.get('custom_params'))

        self.__create_model()
        
        if self.__model_type == 'complete':
            experimental = self._scaling_functions['first_area'].inverse_transform(self._nnY['first_area']),
            predicted = self.model.predict(self._nnX)[list(self._nnY.keys()).index('first_area')]
        
        elif self.__model_type == 'minimal':
            experimental = self._scaling_functions['first_diff'].inverse_transform(nnY)
            predicted = self.model.predict(self._nnX)
        
        plots.model_performance(
            experimental[-int(self.model_params['val_split']*len(experimental)):], 
            predicted[-int(self.model_params['val_split']*len(experimental)):])
        
        if 'model_path' in kwargs:
            self.__model_path = kwargs.get('model_path')
            self.model.save(filepath = self.__model_path)

    def predict(self, nnX):
        ''' 
        Predicts and returns re-scaled (in absolute units) labels
        
        Args:
        nnX: array, for minimal model;  dict, for complete model
        
        '''
        nnY_normalized = self.model.predict(nnX)
        
        if self.__model_type == 'minimal':
            return self._scaling_functions['first_diff'].inverse_transform(nnY_normalized)
        
        nnY_absolute = {}
        
        for index, key in enumerate(self._scaling_functions.keys()):
            nnY_absolute[key] = self._scaling_functions[key].inverse_transform(nnY_normalized[index])
        
        return nnY_absolute