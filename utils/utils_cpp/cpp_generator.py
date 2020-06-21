import numpy as np
import pandas as pd
import random
import math

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, LSTM, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class Generator:
    def __init__(self, **kwargs):
        '''
        Initialize a Generator object which can be used in both modes - train and generate
        
        Examples:
        1. Training a generator - 
        
            generator.Generator(
                data_path = '/path/to/dataset.txt',
                seq_length = 10
            )
            
            generator.train_model()
            
        2. Using a pre-trained model - 
        
            generator.Generator(
                model_path = '/path/to/model.hdf5',
                data_path = '/path/to/dataset.txt',
                seq_length = 10
            )
            
            generator.generate_seed(n_seeds = 2, seed_length = 30)

        Parameters
        ----------
        data_path:      str
                        filepath of current dataset, to be used when training
                        Custom dataset as a *.txt file may be used. 
                        The dataset should contain a list of sequences without any header.
                    
        model_path:     str
                        filepath of pre-trained model, to be used when using for generation of seeds
                    
        seq_length:     int
                        length of seed sequence to train/sample generator
                                            
        '''
        
        self.__data_path = kwargs.get('data_path')
        self.__seq_length = kwargs.get('seq_length', 10)
        
        if 'model_path' in kwargs:
            self.__model_path = kwargs.get('model_path')
            self.model = load_model(self.__model_path)

        elif 'model_path' not in kwargs:
            self.model_params = {
            'epochs': 2,
            'batch_size': 100,
            'val_split': 0.3,
            'save_checkpoint': False,
            'checkpoint_filepath': None,
            'cell_size': 32,
            'dropout': 0.1,
            'optimizer': 'adam',
            'loss_function': 'categorical_crossentropy'
        }
            
        else:
            raise NameError("Cannot understand the utility.")

            
    def __load_data(self):
        
        '''
        Utility function to load the dataset
        
        Returns
        ----------
        self.X: array of dimension (number of seed sequences, seq_length, 1)
                seed sequences used to train the generator

        self.y: array of dimension (number of seed sequences, 1)
                next amino acid (for respective seed sequences) used to train the generator
        
        '''
        
        print ('Loading Data for Training of Generator')
        print ('Pre-Processing Data for Generator')
        
        self.__pre_process()

        print ('Featurizing Data for Generator')
        n_patterns = len(self.dataX)

        self.X = np.reshape(self.dataX, (n_patterns, self.__seq_length, 1))

        self.X = self.X / float(self.n_vocab)
        self.y = to_categorical(self.dataY)
                
        indices = np.random.RandomState(seed=108).permutation(np.arange(self.X.shape[0]))

        self.X = self.X[indices]
        self.y = self.y[indices]
        
        
    def __pre_process(self):

        '''
        Utility function to pre-process the dataset. Used for training and generating random seeds.

        Key Parameters
        -------
        self.dataX: array of dimension (number of seed sequences, seq_length)
                    seed sequences to be processed for training

        self.dataY: array of dimension (number of seed sequences)
                    next amino acid (for respective seed sequences) used to train the generator

        self.n_vocab:   int
                        number of unique characters (residues, linkers) in the text

        self.int_to_char:   dict
                            map index of predicted next amino acid to character
                            
        '''

        raw_text = open(self.__data_path).read()

        # Unique characters in the dataset
        chars = sorted(list(set(raw_text))) 
        char_to_int = dict((c, i) for i, c in enumerate(chars))
        
        self.int_to_char = dict((i, c) for i, c in enumerate(chars))

        self.n_vocab = len(chars)
        n_chars = len(raw_text)

        self.dataX = []
        self.dataY = []
        
        for i in range(0, len(raw_text) - self.__seq_length, 1):
            seq_in = raw_text[i:i + self.__seq_length]
            seq_out = raw_text[i + self.__seq_length]
            if seq_out != "\n" and "\n" not in seq_in:
                self.dataX.append([char_to_int[char] for char in seq_in])
                self.dataY.append(char_to_int[seq_out])

    
    def __random_seed(self):

        '''
        Utility function to generate a seed for the sampling using the generator.
        The seed does not contain any new line character, initiating character or empty spaces

        Returns
        -------
        test_seed:  array of dimension (seq_length)
                    seed sequence for the generator

        '''

        test_seed = self.dataX[np.random.randint(0, len(self.dataX)-1)]
        
        while 0 in test_seed:
            test_seed = self.dataX[np.random.randint(0, len(self.dataX)-1)]
        
        return test_seed

    def __optimizer_seed(self):  

        '''
        Utility function to generate a seed for the optimizer

        Returns
        -------
        seq:    str
                seed sequence for the optimizer

        '''

        self.__pre_process()

        pattern = self.__random_seed()
        seq = [self.int_to_char[value] for value in pattern]

        for i in range(self.__seed_length - self.__seq_length):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.n_vocab)
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            if index != 0:
                result = self.int_to_char[index]
                seq += result
                pattern.append(index)
                pattern = pattern[1:len(pattern)]
            else:
                pattern = self.__generator_seed()

        return ''.join(seq)

        
    '''
    ----------------------------------------------------------------
                           PUBLIC FUNCTIONS
    ----------------------------------------------------------------
    '''

    def train_model(self, **kwargs):
        
        '''
        Public function to train the predictor.
        The model used in the paper was based on a more advanced implementation of LSTM, called Nested LSTM.
        Keras implementation for Nested LSTM can be found here - https://github.com/titu1994/Nested-LSTM

        Parameters
        ----------
        (optional)
        
        model_path:     str
                        filepath to save model
                        
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
                            
                            cell_size: int
                            cell size of LSTM model
                            
                            dropout: float
                            dropout to regularize training, 0.0 < value < 1.0
                                                        
                            optimizer: str
                            optimizer
                            
                            loss_function: str
                            loss function for predictor

        '''
        
        if 'model_params' in kwargs:
            self.__model_params = (kwargs.get('model_params'))

            for key in self.__model_params:
                self.model_params[key] = self.__model_params[key]

        self.__load_data()
        
        print ('Starting Training of Generator')
        
        model_lstm = Sequential()
        
        model_lstm.add(LSTM(self.model_params['cell_size'], 
                            input_shape=(self.X.shape[1], self.X.shape[2]), return_sequences=True))
        model_lstm.add(LSTM(self.model_params['cell_size']))
        model_lstm.add(Activation('relu'))
        model_lstm.add(Dropout(self.model_params['dropout']))
        model_lstm.add(Dense(self.y.shape[1], activation='softmax'))
        
        model_lstm.compile(
            loss=self.model_params['loss_function'], 
            optimizer=self.model_params['optimizer'], 
            metrics=['accuracy']
        )
                
        callbacks_list = []

        if self.model_params['save_checkpoint'] == True:
            checkpoint = ModelCheckpoint(self.model_params['checkpoint_filepath'] + 
                                         "generator-epoch{epoch:02d}-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}.hdf5", 
                                         monitor='val_acc', 
                                         save_best_only=True, 
                                         mode='max')
            callbacks_list = [checkpoint]
        
        model_lstm.fit(self.X, self.y, 
                       epochs=self.model_params['epochs'], 
                       batch_size=self.model_params['batch_size'], 
                       validation_split=self.model_params['val_split'], 
                       callbacks=callbacks_list
                      )

        self.model = model_lstm
        
        if 'model_path' in kwargs:
            self.__model_path = kwargs.get('model_path')
            model_lstm.save(filepath = self.__model_path)
            
            
    def generate_seed(self, **kwargs):  
    
        '''
        Public function to generate list of seeds for the optimizer

        Parameters
        ----------
        n_seeds:   int
                   number of seeds required

        seed_length:  int
                      length of the seed sequences for the optimizer

        Returns
        -------
        list_seeds: list, seq
                    seed sequences for the optimizer

        '''

        print ('Generating Seeds for Optimizer')

        self.__n_seeds = kwargs.get('n_seeds')
        self.__seed_length = kwargs.get('seed_length')

        list_seeds = []

        for counter in range(self.__n_seeds):
            print ('Generating Seed ', counter+1)
            list_seeds += [self.__optimizer_seed()]

        return list_seeds