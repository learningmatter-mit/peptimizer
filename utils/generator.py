import numpy as np
import pandas as pd
import random
import math
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs

from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Dropout, Flatten, Activation, Conv1D
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
import keras.backend as K
import tensorflow as tf

from utils.nested_lstm import NestedLSTM

def load_data(GENERATOR_DATASET = './dataset/generator_dataset.txt', 
              seq_length = 5):
    
    '''
    User function to load the dataset.
    Dataset used to train the generator is available at ./dataset/generator.txt
    Custom dataset as a *.txt file may be used. The dataset should contain a list of sequences without any header.
    
    Parameters
    ----------
    GENERATOR_DATASET:  str
                        filepath of the dataset
    
    (optional)
    seq_length: int, default: 5
                length of the seed sequence for the training of the generator

    Returns
    -------
    X:  array of dimension (number of seed sequences, seq_length, 1)
        seed sequences used to train the generator
    
    y:  array of dimension (number of seed sequences, 1)
        next amino acid (for respective seed sequences) used to train the generator
    '''
    
    print ('Loading Data for Training of Generator')
    print ('Pre-Processing Data for Generator')
    dataX, dataY, n_vocab = pre_process(GENERATOR_DATASET, seq_length, utility = 'train')
    
    print ('Featurizing Data for Generator')
    n_patterns = len(dataX)

    X = np.reshape(dataX, (n_patterns, seq_length, 1))

    X = X / float(n_vocab)
    y = np_utils.to_categorical(dataY)
    
    return X, y
    

def pre_process(GENERATOR_DATASET, seq_length, utility):
    
    '''
    Utility function to pre-process the dataset.
    Used for training and generating random seeds.
    
    Parameters
    ----------
    GENERATOR_DATASET:  str
                        filepath of the dataset
    
    seq_length: int
                length of the seed sequence for the training of the generator
                
    utility:    str
                reason to pre-process - valid options: 'train', 'seed'

    Returns
    -------
    utility = 'train'
    dataX:  array of dimension (number of seed sequences, seq_length)
            seed sequences to be processed for training
    
    dataY:  array of dimension (number of seed sequences)
            next amino acid (for respective seed sequences) used to train the generator
            
    n_vocab:    int
                number of unique characters (residues, linkers) in the text
                
    utility = 'seed'
    dataX:  array of dimension (number of seed sequences, seq_length)
            pool to choose random seed for the generation
            
    int_to_char:    dict
                    map index of predicted next amino acid to character
    '''

    raw_text = open(GENERATOR_DATASET).read()
    
    # Unique characters in the dataset
    chars = sorted(list(set(raw_text))) 
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    n_vocab = len(chars)
    n_chars = len(raw_text)

    dataX = []
    dataY = []
    for i in range(0, len(raw_text) - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        if seq_out != "\n":
            dataX.append([char_to_int[char] for char in seq_in])
            dataY.append(char_to_int[seq_out])
    
    if utility == 'train':
        return dataX, dataY, n_vocab
    
    elif utility == 'seed':
        return dataX, int_to_char, n_vocab
    
def train_model(X, y, 
                epochs=5, batch_size=256, validation_split=0.2, 
                save_checkpoint=False, filepath=None
               ):
    
    '''
    User function to train the generator.
    
    Parameters
    ----------
    X:  array of dimension (number of seed sequences, seq_length, 1)
        seed sequences used to train the generator
    
    y:  array of dimension (number of seed sequences, 1)
        next amino acid (for respective seed sequences) used to train the generator
    
    (optional)
    epochs: int, default: 5 (original model trained at 1000 epochs)
            number of epochs to train the model
            
    batch_size: int, default: 256 (original model trained at 256)
                batch_size for the model
                
    validation_split:   float, default: 0.2 (same as original model)
                        train with (1-validation_split) of the dataset, and validate with the rest
            
    save_checkpoint:    bool, default: False
                        to save or not to save intermediate models

    filepath:   str
                location to save intermediate models

    Returns
    -------
    model_lstm: keras model
                trained generator
    
    '''    
    model_lstm = create_model()
    
    callbacks_list = []

    if save_checkpoint == True:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
    
    print ('Starting Training of Generator')
    
    model_lstm.fit(X, y, 
                   epochs=epochs, batch_size=batch_size, validation_split=validation_split, 
                   callbacks=callbacks_list)
    
    return model_lstm
    
def create_model():
    
    '''
    Utility function to create a model based on the optimized parameters.
    You can customize the model layers and architecture as desired.
    
    Reference for NestedLSTM implementation - https://github.com/titu1994/Nested-LSTM
    
    Parameters
    ----------
    None
        
    Returns
    -------
    model_lstm: keras model
                compiled model which can be used to train or generate
    
    '''
    print ('Creating Model for Generator')

    model_lstm = Sequential()
    model_lstm.add(LSTM(1024, input_shape=(5, 1), return_sequences=True))
    model_lstm.add(NestedLSTM(1024, depth=4, dropout=0.1, recurrent_dropout=0.0, return_sequences=True))
    model_lstm.add(LSTM(1024, return_sequences=True))
    model_lstm.add(Dropout(0.1))
    model_lstm.add(Activation('relu'))
    model_lstm.add(LSTM(512, return_sequences=True))
    model_lstm.add(Dropout(0.1))
    model_lstm.add(Activation('relu'))
    model_lstm.add(LSTM(512))
    model_lstm.add(Dropout(0.1))
    model_lstm.add(Dense(23, activation='softmax'))
    
    model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model_lstm

def seed_model(filepath = './model/pre_trained/generator.hdf5'):
    
    '''
    Utility function to load the pre-trained model. 
    If you are using one or more layers of NestedLSTM, it is required to create and compile the model.
    This is because NestedLSTM is a custom layer and cannot be loaded directly using keras functions.
    
    Parameters
    ----------
    (optional)
    filepath:   str
                filepath of the pre-trained model to be used for sampling of new sequences
            
    Returns
    -------
    model_lstm: keras model
                pre-trained model to sample new sequences
    
    '''
        
    model_lstm = create_model()

    print ('Loading Model for Generator')

    model_lstm.load_weights(filepath)

    return model_lstm

def generator_seed(dataX):
        
    '''
    Utility function to generate a seed for the sampling using the generator.
    The seed does not contain any new line character, initi or empty spaces
    
    Parameters
    ----------
    dataX:  array of dimension (number of seed sequences, seq_length)
            pool to choose random seed for the generation

    Returns
    -------
    test_seed:  array of dimension (seq_length)
                seed sequence for the generator
                
    '''

    test_seed = dataX[np.random.randint(0, len(dataX)-1)]
    while 0 in test_seed:
        test_seed = dataX[np.random.randint(0, len(dataX)-1)]
    return test_seed

def optimizer_seed(GENERATOR_DATASET, model_lstm, seq_length = 5, optimizer_seed_length = 30):  
    
    '''
    Utility function to generate a seed for the optimizer
    
    Parameters
    ----------
    GENERATOR_DATASET:  str
                        filepath of the dataset
    
    model_lstm: keras model
                pre-trained model to sample new sequences

    
    (optional)
    seq_length: int, default: 5
                length of the seed sequence for the training of the generator
                
    optimizer_seed_length:  int
                            length of the seed sequences for the optimizer

    Returns
    -------
    seq:    str
            seed sequence for the optimizer
                
    '''

    dataX, int_to_char, n_vocab = pre_process(GENERATOR_DATASET, seq_length, utility='seed')
    
    pattern = generator_seed(dataX)
    seq = [int_to_char[value] for value in pattern]

    for i in range(optimizer_seed_length - seq_length):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model_lstm.predict(x, verbose=0)
        index = np.argmax(prediction)
        if index != 0:
            result = int_to_char[index]
            seq += result
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        else:
            pattern = generator_seed()

    return ''.join(seq)

def generate_seed(number_seeds, 
                  model_lstm_path,
                  GENERATOR_DATASET = './dataset/generator_dataset.txt', 
                  seq_length = 5, optimizer_seed_length = 30):  
    
    '''
    User function to generate dataframe of seeds for the optimizer
    
    Parameters
    ----------
    number_seeds:   int
                    number of seeds required
    
    model_lstm_path:    str
                        filepath for pre-trained model to sample new sequences
                        
    GENERATOR_DATASET:  str
                        filepath of the dataset
    
    (optional)
    seq_length: int, default: 5
                length of the seed sequence for the training of the generator
                
    optimizer_seed_length:  int
                            length of the seed sequences for the optimizer

    Returns
    -------
    seq:    str
            seed sequence for the optimizer
                
    '''
    
    print ('Generating Seeds for Optimizer')

    model_lstm = seed_model(model_lstm_path)
    
    list_seeds = []
    
    for counter in range(number_seeds):
        print ('Generating Seed ', counter+1)
        list_seeds += [optimizer_seed(GENERATOR_DATASET, 
                                                    model_lstm, 
                                                    seq_length = 5, optimizer_seed_length = 30)
                      ]
        
    return list_seeds