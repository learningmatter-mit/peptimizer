import pandas as pd
import numpy as np
from matplotlib import pylab as plt
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
import random
from sklearn.preprocessing import StandardScaler

import keras
from keras import Sequential
from keras.layers import Conv1D, Dense, Flatten, BatchNormalization, Dropout, Activation
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K

from utils.fingerprint import Fingerprint_Generation

def load_data(PREDICTOR_DATASET = './dataset/predictor_dataset.csv',
              validation_split = 0.2,
              features_max = 108,
              NBITS = 2048,
              RADIUS = 3
             ):
    
    '''
    User function to load the dataset
    Dataset used to train the predictor is available at ./dataset/predictor.csv
    
    Custom dataset as a *.csv file may be used. The dataset should contain a list of sequences and activity
    The column headers should be 'sequences' and 'intensity'
    
    Parameters
    ----------
    (optional)
    PREDICTOR_DATASET:  str, default: predictor training dataset
                        filepath of the dataset
    
    validation_split:   float, default: 0.2 (same as original model)
                        train with (1-validation_split) of the dataset, and validate with the rest

    features_max:   int, default: 108 (same as used for training of predictor)
                    length of the feature map, or maximum permissible length of sequence to be trained and later predicted
                    
    NBITS:  int, default: 2048 (same as used for training of predictor)
            number of Morgan fingerprint bits
            
    RADIUS: int, default: 3 (same as used for training of predictor)
            number of Morgan fingerprint bits

    Returns
    -------
    X:  array of dimension (number of sequences, features_max, NBITS)
        shuffled sequences used to train the predictor
    
    y:  array of dimension (number of sequences, 1)
        shuffled (with same index as nnX) intensities used to train the predictor
        
    X_valid:    array of dimension (number of sequences, features_max, NBITS)
                shuffled sequences used to validate the predictor while training
    
    y_valid:    array of dimension (number of sequences, 1)
                shuffled (with same index as nnX) intensities used to validate the predictor while training

            
    dict_data:  dict
                mean and standard deviation of intensity
    '''
    
    print('Loading Data for Training of Predictor')
    df = pd.read_csv(PREDICTOR_DATASET)
    
    X_df = pd.DataFrame(columns=['sequence', 'feature'])
    Y_df = pd.DataFrame(columns=['intensity'])
    
    print ('Featurizing Data for Predictor')
    for i in range(0, df.shape[0]):
        
        seq = df['sequences'][i]
        X_df.at[i, 'sequence'] = seq
        X_df.at[i, 'feature'] = nn_feature(seq, 
                                           features_max = features_max,
                                           NBITS = NBITS,
                                           RADIUS = RADIUS
                                          )

        Y_df.at[i, 'intensity'] = df['intensity'][i]
        
    X = np.ndarray(shape=(X_df.shape[0],features_max,NBITS), dtype=int)
    for i in range(0,X_df.shape[0]):
        X[i] = X_df.at[i, 'feature']
        
    dict_data = {}
    dict_data['mean_Intensity'] = Y_df['intensity'].mean()
    dict_data['std_Intensity'] = Y_df['intensity'].std()

    scaler = StandardScaler()
    Y_df.fillna(0, inplace=True) #There are few missing values in the Spreadsheet, so replacing them with 0
    Y_df[['intensity']] = scaler.fit_transform(Y_df[['intensity']])

    y = np.asarray(Y_df['intensity'].values.tolist())
    
    indices = np.random.RandomState(seed=108).permutation(np.arange(X.shape[0]))

    X = X[indices]
    y = y[indices]
    
    X_valid = X[indices][-int(len(indices)*validation_split):]
    y_valid = y[indices][-int(len(indices)*validation_split):]*dict_data['std_Intensity'] + dict_data['mean_Intensity']
    
    return X, y, X_valid, y_valid, dict_data
    
def nn_feature(sequence, features_max = 108, NBITS = 2048, RADIUS = 3):
        
    '''
    Utility function to generate feature map
    
    Parameters
    ----------
    sequence:   str
                peptide sequence
    
    (optional)
    features_max:   int
                    length of the feature map, or maximum permissible length of sequence to be trained and later predicted
                    
    NBITS:  int
            number of Morgan fingerprint bits
            
    RADIUS: int
            number of Morgan fingerprint bits

    Returns
    -------
    fp_seq: array of dimension (features_max, NBITS)
            feature map of sequence
    
    '''
        
    fp = Fingerprint_Generation(nbits = NBITS, radius = RADIUS)

    fp_seq = fp.seq(sequence)
    n_rows = features_max - len(sequence)
    shape_padding = (n_rows, NBITS)
    padding_array = np.zeros(shape_padding)
    fp_seq = np.concatenate((fp_seq, padding_array), axis = 0)

    return fp_seq

def train_model(X, y, 
                epochs=5, batch_size=16, validation_split=0.2, 
                save_checkpoint=False, filepath=None
               ):
    
    '''
    User function to train the generator.
    
    Parameters
    ----------
    X:  array of dimension (number of sequences, features_max, NBITS)
        shuffled sequences used to train the predictor
    
    y:  array of dimension (number of sequences, 1)
        shuffled (with same index as nnX) intensities used to train the predictor
    
    (optional)
    epochs: int, default: 5 (original model trained at 400 epochs)
            number of epochs to train the model
            
    batch_size: int, default: 16 (original model trained at 25)
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
    
    print ('Creating Model for Predictor')
    
    model = Sequential()

    DIM = 256
    SIZE = 2

    model.add(Conv1D(DIM, SIZE, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.1))
    model.add(Conv1D(DIM, SIZE))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Conv1D(DIM, SIZE))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(DIM))
    model.add(Activation('softplus'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=optimizer,
                  loss='mse')
    
    callbacks_list = []

    if save_checkpoint == True:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
    
    model.fit(X, y, 
                   epochs=epochs, batch_size=batch_size, validation_split=validation_split, 
                   callbacks=callbacks_list)
    
    return model

def predictor_model(filepath = './model/pre_trained/predictor.hdf5'):
    
    '''
    User function to load the pre-trained model. 
    
    Parameters
    ----------
    (optional)
    filepath:   str, default: pre-trained model
                filepath of the pre-trained model to be used for sampling of new sequences
            
    Returns
    -------
    model_cnn: keras model
                pre-trained model to predict activity of new sequences
    
    '''
    
    print ('Loading Model for Predictor')
    
    return load_model(filepath)
    
def predict(sequence, model_cnn,
            features_max = 108,
            NBITS = 2048,
            RADIUS = 3
           ):
    
    '''
    User function to predict the normalized intensity for an unknown sequence
    Utility function to predict the normalized intensity for an mutant during optimization
    
    
    Parameters
    ----------
    sequence:   str
                peptide sequence

    model_cnn: keras model
                pre-trained model to sample new sequences
    
    (optional)
    features_max:   int, default: 108 (same as used for training of predictor)
                    length of the feature map, or maximum permissible length of sequence to be trained and later predicted
                    
    NBITS:  int, default: 2048 (same as used for training of predictor)
            number of Morgan fingerprint bits
            
    RADIUS: int, default: 3 (same as used for training of predictor)
            number of Morgan fingerprint bits

    Returns
    -------
    y:  float
        predicted intensity (normalized)
        
    '''

    return model_cnn.predict(np.asarray([nn_feature(sequence, 
                                           features_max = features_max,
                                           NBITS = NBITS,
                                           RADIUS = RADIUS
                                          )]))[0][0]