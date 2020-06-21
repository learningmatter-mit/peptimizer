import os
app_dir = os.getcwd()

import sys
sys.path.append(app_dir + '/utils/utils_common')

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=FutureWarning)

from calc_charge import net_charge
import genetic_mutations

import numpy as np
import pandas as pd
import json
import textdistance
import random
import math
from utils.utils_cpp import cpp_predictor

class Optimizer:
    def __init__(self, **kwargs):
        '''
        Initialize a Generator object which can be used in both modes - train and generate
        
        Examples:
        optimizer = optimizer.Optimizer(
            model_path = '/path/to/model.hdf5',
            data_path = '/path/to/dataset.csv',
            smiles_path = '/path/to/smiles.json',
            stats_path = '/path/to/stats.json',
            fp_radius = 3,
            fp_bits = 1024,
            seq_max = 108
        )
        
        optimizer.optimize()

        Parameters
        ----------
        
        model_path:     str
                        filepath of pre-trained model, to be used when using for prediction

        data_path:      str
                        filepath of current dataset, to be used when training
                        Custom dataset as a *.csv file may be used. The dataset should contain a list of sequences and activity
                        The column headers should be 'sequences' and 'intensity'
                    
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
                        
        max_attempts:   int, default: 100 (1000 used in original model)
                        maximum number of mutations per seed
                        
        T:              float, default: 100.0
                        parameter used for simulated annealing
         
        '''
        
        self.__model_path = kwargs.get('model_path')
        self.__data_path = kwargs.get('data_path')
        self.__smiles_path = kwargs.get('smiles_path')
        self.__stats_path = kwargs.get('stats_path')
        self.__fp_radius = kwargs.get('fp_radius')
        self.__fp_bits = kwargs.get('fp_bits')
        self.__seq_max = kwargs.get('seq_max')

        self.seq_list = pd.read_csv(self.__data_path)['sequences'].tolist()
        
        self.predictor = cpp_predictor.Predictor(
            model_path = self.__model_path, 
            smiles_path = self.__smiles_path,
            stats_path = self.__stats_path,
            fp_radius = self.__fp_radius,
            fp_bits = self.__fp_bits,
            seq_max = self.__seq_max
            )
        
        self.genetic_mutations = genetic_mutations.Genetic_Mutations(
            data_path = self.__data_path, 
            smiles_path = self.__smiles_path,
            seq_max = self.__seq_max
        )
        
        with open(self.__stats_path) as f:
            self.dict_data = json.load(f)
    
    def __fitnessfunc(self, sequence):
        
        '''
        Utility function to evaluate the fitness of a mutated sequence against an objective function 

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        Returns
        -------
        value:   float
                 fitness of mutated sequence against the objective function
                 
        nn_pred:    float
                    normalized predicted intensity

        arg_count:  float
                    normalized arginine count
                    
        len_count:  float
                    normalized length
                    
        charge: float
                normalized charge
                
        similarity: float
                    maximum similarity when compared to the prior predicted sequences and predictor training dataset
        '''

        nn_pred = (self.predictor.predict(sequence) - self.dict_data['mean_intensity']) / self.dict_data['std_intensity']
        arg_count = (((sequence).count('R')) - self.dict_data['mean_R_count']) / self.dict_data['std_R_count']
        len_count = (len(sequence) - self.dict_data['mean_len_seq'])/self.dict_data['std_len_seq']
        charge = (net_charge(sequence) - self.dict_data['mean_charge'])/self.dict_data['std_charge']

        similarity_training = max([textdistance.jaro_winkler.similarity(sequence, reference)
                                   for reference in self.seq_list])

        max_similarity_predicted = 0
        similarity_predicted = 0

        try:
            for k in range(1, self.i+1):
                predicted_sequences = list(self.seq_df.at[k, 'new_dict'].keys())
                similarity_predicted = max(np.asarray([
                    textdistance.jaro_winkler.similarity(sequence, predicted_sequences[j])
                                                  for j in range(len(predicted_sequences))]))
                max_similarity_predicted = max(similarity_predicted, max_similarity_predicted)
        except:
            pass

        similarity = max(similarity_training, max_similarity_predicted)

        value =  0.5*nn_pred - 0.5*(
            0.5*arg_count + 0.2*len_count - 0.1*charge + similarity)

        return value, nn_pred, arg_count, len_count, charge, similarity

    def __genetic_algorithm(self):
        
        '''
        Utility function to implement the directed evolution
        
        '''

        oldseq = self.seq_df.at[self.i, 'seed']
        
        for attempt in range(self.__max_attempts):
            oldvalue, nn_pred, arg_count, len_count, charge, similarity = self.__fitnessfunc(oldseq)
            newseq = self.genetic_mutations.mutate(oldseq)
            newvalue, nn_pred, arg_count, len_count, charge, similarity = self.__fitnessfunc(newseq)

            delta = newvalue - oldvalue

            if (newvalue * np.exp(-delta/self.__T)) > oldvalue:
                oldseq = newseq
                self.seq_df.at[self.i, 'new_dict'][newseq] = [newvalue, nn_pred, arg_count, len_count, charge, similarity]
            else:
                continue

    def __post_process(self, new_seq_dict):
        '''
        Utility function to post process the mutations

        Parameters
        ----------
        new_seq_dict:   dict
                        dictionary of new sequences along with intensity and other parameters
                    
        Returns
        -------
        ga_df:  dataframe
                sequences, intensity, length, relative arginine count, relative charge
                
        '''
        
        df_temp = pd.DataFrame.from_dict(new_seq_dict, orient='index')
        df_temp = df_temp.rename(columns = {
            0:'value', 
            1:'norm_intensity', 
            2:'norm_arg_count', 
            3:'norm_len_count', 
            4:'charge',
            5:'similarity'}
                                      )
        df_temp['norm_intensity'] = (
            df_temp['norm_intensity'] * self.dict_data['std_intensity']) + self.dict_data['mean_intensity']
        
        df_temp['arg_count'] = df_temp.index.str.count('R')
        df_temp['sequences'] = df_temp.index
        df_temp['net_charge'] = df_temp['sequences'].apply(net_charge)
        df_temp['len'] = df_temp.index.map(len)

        df_temp.reset_index(drop=False, inplace=True)

        ga_df = pd.DataFrame(columns=['sequences', 'intensity', 'length', 'relative_Arg', 'relative_charge'])
        ga_df_index = 0

        for counter in range(df_temp.shape[0]):
            if '2' not in list(df_temp.at[counter, 'sequences']) and '3' not in list(df_temp.at[counter, 'sequences']):
                ga_df.at[ga_df_index, 'sequences'] = df_temp.at[counter, 'sequences']
                ga_df.at[ga_df_index, 'intensity'] = df_temp.at[counter, 'norm_intensity']
                ga_df.at[ga_df_index, 'length'] = df_temp.at[counter, 'len']
                ga_df.at[ga_df_index, 'relative_Arg'] = df_temp.at[counter, 'arg_count']/df_temp.at[counter, 'len']
                ga_df.at[ga_df_index, 'relative_charge'] = df_temp.at[counter, 'net_charge']/df_temp.at[counter, 'len']

                ga_df_index = ga_df.shape[0]
                
        return ga_df.sort_values(['intensity'], ascending=False)
    
        
    '''
    ----------------------------------------------------------------
                           PUBLIC FUNCTIONS
    ----------------------------------------------------------------
    '''

    def optimize(self, list_seeds, **kwargs):
        
        '''
        Parameters
        ----------
        
        list_seeds:     list, seq
                        seed sequences for the optimizer

        max_attempts:   int
                        maximum number of mutations per seed
                        
        T:  float
            parameter used for simulated annealing
            
            
        Returns
        -------
        ga_df:  dataframe
                sequences, intensity, length, relative arginine count, relative charge

        '''
        
        self.__max_attempts = kwargs.get('max_attempts', 1000)
        self.__T = kwargs.get('T', 1)

        print('Setting up Optimizer')
        
        self.seq_df = pd.DataFrame(columns=['seed', 'new_dict'])
        
        for counter, seed in enumerate(list_seeds):
            self.seq_df.at[counter, 'seed'] = seed
            self.seq_df.at[counter, 'new_dict'] = {}
            
        new_seq_dict = {}
        
        for self.i in range(self.seq_df.shape[0]):
            print ('Optimizing Seed ', self.i+1)
            self.__genetic_algorithm()
            new_seq_dict.update(self.seq_df.at[self.i, 'new_dict'].items())
            
        print ('Post-Processing Optimized Sequences')
        
        return self.__post_process(new_seq_dict)