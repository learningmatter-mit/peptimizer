import numpy as np
import pandas as pd
import json
import textdistance
import random
import math
import utils.predictor as pred

def net_charge(sequence):
    
    '''
    Utility function to calculate net charge of a sequence
    Reference: http://www.chem.ucalgary.ca/courses/351/Carey5th/Ch27/ch27-1-4-2.html

    Parameters
    ----------
    sequence:   str
                peptide sequence

    Returns
    -------
    charge: float
            net charge of sequence
    '''

    acidic = [sequence.count('D'), sequence.count('E'), sequence.count('C'), sequence.count('Y')]
    basic = [sequence.count('R'), sequence.count('K'), sequence.count('H')]

    acidic_pKa = [math.pow(10, 3.65), math.pow(10, 4.25), math.pow(10, 8.37), math.pow(10, 10.46)]
    basic_pKa = [math.pow(10, 10.76), math.pow(10, 9.74), math.pow(10, 7.59)]

    basic_coeff = [x*(1/(x+math.pow(10, 7))) for x in basic_pKa]
    acidic_coeff = [math.pow(10, 7)/(x+math.pow(10, 7)) for x in acidic_pKa]

    charge = - sum(np.multiply(acidic_coeff, acidic)) + sum(np.multiply(basic_coeff, basic))
    return charge

def mutate(sequence, ORIGINAL_AAS, SEQ_LIST, MAX_LEN):
    
    '''
    Utility function to mutate a sequence

    Parameters
    ----------
    sequence:   str
                peptide sequence
                
    ORIGINAL_AAS:   list
                    allowed single residues which can be inserted or swapped
                    
    SEQ_LIST:   list
                predictor training dataset
                
    MAX_LEN:    int
                maximum feature length

    Returns
    -------
    sequence:   str
                mutated sequence
    '''
    
    def deletion(str_sequence):
        
        '''
        Utility function to mutate a sequence by deleting a random residue

        Parameters
        ----------
        sequence:   str
                    peptide sequence

        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        str_sequence = str_sequence.split(" ")[0]
        toremove = random.randint(0, len(str_sequence) - 1)
        new_str = str_sequence[:toremove] + str_sequence[toremove+1:] + " "
        return new_str

    def insertion(str_sequence, aminoacids=ORIGINAL_AAS):
        
        '''
        Utility function to mutate a sequence by inserting a random residue

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        ORIGINAL_AAS:   list
                        allowed single residues which can be inserted or swapped

        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        str_sequence = str_sequence.split(" ")[0]
        toinsert = random.randint(0, len(str_sequence))
        new_str = str_sequence[:toinsert] + random.choice(aminoacids) + str_sequence[toinsert:]
        return new_str[:MAX_LEN]

    def swap(str_sequence, aminoacids=ORIGINAL_AAS):
        
        '''
        Utility function to mutate a sequence by swapping a random residue

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        ORIGINAL_AAS:   list
                        allowed single residues which can be inserted or swapped

        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        existingaas = [i for i in set(str_sequence) if i != " "]
        aatoreplace = random.choice(existingaas)
        aaindices = [index for index, value in enumerate(str_sequence) if value == aatoreplace]
        indextoreplace = random.choice(aaindices)
        new_str = str_sequence[:indextoreplace] + random.choice(aminoacids) + str_sequence[indextoreplace+1:]
        return new_str

    def hybrid(str_sequence, hybrid_list = SEQ_LIST):
        
        '''
        Utility function to mutate a sequence by replacing a random motif  with 
        another random motif from a random sequence 

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        SEQ_LIST:   list
                    predictor training dataset

        Returns
        -------
        sequence:   str
                    mutated sequence
        '''
        
        str_sequence = str_sequence.split(" ")[0]
        tohybrid = random.randint(0, len(str_sequence)-1)
        hybridlen = random.randint(0, int((len(str_sequence) - tohybrid)/5))+1
        hybrid_from = random.choice(hybrid_list)
        leastlen = max(len(min(hybrid_list, key=len)), len(hybrid_from))
        while leastlen < hybridlen:
            hybrid_from = random.choice(hybrid_list)
            leastlen = len(hybrid_from)
        index_hybrid_max = leastlen-hybridlen
        if index_hybrid_max > 0:
            index_hybrid_max = index_hybrid_max - 1
        index_hybrid = random.randint(0, index_hybrid_max)
        hybrid_from = hybrid_from[index_hybrid:index_hybrid+hybridlen]
        new_str = str_sequence[:tohybrid] + hybrid_from + str_sequence[tohybrid+hybridlen:]
        return new_str
    
    mutation_list = [insertion, deletion, swap, hybrid]
    mutation = random.choice(mutation_list)
    
    return mutation(sequence)

def optimize(list_seeds, 
             model_cnn_path, 
             residue_path = './utils/smiles.json', 
             pred_dataset_path = './dataset/predictor_dataset.csv', 
             pred_dataset_stats_filepath = './dataset/predictor_dataset_stats.json', 
             features_max = 108, NBITS = 2048, RADIUS = 3, max_attempts = 20
            ):

    '''

    User function for optimization of a list of seeds

    Parameters
    ----------
    list_seeds: list
                peptide sequences sampled from the generator to be used as seeds in the optimization
                
    model_cnn_path: str
                    filepath of pre-trained predictor
                
    (optional)
    residue_path:   str
                    filepath for residues
                    
    pred_dataset_path:  str
                        filepath for predictor training dataset
                
    pred_dataset_stats_filepath:    str
                                    filepath of training dataset statistics

    features_max:   int, default: 108 (same as used for training of predictor)
                    length of the feature map, or maximum permissible length of sequence to be trained and later predicted

    NBITS:  int, default: 2048 (same as used for training of predictor)
            number of Morgan fingerprint bits

    RADIUS: int, default: 3 (same as used for training of predictor)
            number of Morgan fingerprint bits
            
    max_attempts:   int, default: 100 (1000 used in original model)
                    maximum number of mutations per seed

    Returns
    -------
    ga_df:  dataframe
            sequences, intensity, length, relative arginine count, relative charge

    '''
    
    def fitnessfunc(sequence, i):
        
        '''
        Utility function to evaluate the fitness of a mutated sequence against an objective function 

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        i:  int
            index of seed sequence

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

        nn_pred = pred.predict(sequence, model_cnn, features_max, NBITS, RADIUS)
        arg_count = (((sequence).count('R')) - dict_data['mean_R_count']) / dict_data['std_R_count']
        len_count = (len(sequence) - dict_data['mean_len_seq'])/dict_data['std_len_seq']
        charge = (net_charge(sequence) - dict_data['mean_charge'])/dict_data['std_charge']

        similarity_training = max([textdistance.jaro_winkler.similarity(sequence, reference)
                                   for reference in SEQ_LIST])

        max_similarity_predicted = 0
        similarity_predicted = 0

        try:
            for k in range(1, i+1):
                predicted_sequences = list(Seq_df.at[k, 'new_dict'].keys())
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
    
    
    def genetic_algorithm(sequence, i, max_attempts, 
                          T = 100.0):
        
        '''
        Utility function to implement the directed evolution

        Parameters
        ----------
        sequence:   str
                    peptide sequence
                    
        i:  int
            index of seed sequence
            
        max_attempts:   int
                        maximum number of mutations per seed
        
        (optional)
        T:  float, default: 100.0
            parameter used for simulated annealing
            
        Returns
        -------
        None
        
        '''

        oldseq = sequence
        for attempt in range(max_attempts):

            oldvalue, nn_pred, arg_count, len_count, charge, similarity = fitnessfunc(oldseq, i)
            newseq = mutate(oldseq, ORIGINAL_AAS, SEQ_LIST, features_max)
            newvalue, nn_pred, arg_count, len_count, charge, similarity = fitnessfunc(newseq, i)

            delta = newvalue - oldvalue

            if (newvalue * np.exp(-delta/T)) > oldvalue:
                oldseq = newseq
                Seq_df.at[i, 'new_dict'][newseq] = [newvalue, nn_pred, arg_count, len_count, charge, similarity]
            else:
                continue
    
    print('Setting up Optimizer')
    
    SEQ_LIST = pd.read_csv(pred_dataset_path)['sequences'].tolist()
    
    with open(residue_path) as f:
        lookupsmiles = json.load(f)
        
    ORIGINAL_AAS = [i for i in lookupsmiles.keys() if i not in '@#1234567890']
    
    model_cnn = pred.predictor_model(model_cnn_path)
    
    with open(pred_dataset_stats_filepath) as f:
        dict_data = json.load(f)
    
    
    Seq_df = pd.DataFrame(columns=['seed', 'new_dict'])
    
    for counter, seed in enumerate(list_seeds):
        Seq_df.at[counter, 'seed'] = seed
        Seq_df.at[counter, 'new_dict'] = {}

    new_seq_dict = {}

    for i in range(Seq_df.shape[0]):
        print ('Optimizing Seed ', i+1)
        genetic_algorithm(Seq_df.at[i, 'seed'], i, max_attempts)
        new_seq_dict.update(Seq_df.at[i, 'new_dict'].items())

    print ('Post-Processing Optimized Sequences')
    
    df_temp = pd.DataFrame.from_dict(new_seq_dict, orient='index')
    df_temp = df_temp.rename(columns = {
        0:'value', 
        1:'norm_intensity', 
        2:'norm_arg_count', 
        3:'norm_len_count', 
        4:'charge',
        5:'similarity'}
                                  )
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
            ga_df.at[ga_df_index, 'intensity'] = (df_temp.at[counter, 'norm_intensity']
                                                  *dict_data['std_intensity'] + dict_data['mean_intensity'])
            ga_df.at[ga_df_index, 'length'] = df_temp.at[counter, 'len']
            ga_df.at[ga_df_index, 'relative_Arg'] = df_temp.at[counter, 'arg_count']/df_temp.at[counter, 'len']
            ga_df.at[ga_df_index, 'relative_charge'] = df_temp.at[counter, 'net_charge']/df_temp.at[counter, 'len']
            
            ga_df_index = ga_df.shape[0]
    
    return ga_df.sort_values(['intensity'], ascending=False)