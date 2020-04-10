import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import keras.backend as K
from tensorflow.python.framework import ops
from keras.layers.core import Lambda

import utils.predictor as pred

def activation_analysis(sequence, 
                        model_cnn_path, 
                        index_list_path = './utils/indices_2048bit.npy',
                        features_max = 108, NBITS = 2048, RADIUS = 3):
        
    '''
    User function to find positive/negative gradient activation of a sequence using a pre-trained model
    
    Reference: Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based
    localization." Proceedings of the IEEE international conference on computer vision. 2017.

    Parameters
    ----------
    sequence:   str
                peptide sequence
    
    model_cnn_path: str, default: pre-trained model
                    filepath of the pre-trained model to be used for sampling of new sequences
    
    (optional)
    index_list_path: str
                     filepath of non-redundant activated indices for 2048-bit fingerprints of residues 
                     in predictor training dataset
                
    features_max:   int
                    length of the feature map, or maximum permissible length of sequence to be trained and later predicted
                    
    NBITS:  int
            number of Morgan fingerprint bits
            
    RADIUS: int
            number of Morgan fingerprint bits


    Returns
    -------
    charge: float
            net charge of sequence
            
            
    '''
    
    model = pred.predictor_model(model_cnn_path)
    index_list = np.load(index_list_path)
    
    def target_category_loss_output_shape(input_shape):
    # Utility function for shape of output
        return input_shape

    def target_category_loss(x, category_index, nb_classes):
    # Utility function for calculation of loss for regression
        return tf.multiply(x, K.one_hot([category_index], nb_classes))

    def normalize(x):
    # Utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    category_index = [0]
    nb_classes = 1

    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)

    model.add(Lambda(target_layer,
                         output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)

    conv_output =  model.layers[0].output
    conv_input =  model.layers[0].input

    grads = normalize(K.gradients(loss, conv_input)[0])
    gradient_function = K.function([model.layers[0].input], [grads])
    
    
    
    feature = pred.nn_feature(sequence, features_max, NBITS, RADIUS)
    length = len(sequence)

    grads_val = gradient_function([np.array([feature])])
    
    grads_val = np.maximum(grads_val[0][0], 0)

    output = np.multiply(grads_val, feature)[:length,:]
    
    output_temp = []
    for index_output, activated_temp in enumerate(output):
        output_temp += [activated_temp[index_list]]
    output = np.array(output_temp)
    
    position_mean = np.mean(output, axis = 1)
    residue_mean = np.mean(output, axis = 0)
    
    
    cmap = 'viridis'
    
    print ('Positive activation for feature map')
    
    fig, ax = plt.subplots()
    im = ax.imshow(output.T,
                  cmap=cmap)
    ax.set_aspect('auto')
    plt.xticks(np.arange(len(sequence)), list(sequence))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Fingerprint', fontsize=20)
    plt.xlabel('Residue', fontsize=20)
    
    cbar = ax.figure.colorbar(im, ax=ax, ticks= [])
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_aspect(11)
    cbar.set_label(label='Activation -->', size=18)

    plt.show()
    
    print ('Positive activation averaged over residues')
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([residue_mean]).T, 
                  cmap=cmap)
    ax.set_aspect(0.1)
    ax.yaxis.tick_right()
    plt.xticks([], fontsize=0)
    plt.yticks(fontsize=15, )
    plt.ylabel('Fingerprint', fontsize=20)
    ax.set_xticklabels([''])
    cbar = ax.figure.colorbar(im, ax=ax, ticks= [], pad = 0.15)
    cbar.ax.set_aspect(11)
    cbar.set_label(label='Activation -->', size=18)
    
    plt.show()

    print ('Positive activation averaged over fingerprints')
    
    fig, ax = plt.subplots()
    im = ax.imshow(np.array([position_mean]),
                  cmap=cmap)
    plt.yticks([])
    plt.xticks(np.arange(len(sequence)), list(sequence))
    plt.xticks(fontsize=15)
    plt.xlabel('Residue', fontsize=20)
    cbar = ax.figure.colorbar(im, ax=ax, ticks= [], orientation = 'horizontal', pad = 0.25)
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.set_aspect(0.06)
    cbar.set_label(label='Activation -->', size=18)

    plt.show()