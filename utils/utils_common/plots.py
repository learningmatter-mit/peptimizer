import numpy as np
from matplotlib import pyplot as plt

def model_performance(experimental, predicted):

    '''
    Utility function to plot performance of model against validation dataset

    '''
    
    plt.scatter(experimental, predicted)

    plt.xlabel('Experimental Intensity', fontdict={'size':16})
    plt.ylabel('Predicted Intensity', fontdict={'size':16})
    plt.tick_params(labelsize=14)
    plt.show()

def positive_activation_feature_map(sequence, output, cmap = 'viridis'):
    
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

    
def positive_activation_avg_residues(sequence, output, cmap = 'viridis'):
    
    residue_mean = np.mean(output, axis = 0)
    
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
    ax.set_aspect(0.01)
    
    cbar = ax.figure.colorbar(im, ax=ax, ticks= [], pad = 0.15)
    cbar.ax.set_aspect(11)
    cbar.set_label(label='Activation -->', size=18)
    
    plt.show()

def positive_activation_avg_fingerprint(sequence, output, cmap = 'viridis'):
    
    position_mean = np.mean(output, axis = 1)
    
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
    
def synthesis_overlay_plots(sequence, predicted, ground_truth = None):
    print ('Positive activation averaged over fingerprints')
    
    sequence = sequence[::-1]
    
    fig, ax = plt.subplots()
    plt.plot(np.arange(len(sequence)), predicted, label='predicted')
    
    if ground_truth is not None:
        plt.plot(np.arange(len(sequence)), ground_truth, label='ground_truth')
    
    plt.xticks(np.arange(len(sequence)), list(sequence))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Residue', fontsize=20)
    plt.ylabel('Relative change', fontsize=20)
    plt.legend(fontsize=12)
    plt.ylim(-1.5, 1.5)
    plt.show()