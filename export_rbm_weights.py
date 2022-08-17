import numpy as np
import h5py, os
import pandas as pd
import scipy.sparse
import rbm as rbm
from fishualizer_utilities import Zecording

def export_weights_for_fishualizer(rbm, recording=None, 
                                   labeled_cells_only=True,
                                   path_weights='/home/thijs/',
                                   filename_weights='weights_RBM-name.h5',
                                   save_h5=True):
    '''Export weights from RBM to file compatible with fishualizer viewing.
    
    Parameters:
    ---------
        rbm: RBM class
            RBM object with weights to be exported
        recording: Zecording class
            Zecording object (from Fishualizer) that contains zebrafish data recording on which this rbm has been trained'
        labeled_cells_only: bool
            If the RBM has been trained on the Zbrain Atlas-labeled cells only; then we need to account for this when exporting the weights
        path_weights: str
            folder 
        filename_weights: str
            file name 
    '''
    
    if filename_weights[-3:] != '.h5':
        filename_weights = filename_weights + '.h5'
        
    ## Extract RBM weights:
    weights = np.transpose(RBM.weights)  
    
    ## Take care of unlabeled cells if needed
    if labeled_cells_only:
        assert recording != None, 'the zebrafish recording is needed to account for labeled cells only weights'
        n_cells = rec.n_cells
        selected_neurons = np.unique(scipy.sparse.find(rec.labels[:, np.arange(294)])[0])  # cells with zbrain label
        assert weights.shape[0] == len(selected_neurons)  # make sure shape is neurons x time (RBM has only used labelled neurons)
        print(f'n cells: {n_cells}, n labelled cells: {len(selected_neurons)}')
        full_weights = np.zeros((n_cells, weights.shape[1]), dtype='float32')  # make matrix for all cells (including non-labeled)
        full_weights[selected_neurons, :] = weights  # let non-labelled neurons have w=0 for all HU
    else:
        print('Assuming that all cells were used for RBM training')
        full_weights = weights.copy()
        
    ## Export to h5 via pd DataFrame
    df_weights = pd.DataFrame({'hu_' + str(ii).zfill(3):
                                np.squeeze(full_weights[:, ii]) for ii in range(full_weights.shape[1])})  # save as pd df with each column = one weight vector
    if save_h5:
        df_weights.to_hdf(os.path.join(path_weights, filename_weights), key='all')  # store as h5 file

    ## To view the weights in the Fishualizer: 
    ## Launch the Fishualizer
    ## Load the main data set (= rec) using File -> Open data
    ## Load the saved weights (= df_weights) using File -> Add static data -> Choose Dataset
        
    return df_weights

    
    