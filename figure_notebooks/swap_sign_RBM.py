import numpy as np
import copy
import h5py
import sys,os
import pandas as pd
import scipy.sparse
import pickle
sys.path.append('/home/thijs/repos/dnp-code/PGM3_correct/source/') # the path where the folder PGM is.
sys.path.append('/home/thijs/repos/dnp-code/PGM3_correct/utilities/') # the path where the folder PGM is.
import rbm as rbm
from fishualizer_utilities import Zecording

def swap_sign_RBM(RBM, verbose=0, assert_hu_inds=None):
    '''
    Mostly for neuroscience related applications.
    Excitatory and inhibitory couplings between pairs of neurons are determined by
    the weighted dot product their weights vectors 0.5 \sum_\mu w_{imu} w_{jmu} V_\mu.
    Therefore, inhibitory connections occur between neurons with weights of opposite sign.
    To identify them more easily, swap the sign such that most weights are positive.
    The marginal P(v) is invariant upon change of sign of the weights, and corresponding
    gauge transformation on the hidden layer potential parameters.
    '''
    sign_weights = np.sign( RBM.weights.sum(1) )
    change_sign = (sign_weights<0)
    if verbose > 0:
        print(f'{np.sum(change_sign)}/{len(change_sign)} HU weights are flipped')
    if verbose > 1:
        print(f'Flipped HUs are: {np.where(change_sign)}')
    if assert_hu_inds is not None:  # can be used to assert which HUs should be swapped
        assert (np.where(change_sign)[0] == np.squeeze(np.array(assert_hu_inds))).all(), 'swapped HUs do not correspond to assert_hu_inds input'
    RBM2 = copy.deepcopy(RBM)
    RBM2.weights[change_sign] *= -1
    RBM2.hlayer.theta_plus[change_sign] = RBM.hlayer.theta_minus[change_sign]
    RBM2.hlayer.theta_minus[change_sign] = RBM.hlayer.theta_plus[change_sign]
    RBM2.hlayer.gamma_plus[change_sign] = RBM.hlayer.gamma_minus[change_sign]
    RBM2.hlayer.gamma_minus[change_sign] = RBM.hlayer.gamma_plus[change_sign]
    RBM2.hlayer.recompute_params(which='others')
    RBM2.fantasy_h *= -1
    return RBM2

def save_swapped_weights(rbm_dir='/media/thijs/hooghoudt/new_sweep_april20/RBM_sweep_reruns/',
                         rbm_name='RBM3_20180912-Run01-spontaneous-rbm2_wb_test-segs-267-nseg10_M200_l1-2e-02_duration208093s_timestamp2020-05-16-0844.data',
                         recording_path='/media/thijs/hooghoudt/Zebrafish_data/spontaneous_data_guillaume/20180912_Run01_spontaneous_rbm2.h5'):
    rbm_path = os.path.join(rbm_dir, rbm_name)
    tmp_RBM = pickle.load(open(rbm_path, 'rb'))
    RBM = swap_sign_RBM(RBM=tmp_RBM, verbose=1)
    new_weights_name = rbm_dir + 'weights/' + 'weights_' + rbm_name[:-5] + '_signswapped.h5'

    rec = Zecording(path=recording_path, kwargs={'ignorelags': True,
                                              'forceinterpolation': False,
                                              'ignoreunknowndata': True,# 'parent': self,
                                              'loadram': True})  # load data

    regions = {'wb': np.arange(294)}
    n_cells = rec.n_cells
    assert n_cells == rec.labels.shape[0]
    selected_neurons = np.unique(scipy.sparse.find(rec.labels[:, regions['wb']])[0])  # cells with zbrain label
    print(f'n cells: {n_cells}, n labelled cells: {len(selected_neurons)}')

    weights = np.transpose(RBM.weights)
    assert weights.shape[0] > weights.shape[1]
    assert weights.shape[0] == len(selected_neurons)  # RBM has only used labelled neurons
    full_weights = np.zeros((n_cells, weights.shape[1]))
    full_weights[selected_neurons, :] = weights.copy()  # let non-labelled neurons have w=0 for all HU
    df_weights = pd.DataFrame({'hu_' + str(ii).zfill(3):
                                np.squeeze(full_weights[:, ii]) for ii in range(full_weights.shape[1])})  # save as pd df with each column = one weight vector
    df_weights.to_hdf(new_weights_name, key='all')
