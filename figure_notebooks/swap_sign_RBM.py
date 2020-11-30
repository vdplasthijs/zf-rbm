import numpy as np
import copy
def swap_sign_RBM(RBM, verbose=0):
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
    RBM2 = copy.deepcopy(RBM)
    RBM2.weights[change_sign] *= -1
    RBM2.hlayer.theta_plus[change_sign] = RBM.hlayer.theta_minus[change_sign]
    RBM2.hlayer.theta_minus[change_sign] = RBM.hlayer.theta_plus[change_sign]
    RBM2.hlayer.gamma_plus[change_sign] = RBM.hlayer.gamma_minus[change_sign]
    RBM2.hlayer.gamma_minus[change_sign] = RBM.hlayer.gamma_plus[change_sign]
    RBM2.hlayer.recompute_params(which='others')
    RBM2.fantasy_h *= -1
    return RBM2
