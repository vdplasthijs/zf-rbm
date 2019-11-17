import h5py
import numpy as np
import pandas as pd
import sys, os
import scipy
import tqdm, datetime
import pickle

curr_dir = os.getcwd()
try:
    os.chdir('/home/thijs/Desktop/RBM/test_profiler/rbm7/')
except FileNotFoundError:
    os.chdir('/home/thijs/Desktop/RBM/For_Thijs/PGM7/')
import utilities7 as utilities
os.chdir(curr_dir)



## Load RBM generate data
def load_gen_data(gpath, verbose=True):
    hfile_gen = h5py.File(gpath, 'r')
    generated_visible = hfile_gen['Data']['dff'].value
    if verbose:
        print(f'{hfile_gen} with shape {generated_visible.shape}.')

    # transpose depending on 1 or multiple chains:
    if len(generated_visible.shape) == 2:
        generated_visible = np.transpose(generated_visible) # units, times
    elif len(generated_visible.shape) == 3:
        generated_visible = np.transpose(generated_visible, (0, 2, 1))  # chains, units, times
    if verbose:
        print(generated_visible.shape)

    try: # try to load hidden data
        generated_hidden = hfile_gen['Data']['hidden_data'].value
        if len(generated_hidden.shape) == 2:
            generated_hidden = np.transpose(generated_hidden) # units, times
        elif len(generated_hidden.shape) == 3:
            generated_hidden = np.transpose(generated_hidden, (0, 2, 1))  # chains, units, times
        if verbose:
            print(f'hidden data loaded with shape {generated_hidden.shape}.')
    except:
        if verbose:
            print('No hidden data (under name hidden_data) in generated file')
#         tmp_new_gh = mapping['rbm'](generated_visible)

    hfile_gen.close()
    return generated_visible, generated_hidden

# generated_data_path = '/home/thijs/Desktop/RBM/For_Thijs/rhomb1_improvedtraining/20180912Run01_rhomb_all_spont_train4k_nit45k_nh100_l1_8e-03_simulated_NC1_LC4000_beta1_Nstep20_NPT1_improvtrain.h5'
# generated_visible, generated_hidden = load_gen_data(gen_files_links[70], verbose=True)

def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy

    source: https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = scipy.stats.pearsonr(x,y)  # pearson correlation coef & p value
    r_z = np.arctanh(r)  # transform to z score
    se = 1/np.sqrt(x.size-3.0)  # standard deviation
    z = scipy.stats.norm.ppf(1.0-alpha/2.0)  # calculate z statistic (for confidence interval alpha)
    lo_z, hi_z = r_z - z * se, r_z + z * se  # make CI
    lo, hi = np.tanh((lo_z, hi_z))  # transform back
    return r, p, lo, hi

def double_nan_check(x, y, verbose=False):
    assert x.shape == y.shape
    assert len(x.shape) == 1 # currently only implemented for 1d
    i1 = set(np.where(~np.isnan(x))[0])
    i2 = set(np.where(~np.isnan(y))[0])
    i3 = np.array(list(i1.intersection(i2)))
    if verbose:
        print('1st data set: {} /{} nans'.format(len(i1), len(x)))
        print('2st data set: {} /{} nans'.format(len(i2), len(y)))
        print('merged data set: {} /{} nans'.format(len(i3), len(y)))
    return i3

def compute_moment_corrs(rv, rh, gv, gh, moment='means_v', mompearson2nd='moment', nv_tresh=10, return_binned_stats=False):
    """r real, g generated, v visible, h hidden
    shapes (N, T) and (M, T)

    moment: str, means_v, means_h, pwcorr_vh, pwcorr_vv, pwcorr_hh
    mompearson2nd: moment, pearson
    """
#     assert rv.shape[1] == rh.shape[1] and rv.shape[1] == gv.shape[1] and rv.shape[1] == gh.shape[1]
    nv = rv.shape[0]
    nh = rh.shape[0]

    # calculating
    if moment == 'means_v':
        n_bins = 100
        stat_r = np.mean(rv, axis=1)
        stat_g = np.mean(gv, axis=1)

    elif moment == 'means_h':
        n_bins = 20
        stat_r = np.mean(rh, axis=1)
        stat_g = np.mean(gh, axis=1)

    elif moment == 'pwcorr_vh':
        n_bins = 200
        stat_r = np.ravel(utilities.average_product(gh.transpose(), gv.transpose()))
        stat_g = np.ravel(utilities.average_product(rh.transpose(), rv.transpose()))

    elif moment == 'pwcorr_vv':
        n_bins = 200
        if nv > nv_tresh:
            print(f'n cells = {nv}, too many, subsampling {nv_tresh}')
            inds = np.random.choice(a=nv, size=nv_tresh, replace=False)
            rv = rv[inds, :]
            gv = gv[inds, :]
            del inds
            nv = nv_tresh
        vv_inds_64 = np.triu_indices(n=nv, k=1)
        vv_inds = (vv_inds_64[0].astype('uint16'), vv_inds_64[1].astype('uint16'))
        del vv_inds_64
        if mompearson2nd == 'pearson':
            stat_r = np.corrcoef(rv)
            stat_r = stat_r[vv_inds]
            stat_g = np.corrcoef(gv)
        elif mompearson2nd == 'moment':
            stat_r = np.dot(rv,np.asarray(rv.transpose(),dtype='float32'))/rv.shape[1]
            stat_r = stat_r[vv_inds]
            stat_g = np.dot(gv,np.asarray(gv.transpose(),dtype='float32'))/gv.shape[1]
        stat_g = stat_g[vv_inds]
        del vv_inds
    elif moment == 'pwcorr_hh':
        n_bins = 50
        if mompearson2nd == 'pearson':
            stat_r = np.corrcoef(rh)
            stat_g = np.corrcoef(gh)
        elif mompearson2nd == 'moment':
            stat_r = utilities.average_product(rh.transpose(), rh.transpose())
            stat_g = utilities.average_product(gh.transpose(), gh.transpose())
        hh_inds = np.triu_indices(n=nh, k=1)
        stat_r = stat_r[hh_inds]
        stat_g = stat_g[hh_inds]
    elif moment == 'twcorr_hhh':
        n_bins = 100
        corr_mats = thirdorder(real=rh, gen=gh)
        stat_r = np.ravel(corr_mats['real'])
        stat_g = np.ravel(corr_mats['gen'])



    # ind_non_nan = double_nan_check(stat_r, stat_g)
    if np.isnan(stat_r).sum() == 0 and np.isnan(stat_g).sum() == 0:
        # print('no nans present')
        result =  pearsonr_ci(stat_r, stat_g)
    else:
        print(f'NANs found: in stat_r { np.isnan(stat_r).sum()} nans, in stat_g { np.isnan(stat_g).sum()} nans, breaking')
        result = [np.nan, np.nan]
    if return_binned_stats is False:
        del stat_r
        del stat_g
        return result
    elif return_binned_stats:
        r_bins = np.linspace(np.percentile(stat_r, 1), np.percentile(stat_r, 99), n_bins + 1)
        mean_g_bins = np.zeros(n_bins)
        std_g_bins = np.zeros(n_bins)
        for i in range(n_bins):
            inds = np.logical_and((stat_r <= r_bins[i+1]), (stat_r > r_bins[i]))
            mean_g_bins[i] = np.mean(stat_g[inds])
            std_g_bins[i] = np.std(stat_g[inds])
        plot_bins = 0.5 * (r_bins[1:] + r_bins[:-1])
        del stat_r
        del stat_g
        return result, (plot_bins, mean_g_bins, std_g_bins)

def thirdorder(real, gen):
    ## 3rd order moments
    assert real.shape[0] == gen.shape[0] and real.shape[0] < real.shape[1]
    rnorm = real - np.mean(real, axis=1)[:, np.newaxis]
    gnorm = gen - np.mean(gen, axis=1)[:, np.newaxis]
    NN = rnorm.shape[0]
    normdata = {'real': rnorm, 'gen': gnorm}
    C = {'real': np.ones((NN, NN ,NN)), 'gen': np.ones((NN, NN ,NN))}
    for dd in ['real', 'gen']:
#         print dd
        for i in range(NN):
            for j in range(NN):
                for k in range(NN):
                    C[dd][i, j, k] = np.mean(np.multiply(np.multiply(normdata[dd][i, :],
                                                                     normdata[dd][j, :]),
                                                         normdata[dd][k, :]))
    return C


def llh1d(real_trace, pred_trace):
    tmp = False
    real_trace = np.squeeze(real_trace)
    real_trace = real_trace.astype('float32')
    pred_trace = np.squeeze(pred_trace)
    assert len(real_trace) == len(pred_trace)
    ## assert real_trace is only 0s and 1s
    #     llho = np.sum(np.log(np.multiply(real_trace, (2*pred_trace - 1)) + np.ones_like(pred_trace) - pred_trace))
    llh = np.sum(np.log(np.clip(np.multiply(real_trace, pred_trace) +
                                np.multiply((1 - real_trace), (1 - pred_trace)),
                                a_min=0.0000001, a_max=np.inf)))  # clip for pca < 0 (and log)

    return llh, tmp
