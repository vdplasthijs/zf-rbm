# @Author: Thijs L van der Plas <thijs>
# @Date:   2020-11-25
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: analysis_functions.py
# @Last modified by:   thijs
# @Last modified time: 2020-11-25

import numpy as np
from tqdm import tqdm
import pandas as pd

def count_connections(weight_matrix, K_arr = np.arange(11),
                      perc_top= 99.999, n_w_th = 250):
    '''Function to compute how uniform connections are. Computes for
    a varying threshold w, to how many (K) HUs each VU connects. For both
    weight_matrix and the shuffled one.

    Parameter:
    ---------------
        weight_matrix: 2D np array
            HUs x VUs
        K_arr: np array
            array of K values
        perc_top: float <= 100
            where to stop (to prevent extreme values taking over linspace #TODO could be fixed by taking intervals rather than linspace for w)
        n_w_th: int
            number of w data points (threhsold)
    Returns:
    ---------------
        (2D matrix with percentage (K x w); normalised per K, shuffled counterpart)
    '''
    n_hidden, n_sel_cells = weight_matrix.shape
    w_th_arr = np.linspace(0, np.percentile(np.abs(weight_matrix), 99.9), n_w_th)  # use 0
    shuffled_weights = weight_matrix.reshape(n_sel_cells * n_hidden)  # reshape
    shuffled_inds = np.arange(n_sel_cells * n_hidden)  # create new idns
    np.random.shuffle(shuffled_inds)  # shuffle
    shuffled_weights = shuffled_weights[shuffled_inds]  # resample
    shuffled_weights = shuffled_weights.reshape(n_hidden, n_sel_cells)  # reshape

    curve_NVUgeqKw = np.zeros((len(K_arr), n_w_th)) #{K: np.zeros_like(w_th_arr[dr]) for K in K_arr} # number of vu with geq K connections greater than threshold w
    sh_curve_NVUgeqKw = np.zeros((len(K_arr), n_w_th))   # number of shuffled vu with geq K connections greater than threshold w

    for iloop, w_th in tqdm(enumerate(w_th_arr)):  # loop through thresholds
        for iK, K in enumerate(K_arr): # loop through number of connecitons
            wmat_th = np.abs(weight_matrix) >= w_th  # bool of weights geq than
            sh_wmat_th = np.abs(shuffled_weights) >= w_th  # shuffled eq
            wmat_th_perVU = np.sum(wmat_th, axis=0)  # sum over hidden -> to how many HU do VUs connect with geq threshold
            sh_wmat_th_perVU = np.sum(sh_wmat_th, axis=0)
            curve_NVUgeqKw[iK, iloop] = np.sum(wmat_th_perVU >= K) / float(n_sel_cells) * 100.0  # count if geq K HUs
            sh_curve_NVUgeqKw[iK, iloop] = np.sum(sh_wmat_th_perVU >= K) / float(n_sel_cells) * 100.0
    return (w_th_arr, curve_NVUgeqKw, sh_curve_NVUgeqKw)


def count_connections_2(weight_matrix, K_arr = np.arange(11),
                        n_w_th = 250):
    '''Function to compute how uniform connections are. Computes the cumulative pdf of
    w for the Kth largest weight. For both
    weight_matrix and the shuffled one.

    Parameter:
    ---------------
        weight_matrix: 2D np array
            HUs x VUs
        K_arr: np array
            array of K values
        perc_top: float <= 100
            where to stop (to prevent extreme values taking over linspace #TODO could be fixed by taking intervals rather than linspace for w)
        n_w_th: int
            number of w data points (threhsold)
    Returns:
    ---------------
        (2D matrix with percentage (K x w); normalised per K, shuffled counterpart)
    '''
    n_hidden, n_sel_cells = weight_matrix.shape
    # w_th_arr = np.linspace(0, np.percentile(np.abs(weight_matrix), 99.9), n_w_th)  # use 0
    shuffled_weights = weight_matrix.reshape(n_sel_cells * n_hidden)  # reshape
    shuffled_inds = np.arange(n_sel_cells * n_hidden)  # create new idns
    np.random.shuffle(shuffled_inds)  # shuffle
    shuffled_weights = shuffled_weights[shuffled_inds]  # resample
    shuffled_weights = shuffled_weights.reshape(n_hidden, n_sel_cells)  # reshape

    all_w_arrs = np.zeros((len(K_arr), n_w_th + 1))
    all_sh_w_arrs = np.zeros((len(K_arr), n_w_th + 1))
    all_cdfs = np.zeros((len(K_arr), n_w_th)) #{K: np.zeros_like(w_th_arr[dr]) for K in K_arr} # number of vu with geq K connections greater than threshold w
    all_sh_cdfs = np.zeros((len(K_arr), n_w_th))   # number of shuffled vu with geq K connections greater than threshold w

    for iK, K in tqdm(enumerate(K_arr)): # loop through number of connecitons
        # ## To find largest (postive) element:
        # wmat_sorted = np.sort(weight_matrix, 0)  # sort weights per VU small to large
        # wmat_klargest = wmat_sorted[-1 - K, :]  # get K largest (incl zero indexing )
        ## To find largest positive or negative element (e.g. -7 > 5):
        wmat_sorted_inds = np.argsort(np.abs(weight_matrix), 0)  # sort by absolute to get both pos and neg
        wmat_klargest = np.array([weight_matrix[wmat_sorted_inds[-1 - K, x], x] for x in range(n_sel_cells)])  # get corresponding element

        sh_wmat_sorted_inds = np.argsort(np.abs(shuffled_weights), 0)  # sort by absolute to get both pos and neg
        sh_wmat_klargest = np.array([shuffled_weights[sh_wmat_sorted_inds[-1 - K, x], x] for x in range(n_sel_cells)])  # get corresponding element

        common_w_bins = np.linspace(np.minimum(wmat_klargest.min(), sh_wmat_klargest.min()),
                                    np.maximum(wmat_klargest.max(), sh_wmat_klargest.max()),
                                    n_w_th + 1)  # +1 for bin edges
        assert wmat_klargest.ndim == 1
        hist_wmat, w_bin_edges = np.histogram(wmat_klargest, bins=common_w_bins)
        cdf = np.cumsum(hist_wmat)
        assert cdf[-1] == n_sel_cells
        cdf = cdf / cdf[-1]
        all_cdfs[iK, :] = cdf
        all_w_arrs[iK, :] = w_bin_edges

        assert wmat_klargest.ndim == 1
        sh_hist_wmat, sh_w_bin_edges = np.histogram(sh_wmat_klargest, bins=common_w_bins)
        sh_cdf = np.cumsum(sh_hist_wmat)
        assert sh_cdf[-1] == n_sel_cells
        sh_cdf = sh_cdf / sh_cdf[-1]
        all_sh_cdfs[iK, :] = sh_cdf
        all_sh_w_arrs[iK, :] = sh_w_bin_edges
    assert (all_w_arrs == all_sh_w_arrs).all() # use same weights for shuffled and normal
    return (all_w_arrs, all_sh_w_arrs, all_cdfs, all_sh_cdfs)


def freq_distr_weighted_regions(w_vector, m_labels):
    w_vector = np.abs(w_vector)
    weighted_labels = np.dot(w_vector, m_labels)
    weighted_labels = weighted_labels / m_labels.sum(0)
    weighted_labels = weighted_labels / np.sum(weighted_labels)
    return weighted_labels

def freq_entropy(prob_distr):
    prob_distr = prob_distr / np.sum(prob_distr)
    prob_distr += 1e-15  # to solve zero issue
    log_p = np.log2(prob_distr)
    tmp = prob_distr * log_p
    entropy = -1 * np.sum(tmp)
    return entropy

def p_metric_per_hu(w_vector):
    w_vector = np.squeeze(w_vector)
#     assert w_vector.dim == 1
    p = np.power(np.sum(np.power(w_vector, 2)), 2) / np.sum(np.power(w_vector, 4))
    p = p / len(w_vector)
    return np.squeeze(p)

def bin_stats(stat_r, stat_g, n_bins=100):
    result = pf.pearsonr_ci(stat_r, stat_g)
    r_bins = np.linspace(np.percentile(stat_r, 1), np.percentile(stat_r, 99), n_bins + 1)
    mean_g_bins = np.zeros(n_bins)
    std_g_bins = np.zeros(n_bins)
    for i in range(n_bins):
        inds = np.logical_and((stat_r <= r_bins[i+1]), (stat_r > r_bins[i]))
        mean_g_bins[i] = np.mean(stat_g[inds])
        std_g_bins[i] = np.std(stat_g[inds])
    plot_bins = 0.5 * (r_bins[1:] + r_bins[:-1])
    return result, (plot_bins, mean_g_bins, std_g_bins)

def correct_stat(s_old, s_naive, s_opt):
    s_corr = (s_old - s_naive) / (s_opt - s_naive)
    return s_corr

def create_normalised_df(df, stats_bin, tt_stats,
                         exception_ts=['2020-04-27-2326']):
    ts_list_df = list(df['timestamp'])
    ts_list_dict = [x for x in stats_bin.keys() if x != 'settings']
    assert (np.array(ts_list_df) == np.array(ts_list_dict)).all()
    ts_list_use = [x for x in ts_list_df if x not in exception_ts]

    df_columns = ['nhu', 'l1', 'means_v', 'means_h', 'pwcorr_vh_corr',
                  'pwcorr_vv_corr', 'pwcorr_hh_corr', 'reconstruct']
    new_df = pd.DataFrame({**{'timestamp': ts_list_use},
                           **{x: np.zeros(len(ts_list_use)) for x in df_columns}})
    for i_ts, ts in enumerate(ts_list_use):
        curr_entry = new_df[new_df['timestamp'] == ts].index[0]
        assert (new_df['timestamp'].iat[curr_entry] == ts)
        old_df_entry = df[df['timestamp'] == ts].index[0]
        assert (df['timestamp'].iat[old_df_entry] == ts)
        new_df['nhu'].iat[curr_entry] = df['nhu'].iat[old_df_entry]
        new_df['l1'].iat[curr_entry] = df['l1'].iat[old_df_entry]
        for stat in ['means_v', 'pwcorr_vv_corr']:
            tmp_stat_old = stats_bin[ts][stat]['rmse']
            tmp_stat_naive = stats_bin[ts][stat]['rmse_shuffled']
#             tmp_stat_naive = np.sqrt(tt_stats[stat + '_shuffled']['sse'] / tt_stats[stat + '_shuffled']['n_elements'])
            tmp_stat_opt = np.sqrt(tt_stats[stat]['sse'] / tt_stats[stat]['n_elements'])
            tmp_stat_corr = correct_stat(s_old=tmp_stat_old, s_naive=tmp_stat_naive,
                                         s_opt=tmp_stat_opt)
            new_df[stat].iat[curr_entry] = tmp_stat_corr

        for stat in ['means_h', 'pwcorr_vh_corr', 'pwcorr_hh_corr']:
            tmp_stat_old = stats_bin[ts][stat]['rmse']
#             tmp_stat_naive = stats_bin[ts][stat]['traindata_rmse_shuffled']
            tmp_stat_naive = stats_bin[ts][stat]['rmse_shuffled']
            tmp_stat_opt = stats_bin[ts][stat]['traindata_rmse']
            tmp_stat_corr = correct_stat(s_old=tmp_stat_old, s_naive=tmp_stat_naive,
                                         s_opt=tmp_stat_opt)
            new_df[stat].iat[curr_entry] = tmp_stat_corr


        new_df['reconstruct'].iat[curr_entry] = df['reconstruct_med'].iat[old_df_entry]
    return new_df
