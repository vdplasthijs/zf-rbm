# @Author: Thijs L van der Plas <thijs>
# @Date:   2020-11-25
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: analysis_functions.py
# @Last modified by:   thijs
# @Last modified time: 2020-11-25

import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial, scipy.cluster
from sklearn.mixture import GaussianMixture

def opt_leaf(w_mat, dim=0):
    '''create optimal leaf order over dim, of matrix w_mat. if w_mat is not an
    np.array then its assumed to be a RNN layer. see also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering'''
    if type(w_mat) != np.ndarray:  # assume it's an rnn layer
        w_mat = [x for x in w_mat.parameters()][0].detach().numpy()
    assert w_mat.ndim == 2
    if dim == 1:  # transpose to get right dim in shape
        w_mat = w_mat.T
    dist = scipy.spatial.distance.pdist(w_mat, metric='euclidean')  # distanc ematrix
    link_mat = scipy.cluster.hierarchy.ward(dist)  # linkage matrix
    opt_leaves = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
    return opt_leaves

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


def count_connections_3(weight_matrix):
    n_hidden, n_sel_cells = weight_matrix.shape
    assert n_hidden < n_sel_cells

    shuffled_weights = weight_matrix.reshape(n_sel_cells * n_hidden)  # reshape
    shuffled_inds = np.arange(n_sel_cells * n_hidden)  # create new idns
    np.random.shuffle(shuffled_inds)  # shuffle
    shuffled_weights = shuffled_weights[shuffled_inds]  # resample
    shuffled_weights = shuffled_weights.reshape(n_hidden, n_sel_cells)  # reshape

    # degree_vu = np.sqrt(np.sum(np.abs(weight_matrix) ** 2, 0))
    assert 100 * (1 - 1 / n_hidden) ==  99.5, 100 - 1 / n_hidden
    threshold = np.percentile(np.abs(weight_matrix), 99.5)  # 99.5 = 100 - 1/n_hu
    degree_vu = np.sum(np.abs(weight_matrix) > threshold, 0)
    assert len(degree_vu) == n_sel_cells and degree_vu.ndim == 1
    # degree_vu_sh = np.sqrt(np.sum(np.abs(shuffled_weights) ** 2, 0))
    degree_vu_sh = np.sum(np.abs(shuffled_weights) > threshold, 0)

    return (degree_vu, degree_vu_sh)

def freq_distr_weighted_regions(w_vector, m_labels):
    w_vector = np.abs(w_vector)
    weighted_labels = np.dot(w_vector, m_labels)  # weighted product of region labels
    nz_inds = np.where(m_labels.sum(0) != 0)[0]  # avoid zero-division
    weighted_labels[nz_inds] = weighted_labels[nz_inds] / m_labels[:, nz_inds].sum(0)
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

def load_reprod_matrix(path, swap=False):
    with open(path, 'r') as f:
        content = f.readlines()
    dict_matrices = {}
    for line in content:
        if line[:7] == 'Method:':
            current_meth = line[8:].rstrip()
            current_meth = current_meth.replace('=', '')  # extract name
            dict_matrices[current_meth] = {'run_names': [], 'raw_mat': []}
        elif line[:3] == 'Run':
            dict_matrices[current_meth]['run_names'].append(line[7:].rstrip())
        else:
            dict_matrices[current_meth]['raw_mat'].append(line.rstrip())
    for meth in dict_matrices.keys():
        dict_matrices[meth]['pearson'] = np.zeros((len(dict_matrices[meth]['raw_mat']) - 1,
                                               len(dict_matrices[meth]['raw_mat']) - 1))
        for i_line, line in enumerate(dict_matrices[meth]['raw_mat']):
            if i_line == 0:
                continue  # skip first one because its just indices
            list_el = line.split()
            list_el = [float(x) for x in list_el[1:]]  # skip first becaues its an index
            dict_matrices[meth]['pearson'][i_line - 1, :] = np.array(list_el)  # -1 because of i_line > 0

    if swap:
        dict_matrices['RBM'], dict_matrices['covariance'] = dict_matrices['covariance'], dict_matrices['RBM']
    return dict_matrices


def compute_median_state_occupancy(activity, bimodality=0, freq=1):
    n_units, n_times = activity.shape
    median_activity_period = np.zeros((n_units, 2))

    for mu in range(n_units):
        tmp = np.sign(activity[mu, :] - bimodality)  # +1 or -1, as bimodality is around 0
        tmp_inds = np.where(tmp[1:] - tmp[:-1])[0] # where non zero? = state change

        duration_1 = (tmp_inds[1::2] - tmp_inds[:-1:2]) / freq  # skip 1 to only get 1 state type.
        duration_2 = (tmp_inds[2::2] - tmp_inds[1:-1:2]) / freq # get the other type

        median_activity_period[mu, 0] = np.maximum(np.median(duration_1), np.median(duration_2))
        median_activity_period[mu, 1] = np.minimum(np.median(duration_1), np.median(duration_2))
    return median_activity_period

def create_mapping_kunstea_order(current_regions):

    file_translation = '/home/thijs/repos/zf-rbm/baier_atlas_labels/region_names_baier_abbreviations.txt'
    file_order = '/home/thijs/repos/zf-rbm/baier_atlas_labels/order_baier_regions_kunstetal.txt'

    dict_long_to_short = {}
    dict_short_to_long = {}
    with open(file_translation, 'r') as f:
        content = f.readlines()
    for line in content:
        split_line = [x.rstrip() for x in line.split(' ')]
        assert len(split_line) == 2, split_line
        long, short = split_line
        dict_long_to_short[long] = short
        dict_short_to_long[short] = long

    with open(file_order, 'r') as f:
        array_order = np.array([x.rstrip() for x in f.readlines()])

    new_inds = np.zeros(len(current_regions))
    for i_reg, reg in enumerate(current_regions):
        # print( np.where(array_order == dict_long_to_short[reg]), dict_long_to_short[reg])
        new_inds[i_reg] = int(np.where(array_order == dict_long_to_short[reg])[0][0])
    new_inds = new_inds.astype('int')
    return new_inds, array_order

def discretize(h, margin=0.25, plot=False):
    '''discretize 1 HU into 3 intervals: mode 1 - no mans land - mode 2. 
    Dependent on margin (0 -> ths on peaks, 0.5 -> ths both on middle)'''
    gmm = GaussianMixture(n_components=2).fit(h[:, np.newaxis])
    mus = gmm.means_
    order = np.argsort(mus[:, 0])
    mus = mus[order]
    threshold1 = mus[0] + (mus[1] - mus[0]) * margin
    threshold2 = mus[1] - (mus[1] - mus[0]) * margin
    if plot:
        plt.hist(h, bins=100, normed=True, label='HU activity');
        plt.plot([threshold1, threshold1], [0, 5], c='red', linestyle=':',
                 label='lower threshold')
        plt.plot([threshold2, threshold2], [0, 5], c='red', label='upper threshold')
        plt.xlabel('HU activity'); plt.ylabel('PDF')
        plt.legend()
        
    return 1 * (h > threshold1) + 1 * (h > threshold2)  # discretise to [0, 1, 2] = [<= th1, >th1 & <= th2, > th2]


def get_burst_and_silence_times(sequence):
    '''
    Input:  a discrete time series occupying 3 states {0, 1, 2}
    Outputs: 
    - a list of burst durations
    - a list of silence durations.
    - a list of period durations (= silence + burst)
    '''    
    assert np.array([x in [0, 1, 2] for x in np.unique(sequence)]).all(), 'input sequence is not discretised'
   
    list_burst_intervals = []
    list_silence_intervals = []
    T = len(sequence)

    interval_is_silence = True
    interval_start = 0
    interval_end = 0

    while interval_start < T-1:
        interval_end = interval_start  # reset 
        if interval_is_silence:  # if in down state, loop while in down state
            while (sequence[interval_end] < 2) & (interval_end < T - 1): 
                interval_end += 1  # increase lenght of down state interval
        else:  # if in up state, loop while in up state
            while (sequence[interval_end] > 0) & (interval_end < T - 1):
                interval_end += 1
        ## end of previous interval is reached        
        
        if interval_is_silence:  # if previous interval was down
            if (interval_end < T - 1) & (interval_start > 0):  # if not first and not end          
                list_silence_intervals.append((interval_start, interval_end))  # add to down list
            interval_is_silence = False  # switch
        else:  # reverse 
            if (interval_end < T - 1) & (interval_start > 0):            
                list_burst_intervals.append((interval_start, interval_end))
            interval_is_silence = True

        interval_start = interval_end  # se tup next interval
        
    list_silence_durations = [end - start for start, end in list_silence_intervals]  # calculate durations
    list_burst_durations = [end - start for start, end in list_burst_intervals]
    list_period_durations = [list_silence_durations[x] + list_burst_durations[x] 
                             for x in range(np.minimum(len(list_silence_durations), 
                                                       len(list_burst_durations)))]
    assert np.abs(len(list_silence_durations) - len(list_burst_durations)) <= 1
    return list_silence_durations, list_burst_durations, list_period_durations

def compute_median_discretised_state_occupancy(activity_mat, frequency=1, margin=0.4, verbose=1):
    '''Compute state occupancy stats by discretising first'''
    
    n_hu = activity_mat.shape[0]
    if verbose > 0:
        print(f'Number of units: {n_hu}')
    median_silence_duration = np.zeros(n_hu)
    median_burst_duration = np.zeros(n_hu)
    median_period_duration = np.zeros(n_hu)
    count_bursts = np.zeros(n_hu)

    for mu in range(n_hu):
        discrete_h = discretize(activity_mat[mu, :], margin=margin, plot=False)  # discretised signal to {0, 1, 2}
        if verbose > 1:
            print('Unit %s, P(h=-1)=%.3f,P(h=0)=%.3f,P(h=+1)=%.3f'% (
                mu, 
                (discrete_h==0).mean(),
                (discrete_h==1).mean(),
                (discrete_h==2).mean()  )
                 )  # print average occupancy rates of the 3 states 

        list_silence_durations, list_burst_durations, list_period_durations = get_burst_and_silence_times(discrete_h)
        median_silence_duration[mu] = np.median(list_silence_durations / frequency)
        median_burst_duration[mu] = np.median(list_burst_durations / frequency)
        median_period_duration[mu] = np.median(list_period_durations / frequency)
        count_bursts[mu] = len(list_burst_durations)

    return median_silence_duration, median_burst_duration, median_period_duration, count_bursts

