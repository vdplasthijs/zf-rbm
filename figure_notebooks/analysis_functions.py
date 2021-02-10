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
import scipy.spatial, scipy.cluster, scipy.sparse
from sklearn.mixture import GaussianMixture
import math, pickle, os, sys, gc
import swap_sign_RBM as ssrbm
sys.path.append('/home/thijs/repos/dnp-code/') # PGM3_correct/source/
from fishualizer_utilities import Zecording

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
        if len(duration_1) == 1 and len(duration_2) == 0:
            median_activity_period[mu, :] = duration_1[0]
        elif len(duration_1) == 0 and len(duration_2) == 0:
            median_activity_period[mu, :] = 0
        else:
            median_activity_period[mu, 0] = np.maximum(np.median(duration_1), np.median(duration_2))
            median_activity_period[mu, 1] = np.minimum(np.median(duration_1), np.median(duration_2))
        if np.isnan(median_activity_period[mu, 0]):
            print('ERROR: NaN found - BREAKING')
            return
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

def discretize(h, margin=0.25, plot=False, ax=None):
    '''discretize 1 HU into 3 intervals: mode 1 - no mans land - mode 2.
    Dependent on margin (0 -> ths on peaks, 0.5 -> ths both on middle)'''
    gmm = GaussianMixture(n_components=2).fit(h[:, np.newaxis])
    mus = gmm.means_
    order = np.argsort(mus[:, 0])
    mus = mus[order]
    threshold1 = mus[0] + (mus[1] - mus[0]) * margin
    threshold2 = mus[1] - (mus[1] - mus[0]) * margin
    if plot:
        if ax is None:
            ax = plt.subplot(111)
        ax.hist(h, bins=100, density=True, label='HU activity');
        ax.plot([threshold1, threshold1], [0, 5], c='red', linestyle=':',
                 label='lower threshold')
        ax.plot([threshold2, threshold2], [0, 5], c='red', label='upper threshold')
        ax.set_xlabel('HU activity'); ax.set_ylabel('PDF')
        ax.legend()

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


def kstest(datalist1, datalist2):
    '''An additional KS test:
    ( because the scipy.stats gives the same P value for all 3, for some reason)
    computing with this one gives the sam statisitcs, but lower P values which slightly difer.
    I therefore think its a rounding error so its fine.
    https://gist.github.com/devries/1140510
    '''
    n1 = len(datalist1)
    n2 = len(datalist2)
    datalist1.sort()
    datalist2.sort()

    j1 = 0
    j2 = 0
    d = 0.0
    fn1=0.0
    fn2=0.0
    while j1<n1 and j2<n2:
        d1 = datalist1[j1]
        d2 = datalist2[j2]
        if d1 <= d2:
            fn1 = (float(j1)+1.0)/float(n1)
            j1+=1
        if d2 <= d1:
            fn2 = (float(j2)+1.0)/float(n2)
            j2+=1
        dtemp = math.fabs(fn2-fn1)
        if dtemp>d:
            d=dtemp

    ne = float(n1*n2)/float(n1+n2)
    nesq = math.sqrt(ne)
    prob = ksprob((nesq+0.12+0.11/nesq)*d)
    return d,prob,ne

def ksprob(alam):
    '''An additional KS test:
    ( because the scipy.stats gives the same P value for all 3, for some reason)
    computing with this one gives the sam statisitcs, but lower P values which slightly difer.
    I therefore think its a rounding error so its fine.
    https://gist.github.com/devries/11405101
    '''
    fac = 2.0
    sum = 0.0
    termbf = 0.0

    a2 = -2.0*alam*alam
    for j in range(1,101):
        term = fac*math.exp(a2*j*j)
        sum += term
        if math.fabs(term) <= 0.001*termbf or math.fabs(term) <= 1.0e-8*sum:
            return sum
        fac = -fac
        termbf = math.fabs(term)

    return 1.0

def get_neural_data(dir_path, file_path):
    full_path = os.path.join(dir_path, file_path)
    rec = Zecording(path=full_path, kwargs={'ignorelags': True,
                                              'forceinterpolation': False,
                                              'ignoreunknowndata': True,# 'parent': self,
                                              'loadram': True})  # load data
    print(rec)
    regions = {#'rh1': np.array([218]), 'rhall': np.array([113]),
              'wb': np.arange(294)}
    selected_neurons = {}
    n_sel_cells = {}
#     train_data = {}
#     test_data = {}
    full_data = {}

#     dict_tt_inds = pickle.load(open(train_inds_path, 'rb'))  # load dictionary with training indices
#     train_inds = dict_tt_inds['train_inds']  # load training inds, note that: # test_inds = dict_tt_inds['test_inds']
#     test_inds = dict_tt_inds['test_inds']
#     print(f'len test inds {len(test_inds)}')
    for ir in list(regions.keys()):
        selected_neurons[ir] = np.unique(scipy.sparse.find(rec.labels[:, regions[ir]])[0])
        assert rec.spikes.shape[0] > rec.spikes.shape[1]
#         train_data[ir] = rec.spikes[selected_neurons[ir], :][:, train_inds]
#         test_data[ir] = rec.spikes[selected_neurons[ir], :][:, test_inds]
        n_sel_cells[ir] = len(selected_neurons[ir])
        full_data[ir] = rec.spikes[selected_neurons[ir], :]
    vu_data = full_data[ir].copy()
    rec = None
    full_data = None
    return vu_data

def get_demeaned_hu_dynamics_from_rbm_file(vu_data, rbm_path):
    # tmp_RBM = pickle.load(open(rbm_path, 'rb'))
    # RBM = ssrbm.swap_sign_RBM(RBM=tmp_RBM, verbose=2, assert_hu_inds=hu_assert)
    RBM = pickle.load(open(rbm_path, 'rb'))
    assert vu_data.shape[0] > vu_data.shape[1], 'expected more neurons than time points'
    hu_act = np.transpose(RBM.mean_hiddens(vu_data.T))
    # ol = af.opt_leaf(hu_act_test)
    # hu_act_test_remap = hu_act_test[ol, :]

    ## demean HU activity by using average between its 2 peaks (foudn by GMM)
    assert hu_act.shape[0] < hu_act.shape[1], 'expected more time points than HUs'
    hu_activity_effectively_demeaned = hu_act.copy()
    for mu in range(hu_act.shape[0]):
        gmm = GaussianMixture(n_components=2).fit(hu_act[mu, :, np.newaxis])
        two_peaks = gmm.means_[:2]
        effective_mean = two_peaks.mean()
        hu_activity_effectively_demeaned[mu, :] -= effective_mean
    hu_act = hu_activity_effectively_demeaned.copy()
    RBM = None
    return hu_act

def part_ratio_hu_activity(hu_activity, set_zero=True):
    ## PR activity
    n_hu, n_times = hu_activity.shape
    tmp_hu_act = hu_activity
    if set_zero:
        tmp_hu_act[tmp_hu_act <= 0] = 0
    pr_hus = np.zeros(n_times)
    for tt in range(n_times):
        pr_hus[tt] = p_metric_per_hu(tmp_hu_act[:, tt])
    return pr_hus.copy()

def get_all_prs_rbms(all_rbm_dir = '/media/thijs/hooghoudt/RBM_many_fishes_used_for_connectivity',
                     all_data_dir_list=['/media/thijs/hooghoudt/Zebrafish_data/spontaneous_data_guillaume',
                                        '/media/thijs/hooghoudt/Zebrafish_data/spontaneous_data_guillaume_T22']):
    all_rbm_list = [x for x in os.listdir(all_rbm_dir) if x[-5:] == '.data']
    n_rbms = len(all_rbm_list)
    pr_dict = {}
    data_name_dict = {k: [x for x in os.listdir(k) if x[-3:] == '.h5'] for k in all_data_dir_list}
    # print(data_name_dict)
    for i_rbm, rbm_name in enumerate(all_rbm_list):  # rbm names, see if they match with data names
        print(f'{i_rbm}/{len(all_rbm_list)}')
        gc.collect()  # explicit garbage collect is needed because otherwise it doesn't free memeroy of previous runs
        save_name = rbm_name[:-5]
        name_dataset = rbm_name[4:18]
        current_data_file_path = ''
        for data_dir, data_name_list in data_name_dict.items():
            assert len(data_name_list) == len(np.unique(data_name_list))
            for i_data_name, data_name in enumerate(data_name_list):
                if name_dataset == data_name[:14]:  # match
                    current_data_dir_path = data_dir
                    current_data_file_path = data_name
                    break
        if current_data_file_path == '':
            print('NO data match found for rbm', rbm_name)
            return
        vu_data = get_neural_data(dir_path=current_data_dir_path,
                                  file_path=current_data_file_path)
        hu_data = get_demeaned_hu_dynamics_from_rbm_file(vu_data=vu_data,
                                                         rbm_path=os.path.join(all_rbm_dir, rbm_name))
        pr_hus = part_ratio_hu_activity(hu_activity=hu_data, set_zero=True)
        pr_dict[rbm_name[:-5]] = pr_hus.copy()
        vu_data, hu_data = None, None
    return pr_dict

def n_cells_all_recordings(all_rbm_dir = '/media/thijs/hooghoudt/RBM_many_fishes_used_for_connectivity',
                     all_data_dir_list=['/media/thijs/hooghoudt/Zebrafish_data/spontaneous_data_guillaume',
                                        '/media/thijs/hooghoudt/Zebrafish_data/spontaneous_data_guillaume_T22']):

    all_rbm_list = [x for x in os.listdir(all_rbm_dir) if x[-5:] == '.data']
    n_rbms = len(all_rbm_list)
    data_name_dict = {k: [x for x in os.listdir(k) if x[-3:] == '.h5'] for k in all_data_dir_list}

    n_cells_dict = {}
    i_rbm = 0
    # for i_rbm, rbm_name in enumerate(all_rbm_list):  # rbm names, see if they match with data names
    list_full_paths_datasets = []
    while i_rbm < len(all_rbm_list):
        rbm_name = all_rbm_list[i_rbm]
        print(f'{i_rbm}/{len(all_rbm_list)}')
        save_name = rbm_name[:-5]
        name_dataset = rbm_name[4:18]
        current_data_file_path = ''
        for data_dir, data_name_list in data_name_dict.items():
            assert len(data_name_list) == len(np.unique(data_name_list))
            for i_data_name, data_name in enumerate(data_name_list):
                if name_dataset == data_name[:14]:  # match
                    full_path = os.path.join(data_dir, data_name)
                    if full_path not in list_full_paths_datasets:
                        list_full_paths_datasets.append(full_path)
                    i_rbm += 1

    # print(len(list_full_paths_datasets))
    assert len(list_full_paths_datasets) == 8, len(list_full_paths_datasets) == len(np.unique(list_full_paths_datasets))

    for i_dataset, data_path in enumerate(list_full_paths_datasets):
        print(i_dataset, '/', len(list_full_paths_datasets))
        gc.collect()  # explicit garbage collect is needed because otherwise it doesn't free memeroy of previous runs
        rec = Zecording(path=data_path, kwargs={'ignorelags': True,
                                                  'forceinterpolation': False,
                                                  'ignoreunknowndata': True,# 'parent': self,
                                                  'loadram': True})  # load data


        selected_neurons = np.unique(scipy.sparse.find(rec.labels[:, np.arange(294)])[0])
        assert rec.spikes.shape[0] > rec.spikes.shape[1]
        n_sel_cells = len(selected_neurons)
        short_data_name = data_path.split('/')[-1].rstrip('.h5')
        n_cells_dict[short_data_name] = n_sel_cells

    return n_cells_dict
