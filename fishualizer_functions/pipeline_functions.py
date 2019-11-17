## Functions for pipeline


import xarray as xr
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors.kde import KernelDensity
import scipy
from scipy.interpolate import interp1d
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm_notebook as tqdm

def add_supp_data(s_name, s_data):
    # function to add supp data to rec
    setattr(rec, s_name, s_data)
    rec.available_data.add(s_name)
    rec.single_data.add(s_name)

def exp_smooth(trace, time_constant = 2.6):
    """Exponential smoothing

    Defined inside add_supplementary_traces()

    Parameters
    ----------
        trace: np.array,
            trace to smooth
        time_constant: float, optional (=2.6 default, GCaMP6s line of Migault et al., submitted)

    Returns
    ----------
        conv_trace:, float
            convolved trace
    """
    alpha_test = (1 - np.exp(-1/ time_constant)) # exponential decay time constant from Migault et al., 2018, biorxiv paper supp. methods (vestibular)
    k = lambda tau: alpha_test*(1-alpha_test)**tau
    k_len = len(trace)//10
    kernel = np.hstack((np.zeros(k_len), k(np.arange(k_len))))
    conv_trace = np.convolve(trace, kernel, mode='same') / np.sum(kernel)
    return conv_trace

def create_onset_offset_pairs_step(recording=None, stimulus=None, abs_tol=0.1, n_stimuli_per_trial = 6, stimtype='step'):
    """
    Parameters:
    --------------
        recording: Zecording class
        stimulus: data of stimulus or None (then rec.stimulus is selected)
        n_stimuli_per_trial, int
            number of stimuli (both positive and negative) per trial
        stimtype: str ('step' or 'sine')
            state stimulus type
    Returns:
    -------------
        stimulus_pairs:
            list of tuples of (onset, offset)
    """
    onset = []
    offset = []

    if stimulus is None:
        stimulus = recording.stimulus

    if stimtype == 'step':
        # non_zero_stim = np.where(stimulus !=  0)[0] # non zero elements
        zero_stim = np.isclose(stimulus, 0, atol=abs_tol)
        non_zero_stim = np.where(np.logical_not(zero_stim))[0]
        for t in range(1, len(stimulus)): # find boundaries of stimulus
            if t in non_zero_stim and t-1 not in non_zero_stim:
                onset.append(t-1)
            if t-1 in non_zero_stim and t not in non_zero_stim:
                offset.append(t)
    elif stimtype == 'sine':
        local_minima = np.r_[True, stimulus[1:] < stimulus[:-1]] & np.r_[stimulus[:-1] < stimulus[1:], True]
        onset = np.where(local_minima[:-1])[0] + 1
        offset = np.where(local_minima[1:])[0]

    onset_trial = onset[::n_stimuli_per_trial] # take every nth, starting at 1
    offset_trial = offset[n_stimuli_per_trial-1::n_stimuli_per_trial] # take every nth, starting at n
    stimulus_pairs = [(onset_trial[x], offset_trial[x]) for x in range(len(offset_trial))] # match pairs, conditioned to existence of offset
    return stimulus_pairs

def get_snippets(df, trans_pairs, t_back=5, t_forw=25):
    """Get snippets based on onset/offset pairs"""
    snips = [df.sel(time=slice(s[0], s[1]+t_forw)) - df.sel(time=slice(s[0]-t_back, s[0])).mean() for s in trans_pairs]
    ms = min(len(s.time) for s in snips)
    snips = np.dstack(s[:, :ms] for s in snips)
    snippets = xr.DataArray(snips, coords=[df.coords[df.dims[0]], np.arange(ms), np.arange(len(trans_pairs))],
                            dims=[df.dims[0], 'time', 'stim'])
    return snippets


def add_supplementary_traces(recording, stimulus=None, derivative=None, timeconstant=2.6):
    """ Add some supplementary single traces to the data set.

    Calling this function adds some pre-defined functions (related to the stimulus) to Zecording class
    This function can currently only be accessed from the console.
    """
    if stimulus is None:
        stimulus = recording.stimulus

    pos_only = np.zeros_like(stimulus)
    neg_only = np.zeros_like(stimulus)
    pos_only[np.where(stimulus > 0)[0]] = stimulus[np.where(stimulus > 0)[0]]  # positive stimulus only
    neg_only[np.where(stimulus < 0)[0]] = stimulus[np.where(stimulus < 0)[0]]  # negative stimulus only

    if derivative is None:
        derivative = np.gradient(stimulus)

    supp_data = {'deriv_stim': derivative,
                 'abs_deriv_stim': np.abs(np.gradient(stimulus)),
                 'pos_only_stim': pos_only,
                 'abs_neg_only_stim': np.abs(neg_only)}  # dictionary of supplementary data to add

    for data_name in supp_data:  # put all supp. data in rec
        recording.add_supp_single_data(s_name=data_name, s_data=supp_data[data_name])

    pos_deriv = derivative.copy()
    pos_deriv[pos_deriv < 0] = 0
    neg_deriv = derivative.copy()
    neg_deriv[neg_deriv > 0] = 0
    recording.add_supp_single_data('pos_deriv_stim', pos_deriv)  # positive derivative only
    recording.add_supp_single_data('neg_deriv_stim', neg_deriv)  # postiive derivative only

    # Add exponentially smoothed functions. Especially useful for fast derivatives.
    recording.add_supp_single_data('conv_stim', exp_smooth(stimulus, time_constant=timeconstant))
    recording.add_supp_single_data('conv_deriv_stim', exp_smooth(derivative, time_constant=timeconstant))
    recording.add_supp_single_data('conv_pos_stim', exp_smooth(recording.pos_only_stim, time_constant=timeconstant))
    recording.add_supp_single_data('conv_neg_stim', exp_smooth(recording.abs_neg_only_stim, time_constant=timeconstant))
    recording.add_supp_single_data('conv_pos_deriv_stim', exp_smooth(recording.pos_deriv_stim, time_constant=timeconstant))
    recording.add_supp_single_data('conv_neg_deriv_stim', exp_smooth(recording.neg_deriv_stim, time_constant=timeconstant))
    recording.add_supp_single_data('abs_conv_neg_deriv_stim', np.abs(exp_smooth(recording.neg_deriv_stim, time_constant=timeconstant)))
    recording.add_supp_single_data('abs_conv_all_deriv_stim', np.abs(exp_smooth(derivative, time_constant=timeconstant)))

def create_regressors(recording, stimulus_path, regressor_names, average_peak_heights=True, verbose=False,
                      plot_regressors=False, names_force5to4_regressors=[], data_key='default'):
    """Add stimulus derived functions to recording, create and return
    stimulus_pairs and (standard scaled) avg_regressors based on stimulus_path.
    """
    stimulus_file = pd.read_csv(stimulus_path, sep='\t', header=None)  # read in default format
    stimulus_file.columns=['computer_time', 'real_time', 'position']
    position = np.array(stimulus_file.position)
    real_time = np.array(stimulus_file.real_time - stimulus_file.real_time[0])  # set t0 =0
    # time_offset = 0.141  # One could add an offset
    real_time = real_time #- time_offset
    position_interp_fun = interp1d(real_time, position, assume_sorted=True, kind='cubic')  # interpolate function of stimulus

    time_diffs = np.diff(real_time)
    derivative = np.zeros_like(position)  #interpolate derivative manually (based on np.gradient) due to non constant time_diffs
    len_stimulus = len(position)
    derivative[0] = (position[1] - position[0]) / time_diffs[0]
    derivative[len_stimulus - 1] = (position[-1] - position[-2]) / time_diffs[-1]
    middle_inds = np.arange(1, len_stimulus-2)
    derivative[middle_inds] = ((position[middle_inds + 1] - position[middle_inds - 1]) /
                               (time_diffs[middle_inds] + time_diffs[middle_inds - 1]))

    new_stim = position_interp_fun(recording.times)  # interpolate recording times for stimulus
    time_diffs = np.diff(recording.times)
    new_der = np.zeros_like(new_stim)  # get gradient of new_stim for derivative
    len_stimulus = len(new_stim)
    new_der[0] = (position[1] - new_stim[0]) / time_diffs[0]  # compute gradient by hand to account for non constant time_diffs (following np.gradient() method)
    new_der[len_stimulus - 1] = (new_stim[-1] - new_stim[-2]) / time_diffs[-1]
    middle_inds = np.arange(1, len_stimulus-2)
    new_der[middle_inds] = ((new_stim[middle_inds + 1] - new_stim[middle_inds - 1]) /
                               (time_diffs[middle_inds] + time_diffs[middle_inds - 1]))

    sr_signal = (len(new_stim) -1 ) /  recording.times[-1] # Hz  # sampling rate
    if verbose:
        print(f'sampling rate {sr_signal}')
    new_stim_smoothed = exp_smooth(new_stim, time_constant=(sr_signal * 2.6))  # smoothed stim
    new_der_smoothed = exp_smooth(new_der, time_constant=(sr_signal * 2.6))

    # sr_full = (len(position) -1 ) / real_time[-1] # Hz  # sampling rate
    # print(f'sampling rate {sr_full}')
    # true_stim_smoothed = pf.exp_smooth(position, time_constant=(sr_full * 2.6))  # smoothed stim
    # true_der_smoothed = pf.exp_smooth(derivative, time_constant=(sr_full * 2.6))

    ## add stim functions to recording
    recording.add_supp_single_data('conv_stim', new_stim_smoothed)
    recording.add_supp_single_data('conv_deriv_stim', new_der_smoothed)
    recording.add_supp_single_data('derivative', new_der)
    recording.add_supp_single_data('conv_pos_stim', np.clip(a=new_stim_smoothed, a_min=0, a_max=1000))
    recording.add_supp_single_data('conv_neg_stim', np.clip(a=new_stim_smoothed, a_max=0, a_min=-1000) * -1)
    recording.add_supp_single_data('conv_pos_deriv_stim', np.clip(a=new_der_smoothed, a_min=0, a_max=1000))
    recording.add_supp_single_data('abs_conv_neg_deriv_stim', np.clip(a=new_der_smoothed, a_max=0, a_min=-1000) * -1)

    stimulus_pairs = create_onset_offset_pairs_step(recording=recording, stimulus=new_stim, abs_tol=0.1, n_stimuli_per_trial=6, stimtype='step')  # get onset offset stimulus pairs
    if data_key in names_force5to4_regressors:
        stimulus_pairs_use = stimulus_pairs[:-1]
    else:
        stimulus_pairs_use = stimulus_pairs

    avg_regressors = trial_average_recdata(recording=recording, stim_pairs=stimulus_pairs_use, names=regressor_names,
                                          datatype_name='regressor', standardscale=True)  # get trial averaged regressors

    if average_peak_heights:
        ## average peaks
        arr_regr = np.array(avg_regressors)
        tmp_local_maxima_pos = np.where(np.r_[True, arr_regr[2, 1:] > arr_regr[2, :-1]] & np.r_[arr_regr[2, :-1] > arr_regr[2, 1:], True])[0]
        tmp_local_maxima_neg = np.where(np.r_[True, arr_regr[3, 1:] > arr_regr[3, :-1]] & np.r_[arr_regr[3, :-1] > arr_regr[3, 1:], True])[0]

        true_peaks = np.array([np.array([50*i -10, 50*i+ 10]) for i in range(12)])  # create ranges in which true peaks should be
        pos_true_peaks = true_peaks[np.array([0, 3, 4, 7, 8, 11]), :]  # split ranges into positive and negative peaks
        neg_true_peaks = true_peaks[np.array([1, 2, 5, 6, 9, 10]), :]

        local_maxima_pos = []
        local_maxima_neg = []
        k=0
        for ii in tmp_local_maxima_pos:  # loop through found peaks to extract peaks within true range
            if (ii > pos_true_peaks[k, 0]) and (ii < pos_true_peaks[k, 1]):
                local_maxima_pos.append(ii)  # save extracted peaks in this var
                k += 1
        k=0
        for ii in tmp_local_maxima_neg:
            if (ii > neg_true_peaks[k, 0]) and (ii < neg_true_peaks[k, 1]):
                local_maxima_neg.append(ii)
                k += 1

        if len(local_maxima_pos) == 6 and len(local_maxima_neg) == 6:  # if number of peaks is correct, average peaks which should have equal height
            for i_peak in range(3):
                avg_regressors[2, local_maxima_pos[i_peak * 2]] = 0.25 * (avg_regressors[2, local_maxima_pos[i_peak * 2]] + avg_regressors[2, local_maxima_pos[i_peak * 2 + 1]] +
                                                                     avg_regressors[3, local_maxima_neg[i_peak * 2]] + avg_regressors[3, local_maxima_neg[i_peak * 2 + 1]])
                avg_regressors[2, local_maxima_pos[i_peak * 2 + 1]] =  avg_regressors[2, local_maxima_pos[i_peak * 2]]
                avg_regressors[3, local_maxima_neg[i_peak * 2]] = avg_regressors[2, local_maxima_pos[i_peak * 2]]
                avg_regressors[3, local_maxima_neg[i_peak * 2 + 1]] =  avg_regressors[2, local_maxima_pos[i_peak * 2]]

        # rescale back to zero mean unit variance
        regressor_scaler = StandardScaler()
        tmp_scaled_reg = regressor_scaler.fit_transform(X=avg_regressors.transpose())

        avg_regressors = tmp_scaled_reg.transpose()
    if verbose:
        print(data_key)
        print(stimulus_pairs_use)
        print(f'mean: {avg_regressors.mean(axis=1)},\n var: {avg_regressors.var(axis=1)}')
    if plot_regressors:
        plt.plot(avg_regressors.transpose());

    return avg_regressors, stimulus_pairs_use

class LinearRegression(linear_model.LinearRegression):
    """Add t stats and P values to sklearn Linear Regression
    Adapted from https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression"""
    def __init__(self,*args,**kwargs):
        # *args is the list of arguments that might go into the LinearRegression object
        # that we don't know about and don't want to have to deal with. Similarly, **kwargs
        # is a dictionary of key words and values that might also need to go into the orginal
        # LinearRegression object. We put *args and **kwargs so that we don't have to look
        # these up and write them down explicitly here. Nice and easy.

        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = True

        super(LinearRegression,self).__init__(*args,**kwargs)

    # Adding in t-statistics for the coefficients.
    def fit(self,x,y):
        """Adapted from https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        Can be sped up by calculating the sampleVarianceX only once, when doing multiple
        regressions sequentially.

        """
        # This takes in numpy arrays (not matrices). Also assumes you are leaving out the column
        # of constants.

        # Not totally sure what 'super' does here and why you redefine self...
        self = super(LinearRegression, self).fit(x,y)
        n_times, n_regressors = x.shape
        yHat = np.matrix(self.predict(x)).T

        # Change X and Y into numpy matricies. x also has a column of ones added to it.
        x = np.hstack((np.ones((n_times,1)),np.matrix(x)))
        y = np.matrix(y).T

        # Degrees of freedom.
        dof = float(n_times - n_regressors) - 1  # -1 for intercept
        self.dof = dof
        self.residual = np.sum((yHat - y), axis=0)

        # Sample variance.
        sse = np.sum(np.square(yHat - y),axis=0)
        self.sampleVariance = sse/dof

        # Sample variance for x.
        self.sampleVarianceX = x.T*x

        # Covariance Matrix = [(s^2)(X'X)^-1]^0.5. (sqrtm = matrix square root.  ugly)
        self.covarianceMatrix = scipy.linalg.sqrtm(self.sampleVariance[0,0]*self.sampleVarianceX[1:,1:].I)

        # Standard erros for the difference coefficients: the diagonal elements of the covariance matrix.
        self.se = self.covarianceMatrix.diagonal()

        # T statistic for each beta.
        self.betasTStat = np.zeros(n_regressors)
        for i in range(n_regressors):
            self.betasTStat[i]= self.coef_[i]/self.se[i]

        # P-value for each beta. This is a two sided t-test, since the betas can be
        # positive or negative.
        self.betasPValue = 1 - scipy.stats.t.cdf(np.abs(self.betasTStat),dof)

# define a regression function
def regression_fun(data_regression, data_predictors, alp=0.002, t_end=None, splitup=False,
                   method='lasso', njobs=1):

    # single regression!
    # assuming transformed [time, neurons]
    data_regression = np.squeeze(data_regression)
    if len(data_predictors.shape) == 1: # only 1 regressor
        data_predictors = np.array(data_predictors)[:, np.newaxis]
    if t_end is None:
        t_end = data_regression.shape[0]

    if method == 'lasso':
        reg = Lasso(alpha=alp)  # define model
    elif method == 'OLS':
        reg = LinearRegression(n_jobs=njobs)  # define model
    reg.fit(data_predictors[:t_end, :], data_regression[:t_end]) # fit data

    if splitup:
        r2 = reg.score(data_regression[:t_end], data_regression[t_end:]) # return R^2 value
        test_len= len(data_regression[:t_end])
        # make a new one
    elif not splitup:
        r2 = reg.score(data_predictors[:t_end, :], data_regression[:t_end])

    return reg, r2 # return model and score

def correlation_1D2D_datainput(df1, df2): # correlation function from Rémi (local.py)
    """
    Pearson coefficient of correlation between the calcium signals of two neurons
    Calculated manually to be faster in a 1 vector x 1 matrix

    Parameters:
    ----------
        sig_1D: str
            name of single signal
        sigs_2D: str
            name ofmultiple signals
)
    Returns:
    ----------
        r: float
            Coefficient of correlation
    """

    df2 = df2.transpose() # sigs_2D.transpose()
    cov = np.dot(df1 - df1.mean(), df2 - df2.mean(axis=0)) / (df2.shape[0] - 1)
    # ddof=1 necessary because covariance estimate is unbiased (divided by n-1)
    p_var = np.sqrt(np.var(df1, ddof=1) * np.var(df2, axis=0, ddof=1))
    r = cov / p_var
    return r

def make_ref_coords_nsamples(coords_dict, n_voxels_approx = 100000):
    """Create equally spaced rectangular grid of ref coords

    Parameters:
    ------------
        coords1: np.array, (n, 3)
            coordinates of sample 1, remember to use the same coordinate space (i.e. ref. brain)!
        coords2: np.array, (n, 3)
            coordinates of sample 2, remember to use the same coordinate space (i.e. ref. brain)!
        n_voxels_approx, int
            number of grid points/voxels to create approximately
            (depends on rounding)

    Returns:
    -----------
        grid
    """
    min_coords = np.zeros((len(coords_dict), 3))
    max_coords = np.zeros((len(coords_dict), 3))
    index=0
    for i_data, coords in coords_dict.items():
        min_coords[index, :] = coords.min(axis=0)
        max_coords[index, :] = coords.max(axis=0)
        index += 1

    axis_arrays = {}
    min_gen = np.min(min_coords, axis=0)
    max_gen = np.max(max_coords, axis=0)
    volume = (max_gen[0] - min_gen[0]) * (max_gen[1] - min_gen[1]) * (max_gen[2] - min_gen[2])
    scaling_factor = np.cbrt(volume/n_voxels_approx) # because then V = N * scale_factor^3
    for i_dim in range(3):
        axis_arrays[i_dim] = np.arange(min_gen[i_dim], max_gen[i_dim] + scaling_factor, scaling_factor) # create axis array

    gridx, gridy, gridz = np.meshgrid(axis_arrays[0], axis_arrays[1], axis_arrays[2], indexing='ij')
    n_voxels = gridx.shape[0] * gridy.shape[1] * gridz.shape[2]
    grid = np.zeros((n_voxels, 3))
    grid[:, 0] = np.squeeze(gridx.reshape((n_voxels, 1))) # reshape to common format
    grid[:, 1] = np.squeeze(gridy.reshape((n_voxels, 1)))
    grid[:, 2] = np.squeeze(gridz.reshape((n_voxels, 1)))
    return grid

def cluster_density(coords, gridcoords, bw = 0.004, CV_bw=False):
    """Compute KDE and return scores of grid points

   Parameters:
    ------------
        coords: np.array (n, 3)
            coordinates of points in cluster(!)
        gridcoords: np.array (n, 3)
            coordinates of grid, in which the density must be mapped
        bw: float
            bandwidth of Gaussian kernel, now hand tuned, depends on rectangular grid resolution.
            bandwidth also depends on user; how important are 'lonely' neurons in the clusters?

    Returns:
    -----------
        den
            scores of grid coords
    """
    if CV_bw:
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(0.005, 0.03, 11)}, cv=25, verbose=0)
        grid.fit(coords)
        bw = grid.best_params_['bandwidth']
        print(f'Best bandwidth {bw}')
        return grid

    else:
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(coords)
    #    den = np.exp(kde.score_samples(refcoords))  # return scores
        den = kde.score_samples(gridcoords)  # return log scores
        return den

def transfer_clusters(cl_cells, local_coords, grid_coords, bandwidth=0.004, cv_bw=False):
    """Transfer clusters from one fish to rectangular grid

    Parameters:
    -----------
    cl_cells: dict with sets as values
        The keys are cluster ids (ints)
        A value is a set of neuron ids (ints) in cluster {key}
    local_coords: np.array (n_cells, 3)
        coordinates of neurons
    grid_coords: np.array (n_grid_points, 3)
        coordinates of grid points

    Returns:
    -----------
    cl_grid_points: pd.Dataframe
        Rows: grid point ids
        Columns: cl ids;
        Elements: soft cluster assigments (density)

    """
    n_clusters = len(cl_cells)  # assuming cl_cells is a dict
    n_grid_points = grid_coords.shape[0]
    cl_grid_points = pd.DataFrame(np.zeros((n_grid_points, n_clusters)),
                           columns=[f'cluster_{x}' for x in range(n_clusters)])  # output format of grid units
    # for i_cl in tqdm(range(n_clusters)):
    for i_cl in range(n_clusters):
        coords_cl_neurons = local_coords[np.array(list(cl_cells[i_cl])), :]  # find coords of neurons in current cluster
        densities = cluster_density(coords=coords_cl_neurons, gridcoords=grid_coords, bw=bandwidth, CV_bw=cv_bw)  # get density of grid points
        cl_name = f'cluster_{i_cl}'
        cl_grid_points[cl_name] = densities  # use soft assignment
    return cl_grid_points

def threshold_fun(regressors, regression_results, suffix_names=['_Tstat', '_Tstat', '_Tstat', '_Tstat'],
                  absolute_value=[False, False, True, True],
                  percentile=[True, True, False, False],
                  sign_geq=[True, True, True, True],
                  threshold=[97, 97, 0, 0]):
    """Given various conditions, return selection of neurons that fullfills them all.

    Parameters:
    ------------
        regression_results: regr_results pd DataFrame
        suffix_names: list of str
            What suffices to use for the four regressors.
            Can have more elements than 4, should then be full names
            (e.g. ['_Tstat', '_Tstat', '_coef', '_Tstat', 'conv_pos_stim_coef'])
        absolute_values; list of bools
            Same length as suffix_names. Whether to consider absolute value of
            corresponding data set
        percentile; list of bools
            Same length as suffix_names. Whether to consider percentile value of
            corresponding threshold
        sign_geq; list of bools
            Same length as suffix_names. Whether to consider 'greater or equal (>=)'
            comparison (True) or less than (<) (False) of corresponding data set.
        threshold: list of floats/ints
            Same length as suffix_names. Threshold to use for corresponding data set.

    Returns:
    ----------
        indices: np.array of ints
            indices of neuron in selection
        """

    data = {}
    for iloop, rn in enumerate(regressors):  # add data loop
        data[rn] = getattr(regression_results, rn + suffix_names[iloop])
        if absolute_value[iloop]:
            data[rn] = np.abs(data[rn])
    indices = set(np.arange(regression_results.shape[0]))  # start with all
    for iloop, rn in enumerate(regressors):  # find selection loop (can be merged with prev. loop)
        if percentile[iloop]:
            tmp_perc = np.nanpercentile(data[rn], threshold[iloop])
        elif percentile[iloop] == False:
            tmp_perc = threshold[iloop]
        if sign_geq[iloop]:
            selection = np.where(data[rn] >= tmp_perc)[0]
        elif sign_geq[iloop] == False:
            selection = np.where(data[rn] <= tmp_perc)[0]
        indices = indices.intersection(set(selection))  # find intersection


    if len(suffix_names) >= 4:  # if more names given, assume full names
        for iloop, sname in enumerate(suffix_names[4:]):
            kloop = iloop + 4
            data[sname] = getattr(regression_results, sname)
            if percentile[kloop]:
                tmp_perc = np.nanpercentile(data[sname], threshold[kloop])
            elif percentile[kloop] == False:
                tmp_perc = threshold[kloop]
            if sign_geq[kloop]:
                selection = np.where(data[sname] >= tmp_perc)[0]
            else:#if sign_geq[iloop] == False:
                selection = np.where(data[sname] < tmp_perc)[0]
            indices = indices.intersection(set(selection))  # find intersection
    indices = np.array(list(indices))
    return indices

def trial_average_recdata(stim_pairs, regressors_data=None, recording=None, names=['conv_pos_stim', 'conv_neg_stim',
                          'conv_pos_deriv_stim', 'abs_conv_neg_deriv_stim'],
                          standardscale=True, t_backward=5, t_forward=18, datatype_name='regressor'):

    """Function to make trial averaged data of single time traces in rec (such as
    the stimulus-derived functions).

    Parameters:
    ------------
        recording: instance of utilities.Zecording class
        stim_pairs: return instance of create_onset_offset_pairs_step() function
            onset and offset pairs of stimulus
        names: list of strings, should be in recording and be single time traces
            data names in recording
        standardscale: bool
            Whether to standardscale the data from names (i.e. zero mean unit variance)
        t_backward: int
            Should be the same value as that was used to generate averaged neuronal data!
        t_forward: int
            Should be the same value as that was used to generate averaged neuronal data!
        datatype_name: str
            This string becomes the label of the rows in the resulting data array

    Returns:
    ----------
        avg_regressors: xr.DataArray
            rows are trial averaged time traces of the data in names
    """
    regressor_scaler = StandardScaler()  # scaler to use
    if regressors_data is None:
        regressors_data = np.zeros((len(names), recording.n_times))
        for iloop, r in enumerate(names):
            regressors_data[iloop, :] = getattr(recording, r)
    df_reg = xr.DataArray(regressors_data,
                   coords=[names, np.arange(recording.n_times)],
                   dims=[datatype_name, 'time'])
    snippets_regressors = get_snippets(df_reg, trans_pairs=stim_pairs, t_back=t_backward, t_forw=t_forward)
    snippets_regressors = snippets_regressors.mean('stim')
    if standardscale:
        scaled_regressors = regressor_scaler.fit_transform(X=snippets_regressors.transpose())
        data_use = scaled_regressors.transpose()
    else:
        data_use = snippets_regressors
    avg_regressors = xr.DataArray(np.double(data_use),
                       coords=[names, np.arange(len(snippets_regressors.time))],
                       dims=[datatype_name, 'time'])
    return avg_regressors

def make_clusters(regr_results, regressor_names, percentile_threshold=95, position_coefT=True,
                  velocity_coefT=True, demix_position_velocity=False):
    """Function that makes position and velocity clusters for a regr_results instance.
    It calls threshold_fun() to do the actual thresholding.
    The new clusters added the regr_results that is passed as input.

    Parameters:
    ------------
        regr_results, instance of regression results
        regressor_names. standard names
        percentile_threshold: float
            percentile to use for all thresholding
        position_coefT: bool
            whether to make position clusters
        velocity_coefT: bool
            whether to make velocity clusters
        demix_position_velocity: bool
            whether to demix position and velocity per signed stimulus

    Returns:
    -----------
        regr_results
    """
    selection_indices = {}
    if position_coefT:
        selection_names = [f'negstim_high{percentile_threshold}',f'posstim_high{percentile_threshold}']
        selection_indices[selection_names[0]] = threshold_fun(regressors=regressor_names, regression_results=regr_results,
                                                            suffix_names=['_Tstat', '_Tstat', '_Tstat', '_Tstat', 'conv_neg_stim_coef'],
                                                              absolute_value=[True, False, True, True, False],
                                                              percentile=[False, True, False, False, True],
                                                              sign_geq=[False, True, False, False, True],
                                                              threshold=[300,percentile_threshold,300,300, percentile_threshold])
        selection_indices[selection_names[1]] = threshold_fun(regressors=regressor_names,regression_results=regr_results,
                                                              suffix_names=['_Tstat', '_Tstat', '_Tstat', '_Tstat', 'conv_pos_stim_coef'],
                                                              absolute_value=[False, True, True, True, False],
                                                              percentile=[True, False, False, False, True],
                                                              sign_geq=[True, False, False, False, True],
                                                              threshold=[percentile_threshold,300,300,300, percentile_threshold])
        # color_dict_select[selection_names[0]] = color_cycle[1]
        # color_dict_select[selection_names[1]] = color_cycle[0]
        selection_intersection_high = set()
        selection_intersection_high = set(selection_indices[selection_names[0]]).intersection(set(selection_indices[selection_names[1]]))
        if len(selection_intersection_high) >= 5:
            selection_names.append(f'posnegstim_high{percentile_threshold}')
            selection_indices[selection_names[0]] = np.array(list(set(selection_indices[selection_names[0]]).difference(selection_intersection_high)))
            selection_indices[selection_names[1]] = np.array(list(set(selection_indices[selection_names[1]]).difference(selection_intersection_high)))
            selection_indices[f'posnegstim_high{percentile_threshold}'] = np.array(list(selection_intersection_high))
            # color_dict_select['posnegstim_high95'] = color_cycle[9]

    if velocity_coefT:
        selection_names = [f'negderiv_high{percentile_threshold}', f'posderiv_high{percentile_threshold}']
        selection_indices[selection_names[0]] = threshold_fun(regressors=regressor_names,regression_results=regr_results,
                                                              suffix_names=['_Tstat', '_Tstat', '_Tstat', '_Tstat', 'abs_conv_neg_deriv_stim_coef'],
                                                              absolute_value=[True, True, True,False, False],
                                                              percentile=[False, False, False,True, True],
                                                              sign_geq=[False, False, False,True, True],
                                                              threshold=[300,300,300, percentile_threshold,percentile_threshold])
        selection_indices[selection_names[1]] = threshold_fun(regressors=regressor_names,regression_results=regr_results,
                                                              suffix_names=['_Tstat', '_Tstat', '_Tstat', '_Tstat', 'conv_pos_deriv_stim_coef'],
                                                              absolute_value=[True, True,False, True, False],
                                                              percentile=[ False, False, True, False, True],
                                                              sign_geq=[ False, False,True, False, True],
                                                              threshold=[300,300,percentile_threshold,300,percentile_threshold])

        # color_dict_select[selection_names[0]] = color_cycle[3]
        # color_dict_select[selection_names[1]] = color_cycle[2]
        selection_intersection_high = set()
        selection_intersection_high = set(selection_indices[selection_names[0]]).intersection(set(selection_indices[selection_names[1]]))
        if len(selection_intersection_high) >= 1:
            selection_names.append(f'posnegderiv_high{percentile_threshold}')
            selection_indices[selection_names[0]] = np.array(list(set(selection_indices[selection_names[0]]).difference(selection_intersection_high)))
            selection_indices[selection_names[1]] = np.array(list(set(selection_indices[selection_names[1]]).difference(selection_intersection_high)))
            selection_indices[f'posnegderiv_high{percentile_threshold}'] = np.array(list(selection_intersection_high))
            # color_dict_select['posnegderiv_high95'] = color_cycle[8]

    if demix_position_velocity:
        mixing_names = {}
        mixing_names['positive'] = [f'posstim_high{percentile_threshold}', f'posderiv_high{percentile_threshold}']
        mixing_names['negative'] = [f'negstim_high{percentile_threshold}', f'negderiv_high{percentile_threshold}']
        for key, vallist in mixing_names.items():
            name0_excl = vallist[0] + '_excl'
            name1_excl = vallist[1] + '_excl'
            name_mix = key + '_mixed'
            selection = {}
            selection0 = selection_indices[vallist[0]]
            selection1 = selection_indices[vallist[1]]
            selection[name0_excl] = set(selection0).difference(set(selection1))
            selection[name1_excl] = set(selection1).difference(set(selection0))
            selection[name_mix] = set(selection0).intersection(set(selection1))
            for kk in selection.keys():
                selection_indices[kk] = np.array(list(selection[kk]))

    for rname in selection_indices.keys():
        tmp = np.zeros(regr_results.shape[0])
        if len(selection_indices[rname]) > 0:
            tmp[selection_indices[rname]] = 1
            regr_results[rname] = tmp

    return regr_results
