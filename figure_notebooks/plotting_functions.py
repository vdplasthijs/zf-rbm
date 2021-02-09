# @Author: Thijs L van der Plas <thijs>
# @Date:   2020-11-23
# @Email:  thijs.vanderplas@dtc.ox.ac.uk
# @Filename: plotting_functions.py
# @Last modified by:   thijs
# @Last modified time: 2020-11-23



## module with plotting functions

import numpy as np
# import numpy.matlib
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar as mpl_colorbar

import matplotlib.cm
import sys,os
import importlib
import seaborn as sns
import pandas as pd
import scipy.stats

from cycler import cycler
## Create list with standard colors:
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))


dr_legend = {'pca': 'PCA', 'rbm': 'RBM', 'fa': 'FA', 'ica': 'ICA', 'glm': 'GLM'}
dr_colors = {'glm': '#008B8B', 'pca': '#808000', 'rbm': '#800080', 'fa': 'red', 'ica':'#157bf9'}
dr_names = {'rbm': 'cRBM', 'RBM': 'cRBM', 'pca': 'PCA', 'PCA': 'PCA',
            'covariance': 'covariance', 'correlation': 'correlation'}
def set_fontsize(font_size=12):
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.autolimit_mode'] = 'data' # default: 'data'
    params = {'legend.fontsize': font_size,
             'axes.labelsize': font_size,
             'axes.titlesize': font_size,
             'xtick.labelsize': font_size,
             'ytick.labelsize': font_size}
    plt.rcParams.update(params)

def plot_example_hu(ax=None, hu_screenshot_folder='/home/thijs/repos/zf-rbm/figures/HU_screenshots_2020-05-16-0844',
                    hu_id=86, id_reference='internal',
                    filename_prefix='screenshot_hu_', y_min=400, y_max=1600,
                    fontsize=10, raster_order=None):
    '''fontsize: 15 for (10, 10) fig, linearly scaling seems fine '''
    assert id_reference == 'internal' or id_reference == 'raster'  # does the ID refer to internal index or (dynamic) raster plot index?
    if raster_order is not None:
        raster_order_reverse = np.zeros_like(raster_order)  # reverse order to find corresponding names
        for i_map, map in enumerate(raster_order):
            raster_order_reverse[map] = i_map
    if id_reference == 'internal':
        load_id = hu_id
        internal_id = hu_id
        if raster_order is not None:
            raster_id = raster_order_reverse[hu_id]
        else:
            raster_id = 'NA'
    elif id_reference == 'raster':
        assert raster_order is not None
        load_id = raster_order[hu_id]
        internal_id = load_id
        raster_id = hu_id
    # screenshot_files = os.listdir(hu_screenshot_folder)
    filename = filename_prefix + str(load_id).zfill(3) + '.png'
    image = mpimg.imread(os.path.join(hu_screenshot_folder, filename))

    if ax is None:
        ax = plt.subplot(111)
    ax.imshow(image[:, y_min:y_max, :])
    ax.text(s='Ro', x=25, y=71, c='white', fontdict={'size': fontsize})
    ax.text(s='C', x=145, y=71, c='white', fontdict={'size': fontsize})
    ax.text(s='R', x=95, y=1, c='white', fontdict={'size': fontsize, 'va': 'top'})
    ax.text(s='L', x=95, y=124, c='white', fontdict={'size': fontsize})

    ax.text(s='Ro', x=25, y=571, c='white', fontdict={'size': fontsize})
    ax.text(s='C', x=145, y=571, c='white', fontdict={'size': fontsize})
    ax.text(s='D', x=95, y=501, c='white', fontdict={'size': fontsize, 'va': 'top'})
    ax.text(s='V', x=95, y=624, c='white', fontdict={'size': fontsize})
    ax.text(s='0.1 mm', x=1000, y=840, c='white', fontdict={'size': fontsize})

    ax.set_title(f'HU index: raster: {raster_id}, internal: {internal_id}', fontdict={'weight': 'bold'})
    ax.axis('off')

    return ax

def break_word(word, break_size=25):
    '''Function that tries to  break up word at first space after given break_size'''
    if len(word) > break_size:
        for ii in range(break_size, len(word)):
            if word[ii] == ' ': # first space encountered
                word = word[:ii] + '\n' + word[ii:]  #insert break
                break
    return word

def plot_info_hu(raster_order, hu_activity, RBM, rec, selection_neurons,
                 mu=82, n_regions=8, abs_th=0.05, region_absolute=True, save_fig=False,
                 save_folder='/home/thijs/repos/zf-rbm/figures/HU_info_2020-05-16-0844/'):
    '''Function to plot HU with dynamic activity and top regions

    Parameters:
    -----------
        raster_order: np array
            array of length HU, describing the raster mapping (just for name sake)
        hu_activity: 2D np array
            activity matrix (HU x time)
        RBM: RBM class
            RBM to use
        rec: zecording object
            data set to use
        selectoin_neurons: np array
            array of indices neurons to use (of rec )
        mu: int
            current HU index
        n_regions: int
            how many regions to plot
        abs_th: float
            threshold for absolute weights (to count in reigon)
        region_absolute: bool
            if True, sort by absolute number of neurons, if False, by relative
        save_fig: bool
            whether to save

    Returns:
    -----------
        fig: fig handle
    '''
    # plt.rcParams['figure.figsize'] = (10, 5)

    fig = plt.figure(constrained_layout=False)
    gs_im = fig.add_gridspec(ncols=1, nrows=1, bottom=0.3, top=0.95,
                             hspace=0, wspace=0, left=0.05, right=0.5)  # [1, 2.2, 1.2]
    gs_regions = fig.add_gridspec(ncols=1, nrows=1, bottom=0.3, top=0.95,
                                 hspace=0, wspace=0, left=0.85, right=0.95)
    gs_trace = fig.add_gridspec(ncols=1, nrows=1, bottom=0.05, top=0.18,
                                hspace=0, left=0.05, right=0.95)

    freq = 1 / np.mean(np.diff(rec.time))
    raster_order_reverse = np.zeros_like(raster_order)  # reverse order to find corresponding names
    for i_map, map in enumerate(raster_order):
        raster_order_reverse[map] = i_map

    ## plot screenshot
    ax_im = fig.add_subplot(gs_im[0, 0])
    plot_example_hu(fontsize=7.5, ax=ax_im, hu_id=mu)
    ax_im.set_title(f'HU (raster plot index): {raster_order_reverse[mu]}, internal index: {mu}',
                    fontdict={'weight': 'bold'})

    ## plot time trace
    ax_trace = fig.add_subplot(gs_trace[0, 0])
    ax_trace.plot(hu_activity[mu, :], c='k', lw=2)
    ax_trace.set_xticks([x * int(60 * freq) for x in range(9)])  # set time in seconds
    ax_trace.set_xticklabels((np.round(ax_trace.get_xticks() / freq)).astype('int'))
    ax_trace.set_xlabel('Time (s)'); ax_trace.set_ylabel('Activity (a.u.)')
    ax_trace.spines['top'].set_visible(False)
    ax_trace.spines['right'].set_visible(False)

    ## plot region occupancy
    hu_weights = RBM.weights[mu, :]
    vu_labels = rec.labels[selection_neurons, :]
    vu_selection = np.abs(hu_weights) >= abs_th  # VUs with weights absolute value geq threshold
    vu_selection_labels = vu_labels[vu_selection, :]  # their label
    n_vu_per_region_total = np.squeeze(np.array(vu_labels.sum(0)))  # total number of neurons per region
    n_vu_per_region = np.squeeze(np.array(vu_selection_labels.sum(0)))  # number of these neurons per region

    nz_ind = np.where(n_vu_per_region_total)
    divided_arr = np.zeros_like(n_vu_per_region) + np.nan
    divided_arr[nz_ind] = n_vu_per_region[nz_ind] / n_vu_per_region_total[nz_ind]  # divide without zeroes
    inds_sorted_div = np.argsort(divided_arr)[::-1] # nan to large to small
    if region_absolute:
        top_region_labels = np.argsort(n_vu_per_region)[::-1]  # sort by top absolute number
    else:
        top_region_labels = inds_sorted_div[np.logical_not(np.isnan(divided_arr[inds_sorted_div]))]  # sort by relative & remove nans
    height_bars = divided_arr[top_region_labels[:n_regions]]
    height_bars_abs = n_vu_per_region[top_region_labels[:n_regions]]
    with open('/home/thijs/repos/fishualizer/Content/RegionLabels.txt', 'r') as f:
        all_names = f.readlines()
    labelnames = np.array([x.strip().rstrip('-') for x in all_names])

    ax_regions = fig.add_subplot(gs_regions[0, 0])
    ax_regions_abs = ax_regions.twiny()  # create copy with shared y
    ax_regions.barh(y=np.arange(n_regions)- 0.15, color='grey',
                   width=height_bars * 100, height=0.3)  # multiply with 100 for percetnage
    ax_regions_abs.barh(y=np.arange(n_regions) + 0.15, color='m',
                   width=height_bars_abs, height=0.3)

    ax_regions.set_ylabel('Region name')
    ax_regions.set_xlabel('% of neurons in that region', fontdict={'color': 'k'})
    ax_regions_abs.set_xlabel('# neurons', fontdict={'color': 'm'})
    ax_regions_abs.tick_params(axis='x', colors="m")
    yticklabels = labelnames[top_region_labels[:n_regions]]
    for iyt, yt in enumerate(yticklabels):
        new_yt = break_word(word=yt, break_size=30)
        yticklabels[iyt] = new_yt
    ax_regions.set_yticks(np.arange(n_regions))
    ax_regions.set_yticklabels(yticklabels, rotation=0)
    ax_regions_abs.spines['right'].set_visible(False)
    ax_regions.spines['right'].set_visible(False)

    if save_fig:
        plt.savefig(os.path.join(save_folder, f'info_hu_{raster_order_reverse[mu]}.pdf'),
                 dpi=500, bbox_inches='tight')

    return fig

def plot_uniform_distr(w_mat, sh_w_mat, cdf_mat, sh_cdf_mat, k_value=0, dr='rbm',
                       print_pval=True, ax=None, plot_legend=False):
    '''To be used with af.count_connections_2()'''
    if ax is None:
        ax = plt.subplot(111)
    dict_str_number = {0: '1st', 1: '2nd', 2: '3rd'}

    cdf_plot = np.array([0] + list(cdf_mat[dr][k_value, :]))  # add 0 to start of cum pdf, also pans out with extra weight bin
    sh_cdf_plot = np.array([0] + list(sh_cdf_mat[dr][k_value, :]))


    ax.plot(sh_w_mat[dr][k_value, :], sh_cdf_plot, linestyle=':',
             label=f'{dr_legend[dr]} shuffled', linewidth=5, color='k')#dr_colors[dr]);
    ax.plot(w_mat[dr][k_value, :], cdf_plot, color=dr_colors[dr],
         linewidth=5, label=f'{dr_legend[dr]}');
    ax.set_title(f'{dict_str_number[k_value]} strongest connection of {dr_legend[dr]}',
                                   fontdict={'weight': 'bold'})
    ax.set_xlabel('Largest weight per neuron');
    ax.set_ylabel(f'Cumulative PDF');
    if plot_legend:
        ax.legend(bbox_to_anchor=(0.5, 0.5),
                                    frameon=False);

    p_val = scipy.stats.ks_2samp(cdf_plot, sh_cdf_plot)[1]
    x_pos = np.minimum(w_mat[dr][k_value, 0], sh_w_mat[dr][k_value, 0])
    ax.text(s=f'P = {np.round(p_val, 3)}', x=x_pos, y=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax

def plot_degree_distr(degree_dict, degree_dict_sh, ax=None, dr='rbm', plot_shuffled=True,
                      bar_width=None, cutoff=8, normalise=False, colour=None,
                      label=None, label_sh=None, v_spacing_greater=False):
    assert dr in degree_dict.keys(), f'{dr} not in degree_dict'
    assert degree_dict[dr].min() == 0 or degree_dict_sh[dr].min() == 0  # make life easy
    if ax is None:
        ax = plt.subplot(111)
    if colour is None:
        colour = dr_colors[dr]
    if label is None:
        label = dr_legend[dr]
    if label_sh is None:
        label_sh = 'shuffled ' + dr_legend[dr]

    max_degree = degree_dict[dr].max()
    max_degree_sh = degree_dict_sh[dr].max()
    if cutoff > max_degree:
        cutoff = max_degree
        print(f'cut off decreased to max degree {max_degree}')
    if cutoff > max_degree_sh:
        cutoff = max_degree_sh
        print(f'cut off decreased to max degree shuffled {max_degree_sh}')

    degree_values = np.arange(cutoff + 1)
    degree_values_sh = np.arange(cutoff + 1)
    tmp_bar_heights = np.array([np.sum(degree_dict[dr] == d) for d in range(max_degree + 1)])
    tmp_bar_heights_sh = np.array([np.sum(degree_dict_sh[dr] == d) for d in range(max_degree_sh + 1)])
    bar_heights = np.zeros_like(degree_values)
    bar_heights_sh = np.zeros_like(degree_values_sh)
    bar_heights[:cutoff] = tmp_bar_heights[:cutoff]
    bar_heights_sh[:cutoff] = tmp_bar_heights_sh[:cutoff]
    bar_heights[cutoff] = np.sum(tmp_bar_heights[cutoff:])
    bar_heights_sh[cutoff] = np.sum(tmp_bar_heights_sh[cutoff:])

    assert np.sum(bar_heights) == np.sum(tmp_bar_heights)
    if normalise:
        bar_heights = bar_heights / np.sum(bar_heights)
        bar_heights_sh = bar_heights_sh / np.sum(bar_heights_sh)
    if bar_width is None:
        if plot_shuffled:
            bar_width = 0.4
            x_degrees = degree_values - bar_width / 2
        else:
            bar_width = 0.8
            x_degrees = degree_values
    ax.bar(x=x_degrees, height=bar_heights, width=bar_width, color=colour,
           label=label)
    if plot_shuffled:
        ax.bar(x=degree_values_sh + bar_width / 2, height=bar_heights_sh,
               width=bar_width, color=colour, alpha=0.4,
               label=label_sh)

    if plot_shuffled:
        ax.legend(loc='upper right', frameon=False)
    ax.set_xticks(degree_values)
    xlabels = [str(x) for x in degree_values]
    if v_spacing_greater:
        xlabels[-1] = '>'
    else:
        xlabels[-1] = f'>{xlabels[-2]}'
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('# Strong weights per neuron')
    if normalise:
        ax.set_ylabel('PDF')
    else:
        ax.set_ylabel('Frequency')
    ax.set_title(f'Connectivity to hidden layer {label}', fontdict={'weight': 'bold'})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def plot_three_degree_distr(degree_dict, ax=None,
                      bar_width=None, cutoff=8, normalise=False, colour_dict=None, alpha_dict=None,
                      v_spacing_greater=False):
    # assert degree_1.min() == 0 or degree_2.min() == 0 or degree_3.min() == 0 # make life easy
    if ax is None:
        ax = plt.subplot(111)
    if colour_dict is None:
        colour_dict = {k: 'k' for k in degree_dict.keys()}
    if alpha_dict is None:
        alpha_dict = {k: 1 - 0.9 * ii / len(degree_dict.keys()) for ii, k in enumerate(degree_dict.keys())}

    max_degree_dict = {k: v.max() for k, v in degree_dict.items()}
    for k, v in max_degree_dict.items():
        if cutoff > v:
            cutoff = v
            print(f'cut off decreased to max degree {v}')
    tmp_bar_heights, bar_heights = {}, {}
    for k, degree in degree_dict.items():
        degree_values = np.arange(cutoff + 1)
        tmp_bar_heights[k] = np.array([np.sum(degree == d) for d in range(max_degree_dict[k] + 1)])
        bar_heights[k] = np.zeros_like(degree_values)
        bar_heights[k][:cutoff] = tmp_bar_heights[k][:cutoff]
        bar_heights[k][cutoff] = np.sum(tmp_bar_heights[k][cutoff:])

        assert np.sum(bar_heights[k]) == np.sum(tmp_bar_heights[k])

        if normalise:
            bar_heights[k] = bar_heights[k] / np.sum(bar_heights[k])

    if bar_width is None:
        bar_width = 0.8 / len(degree_dict)
        x_degrees = degree_values - bar_width / 2

    i_plot = 0
    for k, b_heights in bar_heights.items():
        ax.bar(x=x_degrees + bar_width * i_plot, height=b_heights, width=bar_width, color=colour_dict[k],
               label=k, alpha=alpha_dict[k])
        i_plot += 1
    ax.legend(loc='upper right', frameon=False)
    ax.set_xticks(degree_values)
    xlabels = [str(x) for x in degree_values]
    if v_spacing_greater:
        xlabels[-1] = '>'
    else:
        xlabels[-1] = f'>{xlabels[-2]}'
    ax.set_xticklabels(xlabels)
    # ax.set_xlabel('# Strong weights per neuron')
    if normalise:
        ax.set_ylabel('PDF')
    else:
        ax.set_ylabel('Frequency')
    # ax.set_title(f'Connectivity to hidden layer {label}', fontdict={'weight': 'bold'})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax



def plot_binned_stats(ax, plot_bins, mean_bins, std_bins, comp_moment=None):
    if comp_moment is not None:
        plot_color = mom_colors[comp_moment]
    else:
        plot_color = 'grey'
    std_se = 'std'
    inds_nn = ~np.isnan(mean_bins)
    plot_bins, mean_bins, std_bins = plot_bins[inds_nn], mean_bins[inds_nn], std_bins[inds_nn]

    ax.plot([plot_bins.min(), plot_bins.max()], [plot_bins.min(), plot_bins.max()], ':', color='k', label='y=x')
    ax.plot(plot_bins, mean_bins, color=plot_color, label='mean')
    ax.fill_between(plot_bins, mean_bins, mean_bins+std_bins, color=plot_color, alpha=0.3, label=std_se)
    ax.fill_between(plot_bins, mean_bins, mean_bins-std_bins, color=plot_color, alpha=0.3)
    ax.set_xlim([plot_bins.min(), plot_bins.max()])
    ax.set_ylim([plot_bins.min(), plot_bins.max()])
    return ax

def plot_reproduc_mat(dict_mat, key, ax=None, corr_type='pearson',
                      one_indexing_labels=True, plot_diag=True):
    if ax is None:
        ax = plt.subplot(111)
    data = dict_mat[key][corr_type]
    mask = np.zeros_like(data)
    if plot_diag:
        mask[np.triu_indices_from(data, k=1)] = True
    else:
        mask[np.triu_indices_from(data, k=0)] = True
    sns.heatmap(data, annot=False, ax=ax, square=True, mask=mask, vmin=0, vmax=1,
                linewidths=0, rasterized=True,
                cmap='pink_r', cbar_kws={'label': f'{corr_type[0].upper() + corr_type[1:]} correlation'})
    ax.set_xlabel('# Fish')
    ax.set_ylabel('# Fish')
    if plot_diag:
        ax.set_xticks(np.arange(len(data)) + 0.5)
        ax.set_yticks(np.arange(len(data)) + 0.5)
    else:
        ax.set_xticks(np.arange(len(data) - 1) + 0.5)
        ax.set_yticks(np.arange(1, len(data)) + 0.5)
    if one_indexing_labels:
        ax.set_xticklabels([str(x + 1) for x in range(len(data))])
        ax.set_yticklabels([str(x + 1) for x in range(len(data))])
    else:
        ax.set_xticklabels([str(x + 0) for x in range(len(data))])
        ax.set_yticklabels([str(x + 1) for x in range(len(data))])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5);
    ax.set_title('Similarity between individual fish', fontdict={'weight': 'bold'})

## Old example plots of indiviudal HUs, using mu_arr = dict('rbm' : [0, 2]) etc
# sax = {}; iplot=0
# # time_slice = slice(0, 500)
# for i_dr, dr in enumerate(plot_methods):
#     for i_mu, mu in enumerate(mu_arr[dr]):
#         weighted_labels[dr] = freq_distr_weighted_regions(w_vector=weights[dr][mu, :], m_labels=plot_labels)
#         iplot += 1
#         sax[iplot] = plt.subplot(9, 5, int(i_dr + 3 + (i_mu*15)))
#     #         sax[iplot].plot(weighted_labels[dr], label=f'{dr_legend[dr]}, S={np.round(freq_entropy(weighted_labels[dr]), 2)}',
#     #                         linewidth=1, alpha=1, color=dr_colors[dr])
# #         sax[iplot].plot(low_dyn_test[dr][mu, time_slice], linewidth=1, color=dr_colors[dr])
# #         sax[iplot].get_yaxis().set_visible(False); sax[iplot].get_yaxis().set_ticks([])
# #         sax[iplot].get_xaxis().set_visible(False); sax[iplot].get_xaxis().set_ticks([])
#     #     plt.xlabel('Region id'); #plt.ylabel('P(region)'); plt.title(f'Example P(region) for HU {mu}')
#         if i_dr == 1 and i_mu == 0:
#             plt.title('Example components and time traces')

def plot_one_stat(density, xbins, ybins, title='', ax=None):

    if ax is None:
        ax = plt.subplot(111)
    image = ax.imshow(np.log10(density),
#                              interpolation='nearest', # do not use. (needed for high number of data points, but messes up pdf)
                                cmap='mako', origin='low', #vmin=-20, vmax=0,
               extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]], aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(image, cax=cax)  # https://stackoverflow.com/questions/46314646/change-matplotlib-colorbar-to-custom-height
    x_ticks = ax.get_xticks()[1:-1]  # first and last one are not shown
    ax.set_yticks(x_ticks)
    ax.set_xticks(x_ticks)
    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)
#     plt.plot(xbins[1:], density.sum(1))
#     plt.plot(xbins[1:], density.sum(0))

    ax.set_xlabel(f'Experimental data');
    ax.set_ylabel(f'Model-generated data')
    cax.set_ylabel('10-log PDF')
    ax.set_title(title)
    return ax

def plot_connectivity_matrix(matrix, region_names, size=15, subset=range(72),
                             cbar=False, cax=None, cmap='jet',
                             scale_limits=True, ax=None, region_order_dict=None,
                             plot_log=False, reverse_x=True, plot_labels=True,
                             mini=None, maxi=None, fill_zeros_with_min=False,
                             fill_nans_with_min=False):
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots()
        fig.set_figheight(size)
        fig.set_figwidth(size)

    else:
        return_fig = False

    matrix = matrix.copy()
    if region_order_dict is not None:
        sorted_region_inds, sorted_region_names = [], []
        # matrix = matrix[sorted_region_inds, :][:, sorted_region_inds]
        tmp_switch_old_new = np.zeros_like(region_order_dict['inds'])
        for old_ind, new_ind in enumerate(region_order_dict['inds']):  # eg 0, 70 => arf (0 alphabet) is at 70 (kunst ea)
            tmp_switch_old_new[new_ind] = old_ind
        for new_ind, old_ind in enumerate(tmp_switch_old_new):
            if old_ind in subset: ## only select region that are in subset
                sorted_region_inds.append(old_ind)
                sorted_region_names.append(region_order_dict['names'][new_ind])
                # sorted_region_names.append(region_names[old_ind])
        matrix_use = matrix[np.array(sorted_region_inds), :][:, np.array(sorted_region_inds)]
        region_names_use = sorted_region_names
        # print(region_names_use)
    else:
        matrix_use = matrix[subset, :][:, subset]
        region_names_use = [region_names[x] for x in subset]

    if scale_limits and (mini is None or maxi is None):  # automatically set mini and/or maxi if needed
        assert matrix_use.min() >= 0
        if plot_log:
            mini = np.percentile(np.log10(matrix_use[matrix_use > 0]), 10)
            maxi = np.percentile(np.log10(matrix_use[matrix_use > 0]), 99.5)
            # maxi = np.max(np.log10(matrix_use[matrix_use > 0]))
        else:
            mini = np.percentile(matrix_use[matrix_use > 0], 1)
            maxi = np.percentile(matrix_use[matrix_use > 0], 99)

        # for i_scale in range(2):  # need the loop because we need to add mini to matrix andd then re estimate in case of log
        #     tmp = matrix_use[np.logical_not(np.isnan(matrix_use))]
        #     tmp = tmp[np.where(tmp != 0)]  # because these ruin logs
        #     if mini is None:
        #         mini = np.percentile(tmp, 10)
        #         if i_scale == 0:  # add for log taking
        #             matrix_use += 0.1 * mini
        #             mini = None # for next it
        #     if maxi is None and i_scale == 1:
        #         maxi = np.percentile(tmp, 95)
        #     if i_scale == 0:
        #         if plot_log:
        #             matrix_use = np.log10(matrix_use)
    # else:
    if fill_zeros_with_min:
        assert np.nanmin(matrix_use) >= 0, f'min matrix is {np.nanmin(matrix_use)}'
        matrix_use[np.where(matrix_use == 0)] = np.nanmin(matrix_use[matrix_use > 0])

    if fill_nans_with_min:
        assert np.nanmin(matrix_use) >= 0, f'min matrix is {np.nanmin(matrix_use)}'
        matrix_use[np.isnan(matrix_use)] = np.nanmin(matrix_use[matrix_use > 0])

    if plot_log:
        matrix_use = np.log10(matrix_use)

    if cmap == 'sparsity':
        cmap_matrix = sns.cubehelix_palette(start=0.25, rot=-1, dark=0., light=0.9, reverse=True, as_cmap=True, n_colors=10)
        # cmap_matrix = sns.cubehelix_palette(start=0.25, rot=-1, dark=0.25, light=1, reverse=True, as_cmap=True, n_colors=10)  # used in sparsity fig
    else:
        cmap_matrix = cmap
    im = ax.imshow(matrix_use, vmin=mini, vmax=maxi, cmap=cmap_matrix)
    if plot_labels:
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(region_names_use, rotation=90, fontsize=6)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels(region_names_use, rotation=0, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5);
    for name_spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[name_spine].set_visible(False)


    if cbar:
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.1)
        plt.colorbar(im, cax=cax)  # https://stackoverflow.com/questions/46314646/change-matplotlib-colorbar-to-custom-height
        cax.set_ylabel('region to region connectivity strength\n(10-log scale)')
    if reverse_x:
        ax.invert_xaxis()
    if return_fig:
        return fig
    else:
        return ax

def plot_distr(mat, min_th=1e-8):
    fig = plt.figure(figsize=[12, 5])
    ax_1 = fig.add_subplot(121)
    ax_log = fig.add_subplot(122)

    # mat = mat[np.where(mat != 0)]
    # mat = mat[np.where(mat < 0.0001)]
    mat = mat[np.where(mat > min_th)]
    ax_1.hist(mat.flatten(), bins=50)

    ax_log.hist(np.log10(mat[np.where(mat != 0)].flatten()), bins=50)

def plot_funct_vs_struct(struct_mat, funct_mat, subset=np.arange(72), ax=None,
                         key=None, compute_in_log=False, fill_diag=False,
                         filter_zeros=False, set_limits=True, title='general',
                         use_one_triangle=True):
    if ax is None:
        ax = plt.subplot(111)
    assert struct_mat.shape == funct_mat.shape
    struct_mat = struct_mat[subset, :][:, subset]
    funct_mat = funct_mat[subset, :][:, subset]

    if fill_diag:
        struct_mat[np.arange(len(struct_mat)), np.arange(
            len(struct_mat))] += 1.0 * struct_mat.max()  # They did not define a diagonal connexion (intra-region connectivity). Set to the maximum value across all entry.

    if use_one_triangle:
        tri_inds = np.triu_indices_from(struct_mat, k=0)
        assert (np.unique(struct_mat) == np.unique(struct_mat[tri_inds])).all()
        struct_mat = struct_mat[tri_inds]
        funct_mat = funct_mat[tri_inds]

    struct_mat = struct_mat.flatten()
    funct_mat = funct_mat.flatten()

    if filter_zeros:
        print('before zero filtering ', funct_mat.shape)

        assert len(np.where(funct_mat == 0)[0]) == 0, np.where(funct_mat == 0)
        # inds = np.where(funct_mat > 0)  # not really needed be
        # print(len(inds), inds[0].shape)
        # funct_mat  = funct_mat[inds]
        # struct_mat = struct_mat[inds]

        inds = np.where(struct_mat > 0)
        funct_mat  = funct_mat[inds]
        struct_mat = struct_mat[inds]

        print('after zero filtering ', funct_mat.shape)

    if set_limits:
        # x_lim = (np.percentile(struct_mat, 10), np.percentile(struct_mat, 99.5))
        # y_lim = (np.percentile(funct_mat, 10), np.percentile(funct_mat, 99.5))
        x_lim = (0.9 * np.min(struct_mat), np.max(struct_mat) * 1.1)
        y_lim = (0.9 * np.min(funct_mat), funct_mat.max() * 1.1)
        # y_lim = (0.9 * np.sort(funct_mat)[4], funct_mat.max() * 1.1)
    #
    if compute_in_log:
        struct_mat_corr = np.log10(struct_mat)
        funct_mat_corr = np.log10(funct_mat)
    else:
        struct_mat_corr = struct_mat
        funct_mat_corr = funct_mat

    pearson, p_val_pearson = scipy.stats.pearsonr(struct_mat_corr, funct_mat_corr)
    spearman, p_val_spearman = scipy.stats.spearmanr(struct_mat_corr, funct_mat_corr)
    print(key, 'Pearson:' , pearson, ', Spearman: ', spearman)
    if key.lower() in dr_colors.keys():
        color_dots = dr_colors[key.lower()]
    else:
        color_dots = 'grey'
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.scatter(struct_mat, funct_mat, color=color_dots, s=10)
    # sns.jointplot(struct_mat, funct_mat, ax=ax)
    ax.set_xlabel('Structural connectivity\n(10-log scale)')
    ax.set_ylabel('Functional connectivity\n(10-log scale)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title == 'specific':
        print(p_val_spearman)
        sci_not = np.format_float_scientific(p_val_spearman, precision=3)
        sci_exp_ceil = int(sci_not.split('e')[1]) + 1
        ax.set_title(f'{dr_names[key]} vs structural connectivity\n' + r'$\mathregular{r_S}$' + f' = {np.round(spearman, 2)}, P < 10^{sci_exp_ceil}',
                        fontdict={'weight': 'bold'})
    else:
        ax.set_title(f'Structural vs functional connectivity\nr = {np.round(spearman, 2)}',
                     fontdict={'weight': 'bold'})
    if set_limits:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    return (pearson, spearman)

def plot_multi_fish_connectivity_scatter(all_connections_tensor, fish_combinations=[(0, 1)],
                                         ax=None, axis_lim=None, one_indexing_labels=True):
    if ax is None:
        ax = plt.subplot(111)
    plot_dens_1 = np.array([])  # save x axis
    plot_dens_2 = np.array([])  # save y axis
    plot_colours = np.array([])

    for i_plot, fish_inds in enumerate(fish_combinations):  # collect data
        tmp_dens_1 = np.log10(all_connections_tensor[: ,:, fish_inds[0]] + 1e-20).flatten()
        tmp_dens_2 = np.log10(all_connections_tensor[: ,:, fish_inds[1]] + 1e-20).flatten()
        plot_dens_1 = np.concatenate((plot_dens_1, tmp_dens_1))
        plot_dens_2 = np.concatenate((plot_dens_2, tmp_dens_2))
        plot_colours = np.concatenate((plot_colours, np.zeros_like(tmp_dens_1) + i_plot))  # int index for oclour
    random_inds = np.random.choice(a=len(plot_colours), size=len(plot_colours), replace=False) # shuffle so points are mixed

    ax.scatter(plot_dens_1[random_inds], plot_dens_2[random_inds], c=plot_colours[random_inds],
                alpha=0.8, s=12)
    ax.set_xlabel('Region to region connectivity\n in Fish A (10-log scale)');
    ax.set_ylabel('Region to region connectivity\n in Fish B (10-log scale)');
    if axis_lim is not None:
        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
        if axis_lim == [-6, -1.5]:
            ax.set_xticks(np.arange(-6, -1))
            ax.set_yticks(np.arange(-6, -1));

    viridis = matplotlib.cm.get_cmap('viridis', len(fish_combinations))  # this is defaul tin scatter
    handle_dict = [matplotlib.lines.Line2D([], [], marker='o',
                                           c='white', markerfacecolor=viridis(i_plot)) for i_plot in range(len(fish_combinations))]
    if one_indexing_labels:
        legend_labels = [f'{str(x[0] + 1)}-{str(x[1] + 1)}' for x in fish_combinations]
    else:
        legend_labels = [f'{str(x[0])}-{str(x[1])}' for x in fish_combinations]
    ax.legend(handles=handle_dict,
              # labels=[f'A: #{str(x[0])}, B: #{str(x[1])}' for x in fish_combinations],
              labels=legend_labels,
              frameon=False, loc='upper left', bbox_to_anchor=(-0.08, 1.07));
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def plot_zero_vs_nz_connectivity(ax_pdf, ax_cdf, funct_mat, inds_struct_zero, inds_struct_nz,
                                 colour_nz='#00b188', colour_zero='#00392c', method='RBM',
                                 funct_nz={}, funct_zero={}, pdf_nz={}, pdf_zero={},
                                 cdf_nz={}, cdf_zero={}, verbose=1):
    funct_nz[method] = funct_mat[inds_struct_nz]
    funct_zero[method] = funct_mat[inds_struct_zero]
    ks_test = scipy.stats.ks_2samp(funct_nz[method].copy(),
                                   funct_zero[method].copy(), mode='exact')
    sci_not = np.format_float_scientific(ks_test[1], precision=3)
    sci_exp_ceil = int(sci_not.split('e')[1]) + 1
    if verbose > 0:
        print(ks_test)
    if verbose > 1:
        print(af.kstest(funct_nz[method], funct_zero[method]))

    plot_bins = np.linspace(np.min(funct_mat[~np.isnan(funct_mat)]),
                            np.max(funct_mat[~np.isnan(funct_mat)]), 100)  # fix bins

    pdf_nz[method], plot_bins = np.histogram(funct_nz[method], bins=plot_bins, density=True)
    pdf_zero[method], plot_bins = np.histogram(funct_zero[method], bins=plot_bins, density=True)

    cdf_nz[method] = np.cumsum(pdf_nz[method]) / np.max(np.cumsum(pdf_nz[method]))
    cdf_zero[method] = np.cumsum(pdf_zero[method]) / np.max(np.cumsum(pdf_zero[method]))

    plot_bins_centered = (plot_bins[1:] + plot_bins[:-1]) / 2

    distance = (plot_bins_centered[int(np.argmin(np.abs(cdf_zero[method] - 0.5)))] -
                plot_bins_centered[int(np.argmin(np.abs(cdf_nz[method] - 0.5)))])
    width_zero = (plot_bins_centered[int(np.argmin(np.abs(cdf_zero[method] - 0.75)))] -
                  plot_bins_centered[int(np.argmin(np.abs(cdf_zero[method] - 0.25)))])
    width_nz = (plot_bins_centered[int(np.argmin(np.abs(cdf_nz[method] - 0.75)))] -
                  plot_bins_centered[int(np.argmin(np.abs(cdf_nz[method] - 0.25)))])
    print(method, distance / (0.5 * (width_nz + width_zero)))

    ax_pdf.plot(plot_bins_centered, pdf_nz[method],
                       label='Non-zero', c=colour_nz)
    ax_pdf.plot(plot_bins_centered, pdf_zero[method],
                       label='Zero', c=colour_zero)

    ax_cdf.plot(plot_bins_centered,
                       cdf_nz[method],
                       label='Non-zero', c=colour_nz)
    ax_cdf.plot(plot_bins_centered,
                       cdf_zero[method],
                       label='Zero', c=colour_zero)

    ax_cdf.text(s=f'P < 10^{sci_exp_ceil}', x=plot_bins_centered[int(np.argmin(np.abs(cdf_zero[method] - 0.5)))] - 0.2, y=0.5,
                       fontdict={'ha': 'right'})
    ax_pdf.set_title(f'{dr_names[method]}', fontdict={'weight': 'bold'})
    ax_cdf.set_xlabel('Functional connection\n(10-log scale)')
    ax_pdf.set_xlim([-7, 0])
    ax_cdf.set_xlim([-7, 0])
    for ax in [ax_pdf, ax_cdf]:
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
