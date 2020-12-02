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
                    hu_id=86, filename_prefix='screenshot_hu_', y_min=400, y_max=1600, fontsize=10):
    '''fontsize: 15 for (10, 10) fig, linearly scaling seems fine '''

    # screenshot_files = os.listdir(hu_screenshot_folder)
    filename = filename_prefix + str(hu_id).zfill(3) + '.png'
    image = mpimg.imread(os.path.join(hu_screenshot_folder, filename))

    if ax is None:
        ax = plt.subplot(111)
    ax.imshow(image[:, y_min:y_max, :])
    # ax.text(s=f'HU {hu_id}', x=100, y=100, c='white')
    ax.text(s='Ro', x=25, y=71, c='white', fontdict={'size': fontsize})
    ax.text(s='C', x=145, y=71, c='white', fontdict={'size': fontsize})
    ax.text(s='R', x=95, y=1, c='white', fontdict={'size': fontsize, 'va': 'top'})
    ax.text(s='L', x=95, y=124, c='white', fontdict={'size': fontsize})

    ax.text(s='Ro', x=25, y=571, c='white', fontdict={'size': fontsize})
    ax.text(s='C', x=145, y=571, c='white', fontdict={'size': fontsize})
    ax.text(s='D', x=95, y=501, c='white', fontdict={'size': fontsize, 'va': 'top'})
    ax.text(s='V', x=95, y=624, c='white', fontdict={'size': fontsize})
    ax.text(s='0.1 mm', x=1000, y=840, c='white', fontdict={'size': fontsize})

    ax.set_title(f'Hidden unit {hu_id}', fontdict={'weight': 'bold'})
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
                      bar_width=None, cutoff=8, normalise=False):
    assert dr in degree_dict.keys(), f'{dr} not in degree_dict'
    assert degree_dict[dr].min() == 0 and degree_dict_sh[dr].min() == 0  # make life easy
    if ax is None:
        ax = plt.subplot(111)


    max_degree = degree_dict[dr].max()
    max_degree_sh = degree_dict_sh[dr].max()

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
    ax.bar(x=x_degrees, height=bar_heights, width=bar_width, color=dr_colors[dr],
           label=dr_legend[dr])
    if plot_shuffled:
        ax.bar(x=degree_values_sh + bar_width / 2, height=bar_heights_sh,
               width=bar_width, color=dr_colors[dr], alpha=0.4,
               label='shuffled '+ dr_legend[dr])

    if plot_shuffled:
        ax.legend(loc='upper right', frameon=False)
    ax.set_xticks(degree_values)
    xlabels = [str(x) for x in degree_values]
    xlabels[-1] = f'>{xlabels[-2]}'
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Degree of connections to hidden layer')
    if normalise:
        ax.set_ylabel('PDF')
    else:
        ax.set_ylabel('Frequency')
    ax.set_title(f'Connectivity to hidden layer {dr_legend[dr]}', fontdict={'weight': 'bold'})
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

def plot_reproduc_mat(dict_mat, key, ax=None):
    if ax is None:
        ax = plt.subplot(111)
    data = dict_mat[key]['mat']
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(data, k=0)] = True
    sns.heatmap(data, annot=True, ax=ax, square=True, mask=mask, vmin=0, vmax=1,
                cmap='pink_r', cbar_kws={'label': 'Pearson correlation'})
    ax.set_xlabel('# Fish')
    ax.set_ylabel('# Fish')
    ax.set_xticks(np.arange(len(data) - 1) + 0.5)
    ax.set_yticks(np.arange(1, len(data)) + 0.5)
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
    image = ax.imshow(np.log(density),
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
    ax.set_title(title)
    return ax
