import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from scipy.stats import spearmanr
import pandas as pd
import pickle
sys.path.append('../')
import seaborn as sns


version = '2'

env = pickle.load(open('/home/iscb/wolfson/jeromet/Data/Neuro/connectivity_plot_files_v%s.data'%version,'rb'))

runs = env['runs'] # List of experimental recordings used.


# Baier matrices.
nRegions_baier = env['nRegions_baier'] # 72
region_names_baier = env['region_names_baier'] # the region names, in the order displayed.

'''
There are various Baier matrices:
- his old version.
- his new version.
- the ones I rederived from his new data.
All viz are made with 'my_new_normalized_by_volume', but we can change at last stage if needed.
'''
selected_connectivity_type = 'my_new_normalized_by_volume'
connectivity_baier = env['connectivity_baier'] # The selected one.
all_baier_connectivities = env['all_baier_connectivities'] # All others.
connectivity_types = list(all_baier_connectivities.keys()) 

nNeurons_per_region = env['nNeurons_per_region'] # Number of neurons / region / recording.
mean_nNeurons_per_region = env['mean_nNeurons_per_region'] # Averaged number of neurons (across all recordings)


all_functional_connectivities = env['all_functional_connectivities']  # All functional connectivities inferred. For each run, each method, and each aggregation (L1 or L2 norm), one 72X72 matrix.

all_averaged_functional_connectivities = env['all_averaged_functional_connectivities'] # All functional connectivities inferred. For each run, each method, and each aggregation (L1 or L2 norm), one 72X72 matrix.
all_neuron_connectivities = env['all_neuron_connectivities']
methods = env['methods']


averaging_weights = env['averaging_weights'] 
# The tensor of pair-weights used for producing the average functional connectivity (of size Nrecordings X Nregions X Nregions).


aggregations =[1,2]
selected_aggregation = 1 # Use L1 norm for the averaged.


output_folder = 'plots_v%s/'%version
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


'''
Determine "subset", the subset of regions onto which we perform analysis.
Here: At least 5 neurons on average for both the left and right region.
'''
Nmin = 5
subset = np.zeros(nRegions_baier, dtype='bool')
for i in range(nRegions_baier // 2):
    if (mean_nNeurons_per_region[2 * i] >= Nmin) & (mean_nNeurons_per_region[2 * i + 1] >= Nmin):
        subset[2 * i] = 1
        subset[2 * i + 1] = 1
subset = np.nonzero(subset)[0]

'''
Determine the recording-level subset of regions. Useful for pairwise comparison and recording visualization.
'''
nNeurons_per_region_symmetrical = np.concatenate([np.concatenate([nNeurons_per_region[2*i+1:2*i+2,:],nNeurons_per_region[2*i:2*i+1]],axis=0) for i in range(nRegions_baier//2)],axis=0)
relevant_regions = np.zeros(nNeurons_per_region.shape,dtype=bool)
relevant_regions[subset,:] = (nNeurons_per_region[subset,:]>=Nmin) & (nNeurons_per_region_symmetrical[subset,:]>=Nmin)
relevant_regions2 = relevant_regions[:,np.newaxis] * relevant_regions[np.newaxis,:,:]    
nRuns = len(runs)    
all_masks = relevant_regions2.reshape([nRegions_baier**2,nRuns]).T 



make_run_plots = True
make_method_plots = True
make_neuron_sparsity_plots = True
make_region_sparsity_plots = True
make_information_file = True
make_reproducibility_numbers = True




# %% Make mapping information file.
if make_information_file:  # %% Make mapping information file.
    mapping_information_file = output_folder +'baier_mapping_information_%s.txt'%version
    with open(mapping_information_file, 'w') as f:
        f.write('----------------- Neurons per Region --------------------\n')
        for i in range(nRegions_baier):
            message = 'Region %s' % region_names_baier[i]
            if i in subset:
                message += '(Included)'
            else:
                message += '(Not included)'
            message += ' nNeurons:'
            for k, run in enumerate(runs):
                message += '%s ' % nNeurons_per_region[i, k]
            message += '\n'
            f.write(message)

        f.write('----------------- %s Selected Regions (Nmin = %s) -------------------\n' %
                (len(subset), Nmin))
        for i in subset:
            f.write('Region %s, <nNeurons>: %.2f\n' %
                    (region_names_baier[i], mean_nNeurons_per_region[i]))


#%% Main text information. Inter-specimen correlation of connectivity matrices.
if make_reproducibility_numbers:
    def show_heatmap(matrix,labels=None):
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(9,8))
        df=pd.DataFrame(matrix,index=labels,columns=labels) 
        sns.heatmap(df,annot=True,ax=ax)
        return fig,ax
        


    def write_correlation(C,file,runs=None):
        '''
        import numpy as np
        C = np.random.rand(4,4)
        runs = ['abc','def','hij','klm']
        file = 'tmp.txt'
        write_correlation(C,file,runs=runs)
        '''
        nruns = C.shape[0]
        with open(file,'a') as f:
            if runs is not None:
                for k,run in enumerate(runs):
                    f.write('Run %s: %s \n'%(k,run))

            header = '  '
            for k in range(nruns):
                header += ' %s   '%k
            header += '\n'
            f.write(header)
            for k in range(nruns):
                format = '%s '%k+ '%.2f ' * nruns + '\n'
                c = tuple(C[k])
                f.write(format%c )
        return

    reproducibility_numbers = output_folder +'connectivity_reproducibility_%s.txt'%version

    # Make region filters.
    all_correlations = {}   
    for method in methods:
        correlation = np.zeros([nRuns,nRuns])
        for k1,run1 in enumerate(runs):
            for k2,run2 in enumerate(runs):
                key1 = 'run:%s_method:%s_agg:%s'%(run1,method,selected_aggregation)
                key2 = 'run:%s_method:%s_agg:%s'%(run2,method,selected_aggregation)
                connections1 = all_functional_connectivities[key1].flatten()
                connections2 = all_functional_connectivities[key2].flatten()
                mask = all_masks[k1] & all_masks[k2]
                correlation[k1,k2] = np.corrcoef(connections1[mask],connections2[mask])[0,1]
                # correlation[k1,k2] = (connections1[mask]*connections2[mask]).mean()/np.sqrt( (connections1[mask]**2).mean() * (connections2[mask]**2).mean() )
        all_correlations[method] = correlation

    for method in methods:
        fig,ax = show_heatmap(all_correlations[method],labels=runs)
        ax.set_title('Inter-specimen correlation of connectivity maps (%s)'%method,fontsize=14)
        plt.tight_layout()
        fig.savefig(output_folder + 'interfish_correlation_connectivity_%s_%s.png'%(method,version),dpi=300)

    upper_diagonal = (np.arange(len(runs))[:,np.newaxis] > np.arange(len(runs))[np.newaxis,:])
    flat_correlations = [all_correlations[method][upper_diagonal] for method in methods]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist(flat_correlations,label=methods,bins=20,histtype='step')
    ax.legend(fontsize=14,loc='upper left')
    ax.set_xlabel('Inter-Fish Pearson correlation',fontsize=14)
    fig.savefig(output_folder + 'interfish_correlation_connectivity_histo_%s.png'%version,dpi=300)


if (make_run_plots| make_method_plots):
    def show_connectivity(matrix, region_names, size=15, subset=range(72), ax=None):
        if ax is None:
            return_fig = True
            fig, ax = plt.subplots()
            fig.set_figheight(size)
            fig.set_figwidth(size)

        else:
            return_fig = False
        matrix = matrix.copy()
        tmp = matrix[subset, :][:, subset]
        off_diagonal = 1 - np.eye(len(subset))
        off_diagonal = off_diagonal.astype(np.bool)
        tmp = tmp[off_diagonal]
        maxi = 3 * np.sqrt(np.nanmean(tmp**2))
        ax.imshow(matrix[subset, :][:, subset], vmin=0, vmax=maxi)
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels([region_names[x] for x in subset], rotation=90)
        ax.set_yticks(range(len(subset)))
        ax.set_yticklabels([region_names[x] for x in subset], rotation=0)
        if return_fig:
            return fig

    all_spearman = {}
    all_pearson = {}
    all_spearman_averaged = {}
    all_pearson_averaged = {}
    connectivity_key_format = 'run:{run}_method:{method}_agg:{aggregation}'
    correlation_key_format = 'run:{run}_method:{method}_agg:{aggregation}_conn:{connectivity_type}'
    averaged_connectivity_key_format = 'method:{method}_agg:{aggregation}'
    correlation_averaged_connectivity_key_format = 'method:{method}_agg:{aggregation}_conn:{connectivity_type}'

    # Compute spearman and pearson correlation.

    for k,run in enumerate(runs):
        for method in methods:
            for aggregation in aggregations:
                key_dict = connectivity_key_format.format(
                    run=run, method=method, aggregation=aggregation)
                prediction = all_functional_connectivities[key_dict].copy()
                for connectivity_type in connectivity_types:
                    target = all_baier_connectivities[connectivity_type].copy()
                    if connectivity_type in ['his_old', 'his_new']:
                        target[np.arange(len(target)), np.arange(
                            len(target))] += 1.0 * target.max()  # They did not define a diagonal connexion (intra-region connectivity). Set to the maximum value across all entry.

                    flat_target = target.flatten()[all_masks[k]]
                    flat_prediction = prediction.flatten()[all_masks[k]]
                    # flat_target = target[subset, :][:, subset].flatten()
                    # flat_prediction = prediction[subset,
                    #                              :][:, subset].flatten()
                    spearman = spearmanr(
                        flat_target, flat_prediction).correlation
                    pearson = np.corrcoef(
                        flat_target, flat_prediction)[0, 1]

                    key = correlation_key_format.format(
                        run=run, method=method, aggregation=aggregation, connectivity_type=connectivity_type)
                    all_spearman[key] = spearman
                    all_pearson[key] = pearson

    # Compute average spearman and pearson correlation.
    for connectivity_type in connectivity_types:
        for method in methods:
            for aggregation in aggregations:
                prediction = all_averaged_functional_connectivities[averaged_connectivity_key_format.format(
                    method=method, aggregation=aggregation)]
                target = all_baier_connectivities[connectivity_type].copy()
                if connectivity_type in ['his_old', 'his_new']:
                    target[np.arange(len(target)), np.arange(
                        len(target))] += 1.0 * target.max()

                flat_target = target[subset, :][:, subset].flatten()
                flat_prediction = prediction[subset,
                                             :][:, subset].flatten()

                spearman = spearmanr(
                    flat_target, flat_prediction).correlation
                pearson = np.corrcoef(flat_target, flat_prediction)[0, 1]

                key = correlation_averaged_connectivity_key_format.format(
                    method=method, aggregation=aggregation,connectivity_type=selected_connectivity_type)
                all_spearman_averaged[key] = spearman
                all_pearson_averaged[key] = pearson



# All methods, one matrix per run.
if make_run_plots:
    nRuns = len(runs)      
    for method in methods:
        figure_name = 'Connectivity_all_runs_method_%s.png' %(method)
        nrows = 2
        ncols = int(np.ceil(nRuns/nrows))
        fig, ax = plt.subplots(nrows, ncols)
        fig.set_figheight(  nrows*12 )
        fig.set_figwidth( ncols * 12)
        for k, run in enumerate(runs):
            i = k // ncols
            j = k % ncols
            key_matrix = 'run:%s_method:%s_agg:%s'%(run,method,selected_aggregation)

            key_matrix = connectivity_key_format.format(
                run=run, method=method, aggregation=selected_aggregation)
            key_correlation = correlation_key_format.format(
                run=run, method=method, aggregation=selected_aggregation, connectivity_type=selected_connectivity_type)
            prediction = all_functional_connectivities[key_matrix]
            prediction_ = prediction.copy()
            prediction_[~all_masks[k].reshape([nRegions_baier,nRegions_baier])] = np.nan

            spearman = all_spearman[key_correlation]
            pearson = all_pearson[key_correlation]
            show_connectivity(prediction_, region_names_baier,
                              size=15, subset=subset, ax=ax[i,j])
            ax[i,j].set_title('%s (%s %s), Spearman = %.3f, Pearson = %.3f' % (
                run, method, selected_aggregation, spearman, pearson), fontsize=20)
        plt.tight_layout()
        fig.savefig(output_folder+figure_name, dpi=300)
        plt.close()

# All methods, one matrix per averaging.
if make_method_plots:
    figure_name = 'Connectivity_averaged_all_methods.png'
    fig, ax = plt.subplots(1, len(methods) + 1)
    fig.set_figheight(12)
    fig.set_figwidth(12 * (len(methods) + 1))
    show_connectivity(
        all_baier_connectivities[selected_connectivity_type], region_names_baier, size=15, subset=subset, ax=ax[0])
    ax[0].set_title('Baier (%s)' % selected_connectivity_type, fontsize=20)
    for k, method in enumerate(methods):
        key_matrix = averaged_connectivity_key_format.format(
            method=method, aggregation=selected_aggregation)
        key_correlation = correlation_averaged_connectivity_key_format.format(
            method=method, aggregation=selected_aggregation,connectivity_type=connectivity_type)

        prediction = all_averaged_functional_connectivities[key_matrix]
        spearman = all_spearman_averaged[key_correlation]
        pearson = all_pearson_averaged[key_correlation]
        show_connectivity(prediction, region_names_baier,
                          size=15, subset=subset, ax=ax[k + 1])
        ax[k + 1].set_title('%s (%s), Spearman = %.3f, Pearson = %.3f' %
                            (method, selected_aggregation, spearman, pearson), fontsize=20)
    plt.tight_layout()
    fig.savefig(output_folder+figure_name, dpi=300)
    plt.close()


make_region_sparsity_plots = True
if make_region_sparsity_plots:
    figure_name = 'Distribution_region_connectivities.png'
    all_region_connectivities = []
    for k, method in enumerate(methods):
        averaged_connectivity_key_format = 'method:{method}_agg:{aggregation}'        
        key_matrix = averaged_connectivity_key_format.format(
            method=method, aggregation=selected_aggregation)
        region_connectivities = all_averaged_functional_connectivities[key_matrix][subset,:][:,subset].flatten()
        all_region_connectivities.append(region_connectivities)
    all_region_connectivities.append(connectivity_baier[subset,:][:,subset].flatten())

    all_region_connectivities = [flat_connectivity/np.sqrt( (flat_connectivity**2).mean()) for flat_connectivity in all_region_connectivities ]

    margin = 0.1
    maxi = 3
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(all_region_connectivities,log=False,histtype='step',bins=100,range=(0,maxi),
           label = (methods+['Baier']),linewidth=2.,normed=False,alpha=1.0)
    plt.xlabel('Inter-region connectivity (normalized)',fontsize=16)
    plt.legend(fontsize=16,frameon=False)
    ax.tick_params(which='both',axis='both',labelsize=16)
    ax.set_xlim([-margin,maxi+margin])
    plt.grid()    
    fig.savefig(output_folder+figure_name,dpi=300)
    plt.close()    


if make_neuron_sparsity_plots:
    check = False
    figure_name = 'Distribution_neuronal_connectivities.png'
    all_flat_connectivities = [all_neuron_connectivities['method:%s'%method] for method in methods]

    if check:
        flat_connectivities = [flat_connectivities[::100] for flat_connectivities in all_flat_connectivities]
    else:
        flat_connectivities = all_flat_connectivities
    flat_connectivities = [flat_connectivity/np.sqrt( (flat_connectivity**2).mean()) for flat_connectivity in flat_connectivities ]

    margin = 0.1
    maxi = 40
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist(flat_connectivities[::-1],log=True,histtype='step',bins=800,range=(-maxi,maxi),
           label = methods[::-1],linewidth=2.,normed=False,alpha=1.0)
    plt.xlabel('Inter-neuron connectivity (normalized)',fontsize=16)
    plt.legend(fontsize=16,frameon=False)
    ax.tick_params(which='both',axis='both',labelsize=16)
    ax.set_xlim([-maxi-margin,maxi+margin])
    plt.grid()    
    fig.savefig(output_folder+figure_name,dpi=300)
    plt.close()