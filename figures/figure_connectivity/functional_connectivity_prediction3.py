import os
num_threads = 4
os.environ["MKL_NUM_THREADS"] = "%s" % num_threads
os.environ["NUMEXPR_NUM_THREADS"] = "%s" % num_threads
os.environ["OMP_NUM_THREADS"] = "%s" % num_threads
os.environ["OPENBLAS_NUM_THREADS"] = "%s" % num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "%s" % num_threads
os.environ["NUMBA_NUM_THREADS"] = "%s" % num_threads

import numpy as np
import sys
import pandas as pd
import pickle
sys.path.append('../')

PGM3_location = '../../../PGM3/'
sys.path.append(PGM3_location)
sys.path.append(PGM3_location+'source/')
sys.path.append(PGM3_location+'utilities/')
from numba import prange, njit
import rbm
import plm_inference





# Compute effective coupling matrix from RBM.
def RBM_to_effective_interactions(RBM, data, weights=None):
    I = RBM.input_hiddens(data)
    var = RBM.hlayer.var_from_inputs(I)
    mean_var = var.mean(0)
    J_eff = 0.5 * np.dot(RBM.weights.T, mean_var[:, np.newaxis] * RBM.weights)
    J_eff[np.arange(RBM.n_v), np.arange(RBM.n_v)] *= 0
    return J_eff


def infer_functional_couplings(spikes, method,n_h=200,l1=0.02,lr=1e-3,batch_size=100):
    print('Im inside %s'%method)
    nNeurons = spikes.shape[1]
    if method == 'covariance':
        couplings = np.cov(1*spikes.T)
    elif method == 'correlation':
        couplings = np.corrcoef(1*spikes.T)
    elif method == 'PLM':
        if check:
            epochs = 1
            spikes = spikes[:32]
        else:
            epochs = 20
        couplings = plm_inference.infer_plm_network(spikes,epochs=epochs,chunk_size=1000,logfile='plm_%s.txt'%run,gpu=False)
    elif method == 'RBM':
        RBM_locations = [RBM_location_format.format(run=run,hidden=hidden,n_h=n_h,l1=l1,repeat=repeat) for repeat in repeated_trainings_RBMs[run]]
        all_RBMs = []
        for RBM_location in RBM_locations:
            if os.path.exists(RBM_location):
                RBM = pickle.load(open(RBM_location,'rb'))
                all_RBMs.append(RBM)
            else:
                if train_missing_RBM:
                    n_iter = (n_updates // (spikes.shape[0] // batch_size))
                    RBM = rbm.RBM(n_v=spikes.shape[1],n_h=n_h,hidden=hidden,visible='Bernoulli')
                    RBM.fit(spikes, verbose=0, learning_rate=lr, lr_final=lr*1e-2, extra_params = [0,0.999,1e-6],
                    decay_after=decay_after, n_iter=n_iter, N_MC=N_MC, l1=l1, vverbose=1,batch_size=batch_size)
                    pickle.dump(RBM,open(RBM_location,'wb'))
                    all_RBMs.append(RBM)
                else:
                    continue

        assert make_only_RBM_training == False,'Only training, exiting script now'
        print('Successfully loaded RBM')
        if RBM_average:
            couplings = np.zeros([nNeurons, nNeurons],dtype=np.float32)
            nTrainings = 0
            for k,RBM in enumerate(all_RBMs):
                print('Inferring couplings (%s repeat %s)'% (run,k) )
                tmp = RBM_to_effective_interactions(RBM, spikes)
                count_outliers1 = (np.abs(RBM.weights)>10).sum()                
                count_outliers2 = (np.abs(tmp)>5).sum()
                std_weights = RBM.weights.flatten().std()
                # if (count_outliers1 > 10) | (count_outliers2>10):
                if not ((std_weights < 0.1) & (std_weights>0.01) ):
                    print('Repeat %s %s does not seem to have converged' % (run,k) )
                else:
                    couplings+= tmp
                    nTrainings += 1
            couplings /= nTrainings
        # else:
        #     if os.path.exists(RBM_selection_location):
        #         best_model = pickle.load(open(RBM_selection_location,'rb'))['best_model']
        #     else:
        #         all_metrics = []
        #         for RBM in all_RBMs:
        #             all_metrics.append(RBM_model_selection.compute_train_metrics(RBM,spikes) )
        #         all_metrics = np.array(all_metrics)
        #         order = np.argsort(all_metrics,axis=0)
        #         all_ranks = np.zeros(all_metrics.shape,dtype=np.int)
        #         for column in range(all_metrics.shape[1]):
        #             for rank in range(all_metrics.shape[0]):
        #                 all_ranks[order[rank,column],column] = rank
        #         mean_rank = all_ranks.mean(1)
        #         best_model = np.argmax(mean_rank)
        #         env = {'all_metrics':all_metrics,'all_ranks':all_ranks,'best_model':best_model}
        #         pickle.dump(env,open(RBM_selection_location,'wb'))
        #
        #     assert make_only_RBM_selection == False,'Only selection, exiting script now'
        #
        #     best_RBM = all_RBMs[best_model]
        #     couplings = RBM_to_effective_interactions(best_RBM, spikes)
    else:
        print('%s not implemented' % method)
        couplings = np.zeros([nNeurons, nNeurons])

    couplings[np.isnan(couplings)] = 0  # Remove nans.
    # Set diagonal elements to zero.
    couplings[range(nNeurons), range(nNeurons)] *= 0
    return couplings



def load_baier_matrix(location_matrix='atlas_data/connectivity_matrix.csv', location_names='atlas_data/list_36_region_names.txt'):
    with open(location_names, 'rb') as fp:
        region_names_baier = pickle.load(fp)  # load region names
    # Add R/L distinction:
    tmp = []
    for rn in region_names_baier:
        tmp.append(rn + '_left')
        tmp.append(rn + '_right')
    region_names_baier = tmp

    regions = np.arange(len(region_names_baier))
    df_struct_baier = pd.read_csv(location_matrix, header=None)
    connectivity_baier = df_struct_baier.values[1:, 1:]
    connectivity_baier = connectivity_baier.astype('float')
    connectivity_baier = connectivity_baier[regions, :][:, regions]
    return connectivity_baier, region_names_baier


@njit(parallel=True)
def aggregate_couplings(M, partition, npartition, power=1):
    nNeurons = M.shape[0]
    Magg = np.zeros((npartition, npartition))
    count = np.zeros((npartition, npartition))
    for i1 in prange(nNeurons):
        for i2 in prange(nNeurons):
            p1 = partition[i1]
            p2 = partition[i2]
            if i1 > i2:
                if (p1 >= 0) & (p2 >= 0):
                    Magg[p1, p2] += np.abs(M[i1, i2])**power
                    count[p1, p2] += 1
    Magg += Magg.T
    count += count.T
    return (Magg / (count + 1e-4))**(1.0 / power)





# Paths to spikes recordings and Baier atlas labels.
spike_datasets_folder = '/specific/netapp5_2/iscb/wolfson/jeromet/Data/Neuro/spontaneous_zebrafish_spiking_only/'
labels_datasets_folder = spike_datasets_folder + 'baier_atlas_labels/'
connectivity_matrices_folder = 'baier_functional_connectivity/'
plot_folder = 'plots/'
trainings_folder = '/specific/netapp5_2/iscb/wolfson/jeromet/Data/Neuro/RBM/'
outputs_folder = 'predicted_functional_connectivity/'
storage_folder = '/specific/netapp5_2/iscb/wolfson/jeromet/Data/Neuro/functional_connectivity/'


RBM_location_format = storage_folder + 'RBM_{run}_hidden_{hidden}_nh_{n_h}_l1_{l1:.2e}_repeat_{repeat}.data'
RBM_selection_location_format = storage_folder + 'selection_RBM_{run}_hidden_{hidden}_nh_{n_h}_l1_{l1:.2e}.data'
connectivities_location_format = storage_folder + 'connectivities_run:{run}_method:{method}.npy'


for folder in [outputs_folder, storage_folder]:
    if not os.path.isdir(folder):
        os.mkdir(folder)


runs = [
    '20180706_Run04',
    '20180911_Run01',
    '20180912_Run01',
    '20180913_Run01',
    '20181218_Run02',
    '20190109_Run04',
    '20190502_Run04',
    '20181206_Run03',
    '20190102_Run01',
    '20190130_Run03',
    '20190503_Run04',
    '20181206_Run05',
    '20190108_Run05',
    '20190425_Run04',
    '20191126_Run02' ]


# runs = [
#     '20180706_Run04',
#     '20180911_Run01',
#     '20180912_Run01',
#     '20180913_Run01',
#     '20181218_Run02',    
#     '20181206_Run03',
#     '20190102_Run01',
#     '20181206_Run05',
#     ]


# runs = [
#     '20180706_Run04',
#     '20180911_Run01',
#     '20180912_Run01',
#     '20180913_Run01',
#     '20181206_Run03',
#     '20190102_Run01',
#     '20181206_Run05',
#     ]

runs = [
    '20180706_Run04',
    '20180911_Run01',
    '20180912_Run01',
    '20180913_Run01',
    '20181218_Run02',
    '20190109_Run04',
#    '20190502_Run04',
    '20181206_Run03',
    '20190102_Run01',
#    '20190130_Run03',
#    '20190503_Run04',
    '20181206_Run05',
#    '20190108_Run05',
#    '20190425_Run04',
#    '20191126_Run02'
    ]


methods = ['RBM','covariance', 'correlation']#,'PLM','RBM']
#methods = ['RBM']#,'PLM','RBM']
aggregations = [1,2]  # L1 or L2 norm to be taken.
selected_connectivity_type = 'my_new_normalized_by_volume' # for plots only
# selected_averaging_method = 'recording_length+product_nNeurons' # for plots only
selected_averaging_method = 'recording_length+sum_nNeurons' # for plots only
selected_run = '20180912_Run01' # for plots only


make_only_RBM_training = False
RBM_average = True
train_missing_RBM = False


check = False

if check:
    repeated_trainings_RBM = range(2)
    n_h_selected = 5
    l1_selected = 0.01
    hidden = 'dReLU'
    n_updates = 100
    N_MC = 1
    lr = 1e-3
    lr_final = 1e-5
    decay_after = 0.5

else:
    # repeated_trainings_RBM = range(5)
    # repeated_trainings_RBMs =    {
    # '20180706_Run04':range(5),
    # '20180911_Run01':range(5),
    # '20180912_Run01':range(5),
    # '20180913_Run01':range(6,9),
    # '20181218_Run02':range(2),
    # '20190109_Run04':range(2),
    # '20190502_Run04':range(2),
    # '20181206_Run03':range(2),
    # '20190102_Run01':range(2),
    # '20190130_Run03':range(2),
    # '20190503_Run04':range(2),
    # '20181206_Run05':range(2),
    # '20190108_Run05':range(2),
    # '20190425_Run04':range(2),
    # '20191126_Run02':range(2) }



    # repeated_trainings_RBMs =    {
    # '20180706_Run04':range(5),
    # '20180911_Run01':range(5),
    # '20180912_Run01':range(5),
    # '20180913_Run01':range(6,9),
    # '20181218_Run02':range(4),
    # '20190109_Run04':range(2,4),
    # '20190502_Run04':range(2,4),
    # '20181206_Run03':range(4),
    # '20190102_Run01':range(2,4),
    # '20190130_Run03':range(2,4),
    # '20190503_Run04':range(2,4),
    # '20181206_Run05':range(2,4),
    # '20190108_Run05':range(2,4),
    # '20190425_Run04':range(2,4),
    # '20191126_Run02':range(2,4) }

    # repeated_trainings_RBMs =    {
    # '20180706_Run04':range(5),
    # '20180911_Run01':range(5),
    # '20180912_Run01':range(5),
    # '20180913_Run01':range(6,9),
    # '20181218_Run02':range(4),
    # '20190109_Run04':range(4,6),
    # '20190502_Run04':range(4,6),
    # '20181206_Run03':range(4),
    # '20190102_Run01':range(4,6), # Ou range(2) et 200.
    # '20190130_Run03':range(4,6),
    # '20190503_Run04':range(4,6),
    # '20181206_Run05':range(2,4),
    # '20190108_Run05':range(4,6), # Eval: range(2,6)
    # '20190425_Run04':range(4,6),
    # '20191126_Run02':range(4,6) }

    repeated_trainings_RBMs =    {
    '20180706_Run04':range(6,9),
    '20180911_Run01':range(5),
    '20180912_Run01':range(5),
    '20180913_Run01':range(6,9),
    '20181218_Run02':range(4),
    '20190109_Run04':range(6,9),
    '20190502_Run04':range(4,6),
    '20181206_Run03':range(4),
    '20190102_Run01':range(6,9),#range(4,6), # Ou range(2) et 200.
    '20190130_Run03':range(4,6),
    '20190503_Run04':range(4,6),
    '20181206_Run05':range(2,4),
    '20190108_Run05':range(6,9),#range(4,6), # Eval: range(2,6)
    '20190425_Run04':range(4,6),
    '20191126_Run02':range(4,6) }

    n_h_selected = 200
    l1_selected = 0.02
    hidden = 'dReLU'
    n_updates = 200000
    N_MC = 15
    # lr = 1e-3
    # lr = 5e-4
    lr = 2.5e-4
    lr_final = 1e-5
    decay_after = 0.25

    n_h_selecteds = {
    '20180706_Run04':200,
    '20180911_Run01':200,
    '20180912_Run01':200,
    '20180913_Run01':200,
    '20181218_Run02':200,
    '20190109_Run04':100,
    '20190502_Run04':50,
    '20181206_Run03':200,
    '20190102_Run01':100,
    '20190130_Run03':100,
    '20190503_Run04':50,
    '20181206_Run05':100,
    '20190108_Run05':100,
    '20190425_Run04':50,
    '20191126_Run02':50
    }

    lr_selecteds = {
    '20180706_Run04':1e-3,
    '20180911_Run01':1e-3,
    '20180912_Run01':1e-3,
    '20180913_Run01':2.5e-4,
    '20181218_Run02':2.5e-4,
    '20190109_Run04':1e-4,
    '20190502_Run04':2.5e-4,
    '20181206_Run03':2.5e-4,
    '20190102_Run01':1e-4,
    '20190130_Run03':2.5e-4,
    '20190503_Run04':2.5e-4,
    '20181206_Run05':2.5e-4,
    '20190108_Run05':1e-4,
    '20190425_Run04':2.5e-4,
    '20191126_Run02':2.5e-4

    }

    l1_selecteds = {
    '20180706_Run04':0.02,
    '20180911_Run01':0.02,
    '20180912_Run01':0.02,
    '20180913_Run01':0.02,
    '20181218_Run02':0.02,
    '20190109_Run04':0.02,
    '20190502_Run04':0.02/4,
    '20181206_Run03':0.02,
    '20190102_Run01':0.02,
    '20190130_Run03':0.02/2,
    '20190503_Run04':0.02/4,
    '20181206_Run05':0.02,
    '20190108_Run05':0.02,
    '20190425_Run04':0.02/5,
    '20191126_Run02':0.02/4

    }

    batch_size_selecteds = {
    '20180706_Run04':400,
    '20180911_Run01':100,
    '20180912_Run01':100,
    '20180913_Run01':100,
    '20181218_Run02':100,
    '20190109_Run04':400,
    '20190502_Run04':100,
    '20181206_Run03':100,
    '20190102_Run01':400,
    '20190130_Run03':100,
    '20190503_Run04':100,
    '20181206_Run05':100,
    '20190108_Run05':400,
    '20190425_Run04':100,
    '20191126_Run02':100,
    }    

    # n_h_selecteds = dict([ (key,200) for key in runs])

if make_only_RBM_training:
    # runs = [runs[int(sys.argv[1])]]
    # repeated_trainings_RBM = [repeated_trainings_RBM[int(sys.argv[2])]]
    runs = [runs[int(sys.argv[1])]]
    for run in runs:
        repeated_trainings_RBMs[run] = [repeated_trainings_RBMs[run][int(sys.argv[2])]]
    print(runs[0],repeated_trainings_RBMs[runs[0]])

nRuns = len(runs)





spike_dataset_locations = [spike_datasets_folder +
                           run + '__spikes_only.npy' for run in runs]


label_dataset_locations = []
for run in runs:
    if run in ['20180706_Run04',
    '20180911_Run01',
    '20180912_Run01',
    '20180913_Run01']:
        label_dataset_locations.append(labels_datasets_folder +
                               'baier_rl-region-labels__' + run + '__labelled-2019-12-18.npy')
    else:
        label_dataset_locations.append(labels_datasets_folder +
                               'baier_rl-region-labels__' + run + '__labelled-2020-10-15.npy')


# label_dataset_locations = [labels_datasets_folder +
#                            'baier_rl-region-labels__' + run + '__labelled-2019-12-18.npy' for run in runs[:4]]

# label_dataset_locations += [labels_datasets_folder +
#                            'baier_rl-region-labels__' + run + '__labelled-2020-10-15.npy' for run in runs[4:]]

# label_dataset_locations = [labels_datasets_folder +
#                            'baier_rl-region-labels__' + run + '__labelled-2020-10-15.npy' for run in runs]


baier_connectivity_locations = {
    'his_old': connectivity_matrices_folder + 'connectivity_matrix.csv',
    'his_new': connectivity_matrices_folder + 'connectivity_matrix_0512.csv',
    'my_new_unnormalized': connectivity_matrices_folder + 'connectivity_jerome_new_normalized_by_neurons.csv',
    'my_new_normalized_by_volume': connectivity_matrices_folder + 'connectivity_jerome_new_normalized_by_neurons_and_volume.csv'
}

connectivity_types = baier_connectivity_locations.keys()

nRegions_baier = 72

# Load all Baier connectivity matrices.

all_baier_connectivities = {}
for key, location in baier_connectivity_locations.items():
    struct_connectivity_baier, region_names_baier = load_baier_matrix(
        location_matrix=location, location_names=connectivity_matrices_folder + 'list_36_region_names.txt')
    all_baier_connectivities[key] = struct_connectivity_baier.copy()


# Load/Generate all functional connectivity matrices.
connectivity_key_format = 'run:{run}_method:{method}_agg:{aggregation}'
correlation_key_format = 'run:{run}_method:{method}_agg:{aggregation}_conn:{connectivity_type}'
averaged_connectivity_key_format = 'method:{method}_agg:{aggregation}'
correlation_averaged_connectivity_key_format = 'method:{method}_agg:{aggregation}_av:{averaging_method}_conn:{connectivity_type}'



all_functional_connectivities = {}
all_neuron_connectivities = {}

for method in methods:
    for k, run in enumerate(runs):
        print(method,run)
        prediction_location1 = outputs_folder + \
            'run:%s_%s_agg:%s.npy' % (run, method, '1')
        prediction_location2 = outputs_folder + \
            'run:%s_%s_agg:%s.npy' % (run, method, '2')

        key_dict1 = connectivity_key_format.format(
            run=run, method=method, aggregation='1')
        key_dict2 = connectivity_key_format.format(
            run=run, method=method, aggregation='2')

        try:
#            assert 1==0
            functional_connectivity1 = np.load(prediction_location1)
            functional_connectivity2 = np.load(prediction_location2)
            if run == selected_run:
                neuron_connectivity = np.load(connectivities_location_format.format(method=method,run=run))
        except:
            spikes = np.load(spike_dataset_locations[k])
            labels_baier = np.load(
                label_dataset_locations[k]).astype(np.int16) - 1
            neuron_connectivity = infer_functional_couplings(spikes, method,n_h=n_h_selecteds[run],l1=l1_selecteds[run],lr=lr_selecteds[run],batch_size=batch_size_selecteds[run])
            functional_connectivity1 = aggregate_couplings(
                neuron_connectivity, labels_baier, nRegions_baier, power=1)
            functional_connectivity2 = aggregate_couplings(
                neuron_connectivity, labels_baier, nRegions_baier, power=2)
            np.save(prediction_location1, functional_connectivity1)
            np.save(prediction_location2, functional_connectivity2)
            if run == selected_run:
                np.save( connectivities_location_format.format(method=method,run=run), neuron_connectivity.astype(np.float32) )


        all_functional_connectivities[key_dict1] = functional_connectivity1
        all_functional_connectivities[key_dict2] = functional_connectivity2
        if run == selected_run:
            nNeurons = neuron_connectivity.shape[0]
            upper_triangle = (np.arange(nNeurons)[:,np.newaxis] >= np.arange(nNeurons)[np.newaxis,:])
            neuron_connectivity = neuron_connectivity[upper_triangle].astype(np.float16)
            all_neuron_connectivities['method:%s'%method] = neuron_connectivity


# Load all Baier connectivity matrices.
all_baier_connectivities = {}
for key, location in baier_connectivity_locations.items():
    struct_connectivity_baier, region_names_baier = load_baier_matrix(
        location_matrix=location, location_names=connectivity_matrices_folder + 'list_36_region_names.txt')
    all_baier_connectivities[key] = struct_connectivity_baier.copy()



# Load Baier labels information.
all_labels_baier = {}
for k, run in enumerate(runs):
    all_labels_baier[run] = np.load(
        label_dataset_locations[k]).astype(np.int16) - 1

nNeurons_per_region = np.zeros([nRegions_baier, nRuns], dtype=np.int16)
for k, run in enumerate(runs):
    labels_baier = all_labels_baier[run]
    for i in range(nRegions_baier):
        nNeurons_per_region[i, k] = (labels_baier == i).sum()

mean_nNeurons_per_region = nNeurons_per_region.mean(1)

Nmin = 10
subset = np.zeros(nRegions_baier, dtype='bool')
for i in range(nRegions_baier // 2):
    if (mean_nNeurons_per_region[2 * i] >= Nmin) & (mean_nNeurons_per_region[2 * i + 1] >= Nmin):
        subset[2 * i] = 1
        subset[2 * i + 1] = 1
subset = np.nonzero(subset)[0]




averaging_methods = [
    'naive',
    'recording_length',
    'product_nNeurons',
    'sum_nNeurons',
    'recording_length+product_nNeurons',
    'recording_length+sum_nNeurons'
]

# Compute recording weights
len_recordings = [np.load(spike_dataset_locations[k]).shape[0]
                  for k in range(nRuns)]
weights1 = np.ones([nRegions_baier, nRegions_baier, nRuns])
weights2 = np.ones([nRegions_baier, nRegions_baier, nRuns])
weights3 = np.ones([nRegions_baier, nRegions_baier, nRuns])
weights4 = np.ones([nRegions_baier, nRegions_baier, nRuns])
weights5 = np.ones([nRegions_baier, nRegions_baier, nRuns])
weights6 = np.ones([nRegions_baier, nRegions_baier, nRuns])

nNeurons_per_region = nNeurons_per_region.astype(np.float64)

for k, run in enumerate(runs):
    weights1[:, :, k] = 1
    weights2[:, :, k] = len_recordings[k]
    weights3[:, :, k] = nNeurons_per_region[:, np.newaxis, k] * \
        nNeurons_per_region[np.newaxis, :, k]
    weights4[:, :, k] = nNeurons_per_region[:, np.newaxis, k] + \
        nNeurons_per_region[np.newaxis, :, k]
    weights5[:, :, k] = weights2[:, :, k] * weights3[:, :, k]
    weights6[:, :, k] = weights2[:, :, k] * weights4[:, :, k]

all_averaging_weights = {
    'naive': weights1,
    'recording_length': weights2,
    'product_nNeurons': weights3,
    'sum_nNeurons': weights4,
    'recording_length+product_nNeurons': weights5,
    'recording_length+sum_nNeurons': weights6
}


all_averaged_functional_connectivities = {}

# Compute average functional matrices
averaging_weights = all_averaging_weights[selected_averaging_method]
for method in methods:
    for aggregation in aggregations:
        averaged_matrix = np.zeros([nRegions_baier, nRegions_baier])
        for k, run in enumerate(runs):
            prediction = all_functional_connectivities[connectivity_key_format.format(
                run=run, method=method, aggregation=aggregation)]
            averaged_matrix += prediction * averaging_weights[:, :, k]
        averaged_matrix /= averaging_weights.sum(-1)

        all_averaged_functional_connectivities[averaged_connectivity_key_format.format(
            method=method, aggregation=aggregation)] = averaged_matrix.copy()



env = {}
env['runs'] = runs
env['nRegions_baier'] = nRegions_baier
env['region_names_baier'] = region_names_baier
env['connectivity_baier'] = all_baier_connectivities['my_new_normalized_by_volume']
env['nNeurons_per_region'] = nNeurons_per_region
env['mean_nNeurons_per_region'] = mean_nNeurons_per_region
env['subet'] = subset
env['all_functional_connectivities'] = all_functional_connectivities
env['all_averaged_functional_connectivities'] = all_averaged_functional_connectivities
env['all_neuron_connectivities'] = all_neuron_connectivities
env['Nmin'] = Nmin
env['methods'] = methods
env['all_baier_connectivities']= all_baier_connectivities
env['averaging_weights'] = averaging_weights
env['aggregations'] = aggregations
env['connectivity_types'] = list(connectivity_types)
env['nRegions_baier'] = 72


pickle.dump(env, open('/home/iscb/wolfson/jeromet/Data/Neuro/connectivity_plot_files_v4.data','wb'),4 )
