## Python 3

#%% General options:
num_threads = 16  # Set number of CPUs to use!
save_weights = True
generate_data = True

#%% Package loading:
import sys,os,pickle
os.environ["MKL_NUM_THREADS"] = "%s"%num_threads
os.environ["NUMEXPR_NUM_THREADS"] = "%s"%num_threads
os.environ["OMP_NUM_THREADS"] = "%s"%num_threads
os.environ["OPENBLAS_NUM_THREADS"] = "%s"%num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "%s"%num_threads
os.environ["NUMBA_NUM_THREADS"] = "%s"%num_threads
import itertools
import numpy as np
import h5py
import datetime
sys.path.append('PGM7/') # the path where the folder PGM is.
import cProfile
import rbm7_trackprogress as rbm
import pandas as pd
import scipy.io
from fishualizer_utilities import Zecording

#%%  # Path specification
# base_path = '/home/thijs/Desktop/RBM/For_Thijs/'
base_path = '/home/thijs/Desktop/'  #/Desktop/'
data_set = '2019-03-26(Run09).h5'
# data_set = '20180912_Run01_spontaneous_rbm2.h5'
# data_set = '20180912Run01_rhomb_all_spont_train4k_nit45k_nh100_l1_8e-03_simulated_NC1_LC4000_beta1_Nstep20_NPT1_improvtrain.h5'

#%% Names for saving:x
rbm_type = 'RBM7ADAM'
data_name = data_set.rstrip('.h5')
rbm_save_folder = '/home/thijs/Desktop/RBM/For_Thijs/may_profiler/'
weights_save_folder = rbm_save_folder + 'weights/'
gen_save_folder = rbm_save_folder + 'generated/'

#%%  # Data set selection parameters:
# NOT LOOPED
subselection = True
selection_method = 'region'  # 'activity' or 'region'
selected_region = np.arange(294)
print('selection_method: ', selection_method)
time_cutoff_start, time_cutoff_end = 0, 8000  # cut off point for training (ie. only time points upto time_cutoff are used for training, others can be used for evaluation post hoc )
train_condition = f'train{time_cutoff_start}-{time_cutoff_end}'

#%% Model design parameters
# LOOPED
list_m = [40, 60, 80]
list_l1 = [0.005, 0.01, 0.05, 0.1, 0.3] #, 0.02] #, 0.02, 0.05]  # warning: this is round to scientific notation with 1 decimal in rbm_title (to avoid dot notation)
list_hiddenprior = ['dReLU']

#%% Model training parameters:
# LOOPED
list_number_mcc= [15]
list_n_updates = [200000]
list_batch_size = [100]
list_lr = [5e-3]
list_lr_final_param = [1e-5]
list_decay_after_param = [0.25]

#%% Data generation parameters
if generate_data:
    gen_lenghtchains = 50  # lenght mc chains
    gen_nchains = 300  # number of chains
    gen_nsteps = 20  # number of steps between saved mc points
    gen_nthermalize = 4000  # number of burn-in points
    gen_npt = 1
    gen_beta = 1



#%% Data loading
rec = Zecording(path=base_path + data_set, kwargs={'ignorelags': True,
                                              'forceinterpolation': False,
                                              'ignoreunknowndata': True,# 'parent': self,
                                              'loadram': True})  # load data
if 'spikes' in rec.available_data:
    try:
        spikes_bin = rec.spikes.value
    except AttributeError: # already of type numpy array
        spikes_bin = rec.spikes
    assert (np.sum(spikes_bin == 0) + np.sum(spikes_bin == 1)) == spikes_bin.size  # assert binary
    print('debug - spikes loaded')
elif 'df' in rec.available_data:  # possibly, spikes are disguised as df data
    if rec.df.dtype == 'int8':  # check by dtype check
        spikes_bin = rec.df.value
        print('debug - df loaded as spikes ')
    else:  # else, perhaps the df matrix is binary but in float 32
        print(f'df data type: {rec.df.dtype}')
        len_zeros = len(np.where(rec.df == 0)[0])
        len_ones = len(np.where(rec.df == 1)[0])
        print(f'debug - zeros {len_zeros}, ones {len_ones}, together {len_zeros + len_ones}, size {rec.df.size}')
        if len_zeros + len_ones == rec.df.size:
            try:
                spikes_bin = rec.df.value
            except AttributeError:
                spikes_bin = rec.df
            print('debug - df loaded ')
        else:
            print('debug - not loaded')
else:
    print('No binary spikes found in data?? aborting')
    sys.exit('quitting')
spikes_bin = spikes_bin.astype('int8')
if subselection:
    if selection_method == 'region':
        tmp_sn = np.nonzero(rec.labels[:, selected_region])  # 218 corresponds to rhombomere 1, 113 to full rhombomere
        # if type(selected_region) is int:  # single region
        selected_neurons = np.unique(tmp_sn[0])
        # elif len(selected_region) > 1:  # multiple regions
            # selected_neurons = np.unique(tmp_sn[1])
    else:
        print('Other selection filter currently not implemented, aborting')
        sys.exit('quitting')
    print(f'debug - shape spikes {spikes_bin.shape}')
    data_original = spikes_bin[selected_neurons, :]
    print('len selected neurons: ', len(selected_neurons))
elif subselection is False:
    data_original = spikes_bin
print('Shape data used {}'.format(data_original.shape))
n_cells = data_original.shape[0]
time_slice = slice(time_cutoff_start, time_cutoff_end)
data_use = data_original[:, time_slice]
if 'weight_bursts' in rec.available_data:  # if weighted data set, weigh:
    weights_bursts = rec.weight_bursts
    print('weights loaded, with shape {}'.format(weights_bursts.shape))
else:
    weights_bursts = None  # default if no weights
data_use = np.transpose(data_use)

#
# hfile = h5py.File(base_path + data_set, 'r')
# # data_use_h5 = hfile['Data']['spikes']
# data_use_h5 = hfile['Data']['dff']
# print('Data used: {}'.format(data_use_h5))
# spikes_bin = data_use_h5.value  # shape should be ok like this? (T, N)
# labels = hfile['Data']['labels'].value
# print('labels shape', labels.shape)
# assert labels.shape[0] == 294 # check right orientation
# if subselection:  # select subset of neurons
#     if selection_method == 'region':
#         tmp_sn = np.nonzero(labels[selected_region, :])  # 218 corresponds to rhombomere 1, 113 to full rhombomere
#         if type(selected_region) is int:  # single region
#             selected_neurons = tmp_sn[0]
#         elif len(selected_region) > 1:  # multiple regions
#             selected_neurons = np.unique(tmp_sn[1])
#     elif selection_method == 'activity':
#         filter_neurons = np.mean(spikes_bin, axis=0)
#         print('shape filter ', filter_neurons.shape)
#         argsorted = np.argsort(filter_neurons)  # small to large
#         n_selected = len(np.nonzero(labels[selected_region, :,])[0])  # corresponding to Rh 1
#         selected_neurons = argsorted[-n_selected:]
#         print('0, -1 of filter_neurons ', filter_neurons[0], filter_neurons[-1])
#         print('0, -1 of filter_neurons[n_selected] ', filter_neurons[selected_neurons[0]], filter_neurons[selected_neurons[-1]])
#     data_original = spikes_bin[:, selected_neurons]
#     print('len selected neurons: ', len(selected_neurons))
# elif subselection is False:
#     data_original = spikes_bin
# print('Shape data used {}'.format(data_original.shape))
# n_cells = data_original.shape[1]
# if 'weight_bursts' in list(hfile['Data'].keys()):  # if weighted data set, weigh:
#     weights_bursts = hfile['Data']['weights_bursts'].value
#     print('weights loaded, with shape {}'.format(weights_bursts.shape))
# else:
#     weights_bursts = None  # default if no weights
# time_slice = slice(time_cutoff_start, time_cutoff_end)
# data_use = data_original[time_slice, :]

if type(selected_region) is not int and len(selected_region) == 294:
    str_region = 'wb'
elif selected_region == 38:
    str_region = 'test38'
elif selected_region == 113:
    str_region = 'rhall'
elif selected_region == 218:
    str_region = 'rh1'
else:
    print("REGION NOT RECOGNIZED")
    str_region = 'region-' + str(selected_region)

#%% Determine number of loops:
total_loops = 0
for i_m, i_l1, i_hiddenprior, i_number_mcc, i_n_updates, i_batch_size, i_lr, i_lr_final_param, i_decay_after_param in \
    itertools.product(list_m, list_l1, list_hiddenprior, list_number_mcc, list_n_updates, list_batch_size, list_lr, \
    list_lr_final_param, list_decay_after_param):
    total_loops += 1




#%% RBM estimation loop
i_loop = 0
for i_m, i_l1, i_hiddenprior, i_number_mcc, i_n_updates, i_batch_size, i_lr, i_lr_final_param, i_decay_after_param in \
    itertools.product(list_m, list_l1, list_hiddenprior, list_number_mcc, list_n_updates, list_batch_size, list_lr, \
    list_lr_final_param, list_decay_after_param):

    i_loop +=1
    print('------------------------------------')
    print('Loop {}/{}'.format(i_loop, total_loops))
    t_start = datetime.datetime.now()
    print(t_start)
    print('------------------------------------')

    i_n_iterations = i_n_updates / (data_use.shape[0] / i_batch_size)
    print(f'Number of iterations {i_n_iterations}')
    RBM = rbm.RBM(n_v=n_cells, n_h=i_m, visible='Bernoulli', hidden=i_hiddenprior)
    RBM.fit(data_use, learning_rate=i_lr, lr_final=i_lr_final_param, decay_after=i_decay_after_param,
    optimizer='ADAM', extra_params=[0, 0.999, 1e-6],
            verbose=0, n_iter=i_n_iterations, l1=i_l1, N_MC=i_number_mcc, weights=weights_bursts)

    t_end = datetime.datetime.now()
    timestamp = str(t_end.date()) + '-' + str(t_end.hour).zfill(2) + str(t_end.minute).zfill(2)
    duration = int(np.round((t_end - t_start).total_seconds()))
    str_l1 = '{:.0e}'.format(i_l1)  # get rid of dot notation
    rbm_name = (f'{rbm_type}_{data_name}_{str_region}_{train_condition}_M{i_m}_l1-{str_l1}_nit{i_n_updates}' +
                f'_nmc{i_number_mcc}_duration{duration}s_timestamp{timestamp}.data')
    tmp_count_name_check = 0
    all_current_time_stamps = [x.split('_')[-1].rstrip('.data').lstrip('timestamp') for x in os.listdir(rbm_save_folder) if x[-5:] == '.data']
    while timestamp in all_current_time_stamps:  # change time stamp until unique stamp is found
        tmp_count_name_check += 1
        print('changing time stamp')
        timestamp = timestamp[:-4] + str(int(timestamp[-4:]) + 1)  # add one
        rbm_name = (f'{rbm_type}_{data_name}_{str_region}_{train_condition}_M{i_m}_l1-{str_l1}_nit{i_n_updates}' +
                    f'_nmc{i_number_mcc}_duration{duration}s_decay{int(i_decay_after_param * 100)}_timestamp{timestamp}.data')
        if tmp_count_name_check > 20:
            print(f'Something is odd about the rbm name check, now at {tmp_count_name_check} alterations.')
            print('ABORTING')
            break
    print(rbm_name)

    if save_weights:
        weights_name = 'weights_' + rbm_name[:-5] + '.h5'
        weights = np.transpose(RBM.weights)
        assert weights.shape[0] > weights.shape[1]  # n_cells > n_hidden
        df_weights = pd.DataFrame({'hu_' + str(ii).zfill(3):
                                    np.squeeze(weights[:, ii]) for ii in range(weights.shape[1])})
    try:
        pickle.dump(RBM,open(rbm_save_folder + rbm_name, 'wb'))
        print(f'RBM # {i_loop}/{total_loops} saved succesfully, total duration: {duration}s.')
        if save_weights:
            df_weights.to_hdf(weights_save_folder + weights_name, key='all')
            print('Weights saved succesfully.')
    except:
        print("saving failed")
        print('Time: ', t_end)
        print(f'Loop # {i_loop}/{total_loops}')

    if generate_data:
        print(f'Initiating data generation. Time: {datetime.datetime.now()}')
        sim_data, hidden_data = RBM.gen_data(Nchains=gen_nchains, batches=None,
                                             Lchains=gen_lenghtchains, Nthermalize=gen_nthermalize,
                                             reshape=True, Nstep=gen_nsteps,
                                             N_PT=gen_npt, beta=gen_beta)
        sim_data = np.squeeze(sim_data)
        sim_data = sim_data.astype('int8')
        hidden_data = np.squeeze(hidden_data)
        hidden_data = hidden_data.astype('float32')
        print('shape data, {}; hidden data {}'.format(sim_data.shape, hidden_data.shape))
        # coordinates = hfile['Data']['coords'].value
        coordinates = rec.coords
        # z_coordinates = hfile['Data']['zbrain_coords'].value
        z_coordinates = rec.zbrain_coords
        labels = rec.labels.A
        gen_name = (f'generated_{rbm_type}_{data_name}_{str_region}__M{i_m}_l1{i_l1}_timestamp{timestamp}__nchains{gen_nchains}' +
                    f'_lchains{gen_lenghtchains}_ntherm{gen_nthermalize}_nsteps{gen_nsteps}.h5')
        gen_hfile = h5py.File(gen_save_folder + gen_name, 'w')
        gg = gen_hfile.create_group('Data')
        gg.create_dataset('coords', data=coordinates[selected_neurons, :])
        gg.create_dataset('zbrain_coords', data=z_coordinates[selected_neurons, :])
        gg.create_dataset('labels', data=labels[selected_neurons, :])
        gg.create_dataset('dff', data=sim_data)
        gg.create_dataset('hidden_data', data=hidden_data)
        gen_hfile.close()
        print('Data generated and saved.')

# hfile.close()
