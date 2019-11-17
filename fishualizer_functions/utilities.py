import h5py
import numpy as np
import logging
import os
import json
import getpass
import psutil as ps
import scipy.sparse
import scipy.interpolate as ip
from PyQt5 import QtCore, QtWidgets, QtGui, Qt
from Controls import AssignLoadedData, DsetConflictDialog
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger('Fishlog')


def get_data_orientation(coords, data):
    """
    Try to guesstimate the orientation of some data based on the coordinates

    Parameters
    ----------
    coords: 2D-array
        Spatial coordinates of neurons
    data: 2D-array
        Any kind of 2D array with one dimension being cells and the other one being time

    Returns
    -------
    n_cells: int
    n_times: int
    orientation: str
        Can be 'cellsXtimes', 'timesXcells'
    """
    n_cells = max(coords.shape)
    if data.shape[0] == n_cells:
        return n_cells, data.shape[1], 'cellsXtimes'
    elif data.shape[1] == n_cells:
        return n_cells, data.shape[0], 'timesXcells'
    else:
        raise ValueError('Data shape does not match the coordinates')


def open_h5_data(filepath, ignorelags=False, forceinterpolation=False,
                 ignoreunknowndata=False, parent=None, loadram=False):
    """
    Soft-load data names from hdf5 file. Guess by name, toggle between coords and
    ref_coords, load labels from hdf5 file, if not load from filepath (in load_data())
    in Fishualizer.py.

    Parameters
    ----------
    filepath: str or Path
    ignorelags: bool (default False)
        whether to ignore time lags between z layers
    forceinterpolation: bool (default False)
        whether to force time interpolation (between layers) even if df_aligned is provided in the h5 file
    ignoreunknowndata: bool (default False)
        whether to ignore unknown data, meaning data that is not recognised by the
        default names dictionary. If true, the user is prompted to assign it to a
        default data set (or cancel), if false this is not asked.
    parent: None or QDialog?
    loadram: boolean

    Returns
    -------
    data: dict
          Dictionary containing the loaded data only, possibilities:
              df: Calcium signal, default shape is Neurons X Times (note: inverted in Matlab H5 saving)
              coords: Position of cells in own space
              ref_coords: Position of cells in reference space
              behavior: behavioral data
              stimulus: stimulus data
              times: Time points
              spikes: spike data
              labels: labels of anatomical alignment with ZBrainAtlas
              not_neuropil: boolean values, True=not neuropil, False=neuropil
    h5_file: File handle
        Handle on the hdf5 file with the data
    assigned_names: dict
        Dictionary containing the mappings between default names (as keys) and
        hdf5 file names (as values)
    """
    # dataformat: 'cellsXtimes' or 'timesXcells' or None (default None)
    #     data orientation
    #    if None, tries to guess with `get_data_orientation`
    # open h5 file in read+ modus
    logger.debug(f'Opening file {filepath} with following options: '
                 f'ignorelags: {ignorelags}, forceinterpolation: {forceinterpolation}, '
                 f'ignoreunknowndata: {ignoreunknowndata}, loadram: {loadram}')

    all_keys = dict()

    def get_datasets(data_path, obj):
        if isinstance(obj, h5py.Dataset):  # obj is a dataset
            tmp_name = data_path.split('/')[-1].lower()
            all_keys[tmp_name] = obj.name

    h5_file = h5py.File(filepath, 'r+')
    h5_file.visititems(get_datasets)
    with open('Content/data_names_links.json', 'r') as tmp:
        names_dict = json.load(tmp)
    # import default names as sets to  {default name: set(possible names)}
    # default_names = {x: set(y) for x, y in names_dict.items()}
    reverse_default_names_map = {vv: k for k, v in names_dict.items() for vv in  #TODO: force all lowercase (in case users input a non-lowercase data name in names_dict file)
                                 v}  # {possible name: default name} mapping
    logger.debug(f'All data paths in h5 file: {all_keys}')
    non_assigned_names = {}
    assigned_names = {}
    data = {}
    static_data = {}
    name_mapping = {}
    conflicting = {}
    # Loop through all data sets in all groups in h5 file and map the names of the datasets to predefined names
    for name, dataset_path in all_keys.items():
        # check data set against standard names (in .json dictionary of links?)
        try:
            tmp_default_name = reverse_default_names_map[name]  # only works if 'name' is among predefined names that map to this dict item, abort (via try) if fails, otherwise continue
            assigned_names[tmp_default_name] = name
            if tmp_default_name in name_mapping.keys():
                # TODO: test this functionality
                names = [name_mapping[tmp_default_name], dataset_path]
                names.extend(conflicting.get(tmp_default_name, []))
                conflicting[tmp_default_name] = names
                logger.debug(conflicting)
            else:  # not defined yet, add to data
                name_mapping[tmp_default_name] = dataset_path

        except KeyError:  # data name not defined
            # TODO: write to json file?
            # Not sure what to do here because if we think of the new format the LJP wants to adopt
            # it would be a lot of non-assigned datasets
            non_assigned_names[name] = dataset_path

    for dset_name, conflicts in conflicting.items():
        if parent is not None:
            chosen_win = DsetConflictDialog(parent, 'Conflicting data sets',
                                            f'Which dataset from the file should be used as {dset_name}?', conflicts)
            chosen = chosen_win.get_datapath()
            name_mapping[dset_name] = chosen

    # Guessing data orientation of df
    n_cells, n_times, dataformat = get_data_orientation(h5_file[name_mapping['coords']], h5_file[name_mapping['df']])
    logger.debug(f'df specifics: n_cells: {n_cells}, n_times: {n_times}, dataformat: {dataformat}')

    # Loop through all the mapped datasets, load, transpose ...
    for name, dataset_path in name_mapping.items():
        if name not in ['df', 'df_aligned', 'spikes']:
            # This is not activity data, so we load in memory
            data[name] = np.float32(h5_file[dataset_path].value)  # load data from h5file  in RAM
            # Sometimes data are in the wrong orientation (X, n_cells). We correct for this here
            if data[name].shape[1] == n_cells and data[name].shape[0] != n_cells:
                data[name] = data[name].T
            if (name == 'behavior' or name == 'stimulus') and data[name].shape[-1] != n_times:
                data[name] = data[name].T
            logger.info(f'{name} is loaded in RAM with shape {data[name].shape}')
        else:
            # If df_aligned is available and we don't want to do the interpolation again, we don't load df
            if name == 'df' and assigned_names.get('df_aligned', None) in all_keys.keys() and not forceinterpolation and not ignorelags:
                logger.debug('df data not loaded because df_aligned in data set.')
                continue
            elif name == 'df_aligned' and assigned_names.get('df', None) in all_keys.keys() and (forceinterpolation or ignorelags):
                logger.debug(f'df_aligned data not loaded because df in data set and ignorelags={ignorelags}.')
                continue

            if name == 'df_aligned':
                new_name = 'df'  # save as df (because df is not loaded anyway)
            else:
                new_name = name

            # Guessing data orientation of current data (df, df_aligned and spikes are not necessarily all aligned)
            n_cells, n_times, dataformat = get_data_orientation(h5_file[name_mapping['coords']], h5_file[name_mapping[name]])
            logger.debug(f'{name} specifics: n_cells: {n_cells}, n_times: {n_times}, dataformat: {dataformat}')
            if not loadram and dataformat == 'timesXcells' and parent is not None:
                # Should we load in RAM because wrong orientation for memory mapping
                bytes_per_el = np.dtype(h5_file[name_mapping[name]].dtype).itemsize  # number of bytes per element (dependent on data type)
                n_els = np.product(h5_file[name_mapping[name]].shape)  # number of elements ( in 2D array)
                d_size = np.round((n_els * bytes_per_el) / (1024 ** 2))  # data size in MB
                available_ram_memory = np.round(ps.virtual_memory().available / (1024 **2))  # in MB
                logger.debug(f'Data {name} with size {d_size}MB, RAM availability {available_ram_memory}MB.')
                if d_size > (0.2 * available_ram_memory):  # if greater than 20% of available ram, ask for permission to load in ram
                    logger.debug('Prompting whether data can be loaded in RAM because it must be transposed')
                    loadram = QtWidgets.QMessageBox.question(parent, 'Loading data in memory',
                                                             f'The {name} data must be loaded in memory, '
                                                             f'because its orientation is not handled otherwise. '
                                                             f'This will require {d_size:.0f}MB of RAM ({available_ram_memory:.0f}MB currently available). \n'
                                                             'Would you like to continue (press Yes) or abort loading (press No)?')
                    loadram = True if loadram == QtWidgets.QMessageBox.Yes else False  # enable loading in RAM
                else:
                    logger.debug(f'Data {name} takes less than 20% of the available RAM, so it is automatically loaded in RAM.')
                    loadram = True

            if loadram:
                data[new_name] = h5_file[dataset_path].value  # load in RAM
                if dataformat == 'timesXcells':
                    data[new_name] = data[new_name].T  # transpose if necessary
                logger.info(f'{name} is loaded in RAM with shape {data[new_name].shape}')
            elif not loadram and dataformat == 'cellsXtimes':  # if to be memory-mapping and in right format
                data[new_name] = h5_file[dataset_path]  # memory map
                logger.info(f'{name} is loaded with memory-mapping with shape {data[new_name].shape}')
            elif not loadram and dataformat == 'timesXcells' and name != 'spikes':  # if to be memory mapping, but in incorrect format
                # Memory mapping not possible; Abort loading
                h5_file.close()
                logger.info(f'{name} not loaded, loading procedure has been aborted. \n Please load this data set in RAM, or choose a different data set.')
                print(f'{name} not loaded, loading procedure has been aborted. \n Please load this data set in RAM, or choose a different data set.')
                return None, None, None

    bool_static_added = False
    for name, dataset_path in non_assigned_names.items():  # loop through non assignmend/recognised name to see if Nx1 data sets exist
        tmp_size = h5_file[dataset_path].shape
        if len(tmp_size) == 1 and np.squeeze(tmp_size) == n_cells:
            static_data[name] = np.float32(np.squeeze(h5_file[dataset_path].value))
            logger.info(f'{name} added as static data set.')
            bool_static_added = True
        else:
            logger.warning(f'{name} with shape {tmp_size} is not recognized, so it cannot be loaded.')
    if bool_static_added:
        data['_additional_static_data'] = static_data  # will be unpacked in zecording

    if (not ignorelags and ("df_aligned" not in all_keys.keys())) or forceinterpolation:  # if open to interpolation
        if ('layerlags' in name_mapping.keys()):# or forceinterpolation:  # thijs; I have removed force here, because no automated layer creation is implemented (I would do this by hand when testing, the relevant line is in comments in layer_lags_correction())
            if ('times' in name_mapping.keys()) and ('df' in name_mapping.keys()):  #required data for layer lag correction
                interp_warn = True
                if parent is not None:
                    tmp_name_layers = name_mapping['layerlags']
                    interp_warn = QtWidgets.QMessageBox.question(parent, 'Time correction for different layers',
                                                                 f'Time delays per layer were found in the h5 data set ({tmp_name_layers}). \n'
                                                                 'Do you want to correct the timing? This is done by interpolating all layers by '
                                                                 'their respective offset. This will take several minutes, but the result'
                                                                 ' will be saved as an additional data set in the h5 file, called df_aligned, so that'
                                                                 ' it can readily be used in the future.')
                    interp_warn = True if interp_warn == QtWidgets.QMessageBox.Yes else False
                if interp_warn:
                    group_df = h5_file[name_mapping['df'].rsplit('/', 1)[0]]  # the group where df is located, df_aligned will be saved here
                    layer_lags_correction(data, group_df, name_mapping, parent=parent)
                    data['df'] = group_df['df_aligned']
                    logger.info(f'Time layer correction is a success! df_aligned saved in {group_df} and loaded.')
                else:
                    logger.info('Time correction of data was canceled by user. `ignorelags` set to True')
                    ignorelags = True
        else:
            logger.warning(
                f"No lags across layers provided in HDF5 file ({filepath}), using uncorrected data, 'ignorelags' set to True.")
            ignorelags = True

    if 'df' not in data:  # necessary to plot
        logger.error(f'No df/f values loaded from data file {filepath}')

    # TODO: align behavior? see below
    else:  # use shape of df to transpose other data sets if needed
        # if 'behavior' in data:
        #     if len(data['behavior'].shape) > 1:  # in case 'behavior' is squeezed 1D, not needed.
        #         if data['behavior'].shape[1] is not data['df'].shape[1]:
        #             data['behavior'] = data['behavior'].transpose() doesn't work with one dataset  # numpy transpose

        if 'stimulus' in data:
            if len(data['stimulus'].shape) > 1:
                if data['stimulus'].shape[1] is not data['df'].shape[1]:
                    data['stimulus'] = data['stimulus'].transpose()

    if any([(x in data.keys()) for x in
            ['coords', 'ref_coords', 'zbrain_coords']]) is False:  # no coordinates available
        logger.error(f'No coordinates loaded from data file {filepath}')

    if 'times' not in data.keys():  # necessary to plot, create np.arange(T) otherwise
        data['times'] = np.arange(np.shape(data['df'])[1])
        data['times'] = data['times'][np.newaxis, :]  # it must be 2D for somewhere in Fishualizer code
        logger.info('time vector created')

    if 'labels' in data:
        """
        HDF5 can not directly store sparse matrices, so it is imported as full
        matrix. It is changed to scipy.sparse so it is compatible with Fishualizer code and for
        memory usage.
        """
        data['labels'] = scipy.sparse.csr_matrix(data['labels'])

    # TODO: Let's get rid of this neuropil data sets/condition?
    if 'neuropil' in data and 'not_neuropil' not in data:  # not_neuropil is needed for Fishualizer.py
        data['not_neuropil'] = np.logical_not(data['neuropil'])
        del data['neuropil']  # because only not_neuropil is needed (and they contain the same information)

    if 'not_neuropil' in data:
        data['not_neuropil'] = data['not_neuropil'].astype('bool')
        data['not_neuropil'] = np.squeeze(
            data['not_neuropil'])  # because it has to be 2D to be loaded by transpose (1,0)

    # assigned_names is returned for coordinate choice when multiple coordinate sets are present (in Fishualizer.py)
    return data, h5_file, assigned_names  # pass on to Zecording class


def layer_lags_correction(dict_data, group, name_mapping, parent=None):
    """
    Time interpolation to correct time offset per layer. Interpolation parameters are
    set inside this function.

    Interpolation is performed over chunks of the data.


    Parameters
    ----------
    dict_data:
        dictionary containing data (as defined in utilities.open_h5_data())
    group:
        handle of group in hfile where df_aligned should be written to (typically the group of df)
    name_mapping: dict
        dictionary with default_name:data_set_path, corresponds to all_keys dict in open_h5_data()

    Returns
    -------
    data_corrected

    """
    if parent is not None:
        tmp_msg = QtWidgets.QMessageBox.information(parent, 'Computing time layer offset correction', 'This can take several minutes, please wait. \n'
                                                                                                      'You can follow the progress in the status bar at the bottom of the Fishualizer interface.')

    logger.debug('Interpolating df data in time to correct the time lags between the layers')
    df = dict_data['df']
    times = dict_data['times']
    lags = dict_data['layerlags']
    # lags = coords[:,2]/(np.max(coords[:,2])+1)*(times[1]-times[0])  # toy lags for testing
    n_cells = df.shape[0]

    try:
        shape_int = df.shape # if dataformat == 'cellsXtimes' else df.shape[::-1]
        logger.debug(f'df_aligned data set created in h5 file with size {shape_int}')
        df_corrected = group.create_dataset('df_aligned', shape_int, dtype='float32')  # to avoid writing float64s
    except RuntimeError:
        df_corrected = group['df_aligned']

    n_chunks = 1  # number of h5 chunks to create

    interpolation_kind = 'cubic'  # interpolation type
    n_overlap = 3  # number of overlap time points (needed for smooth beginning of chunks)

    chunk_ix = np.linspace(0, len(times), n_chunks + 1, dtype=np.intp)  # create h5 chunks
    dtime = 0.25 * (times[1] - times[0])  # resolution of interpolation
    nsteps = 7  # factor of time resolution enhancement
    localtimes = np.linspace(-dtime, dtime, nsteps)  # extra time points locally (around some time point )
    allintertimes = np.zeros((len(times), len(localtimes)))  # create matrix of all new times
    for it in np.arange(len(times)):
        allintertimes[it, :] = times[it] + localtimes

    print(f'Starting interpolation with {n_chunks} chunks and {n_cells} neurons. \n Printing progress during interpolation.')
    for ichunk in np.arange(n_chunks):
        # Define indices to be worked on in this chunk
        cchunk = np.arange(chunk_ix[ichunk], chunk_ix[ichunk + 1])
        cchunk_leadin = np.arange(max(chunk_ix[ichunk] - n_overlap, 0), chunk_ix[ichunk + 1])  # time indices of chunk including lead in

        cdf = df[:, cchunk_leadin]  # select chunk df data
        cdf_interp = np.zeros((n_cells, len(cchunk)))  # interpolated df
        cintertimes = np.reshape(allintertimes[cchunk, :], -1)  # high resolution interpolation times to 1d array

        # Separate the cases for the different chunks to avoid
        nan_ix_start = []
        nan_ix_end = []
        if ichunk == n_chunks - 1:  # special treatment of the last time-point
            nan_ix_end = np.empty(int(np.floor(nsteps / 2)))
            nan_ix_end[:] = np.nan
            cintertimes = cintertimes[0:int(-np.floor(nsteps / 2))]
        if ichunk == 0:  # special treatment of the first time-point
            nan_ix_start = np.empty(int(np.floor(nsteps / 2)))
            nan_ix_start[:] = np.nan
            cintertimes = cintertimes[int(np.floor(nsteps / 2)):]

        mod_count = np.round(n_cells / 100)
        for ineuron in range(n_cells):
            if np.mod(ineuron, mod_count) == 0:
                print(f'  Progress: {np.round((ineuron+1)/n_cells * 100 * (ichunk+1)/n_chunks, 1)}%')  # print progress in console
                if parent is not None:
                    parent.statusBar().showMessage(f'  Progress: {np.round((ineuron+1)/n_cells * 100 * (ichunk+1)/n_chunks, 1)}%')  # print progress in status bar
            x = np.squeeze(times[cchunk_leadin] + lags[ineuron])  # add lag to original data (to get the corrected timing)
            y = cdf[ineuron, :]  # get df of this chunk of ineuron

            # PERFORM INTERPOLATION
            cinterpolator = ip.interp1d(x, y, fill_value='extrapolate', assume_sorted=True, kind=interpolation_kind)  # create interpolation function
            interp_tmp = cinterpolator(cintertimes)  # interpolate high res time points

            # Account for ends of the range
            if len(nan_ix_start):
                interp_tmp = np.concatenate((nan_ix_start, interp_tmp))
            if len(nan_ix_end):
                interp_tmp = np.concatenate((interp_tmp, nan_ix_end))

            cdf_interp[ineuron, :] = np.nanmean(np.transpose(np.reshape(interp_tmp, (len(cchunk), nsteps))), 0)  # average over local high res time points to get original time res
        df_corrected[:, cchunk] = cdf_interp  # save this chunk
        if parent is not None:
            parent.statusBar().showMessage('Interpolation done.')


def load_config():
    """
    Load the JSON configuration file and return parameters corresponding to current user

    Returns
    -------
    user_params: dict
    """
    username = getpass.getuser()
    with open('config.json', 'r') as config_file:
        all_params = json.load(config_file)
    try:
        user_params = all_params[username]
    except KeyError:
        user_params = all_params['default']
    user_params['paths'] = {k: os.path.expanduser(p) for k, p in user_params['paths'].items()}
    # user_params['load_ram'] = np.bool(user_params['load_ram'])
    return user_params


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class Zecording(object):
    def __init__(self, path: str, kwargs) -> None:
        if path == '':
            self.path = None
            return
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f'{path} does not exist')
        create_class = True
        self._data, self.file, self._data_name_mappings = open_h5_data(self.path.as_posix(), **kwargs)  # load data
        if self._data == None:  # None is returned if data loading was aborted
            create_class = False
            #TODO: handle ensuing errors (e.g. return Zecording=None, and handle in Fishualizer?)

        self._sel_coords = self.coords
        self._avail_data = None
        self._single_data = None
        self._multi_data = None
        self._sr = None  # sampling rate
        self._structural_data = None
        self._spatial_scale_factor = np.array([1, 1, 1])  # default to (effectively) no scaling
        self.datasets = KeyDefaultDict(self.__getattribute__,
                                       {'df': self.df, 'spikes': self.spikes, 'input': self.input,
                                        'output': self.output})
        if 'parent' not in kwargs.keys() and '_additional_static_data' in self._data.keys():  # if no parent (ie loaded from script != fishualizer), and static data => add static data right now (because otherwise it is done in Fishualizer.load_data())
            for s_data_name, s_data_set in self._data['_additional_static_data'].items():
                self.add_supp_single_data(s_data=s_data_set, s_name=s_data_name)

        # TODO: Keep track of the static datasets?
        logger.info('Zecording object created')

    @property
    def n_cells(self):
        return self.df.shape[0]

    @property
    def df(self):
        return self._data['df']

    @property
    def coords(self):
        return self._data['coords']

    @property
    def time(self):
        t = np.squeeze(self._data['times'])
        return t

    # The class can also be used to provide aliases
    calcium = df
    times = time
    t = time

    @property
    def n_times(self):
        return len(self.times)

    @property
    def structural_data(self):
        try:
            return self._data['structural_data']
        except KeyError:
            logger.debug(f'Data does not contain structural_data')
            return None

    @property
    def sampling_rate(self):
        # we can additionally load from hdf5 (as attribute?)
        if self._sr is None:  # if not defined, guess
            # the advantage of doing this here instead of in __init__() is that now you can always reset SR to None
            # (in the console)
            self._sr = 1 / (np.mean(np.diff(self.times)))  # guess sr
        return self._sr

    @sampling_rate.setter
    def sampling_rate(self, value):
        self._sr = value

    @property
    def data_names(self):
        return self._data_name_mappings

    @property
    def ref_coords(self):
        try:
            return self._data['ref_coords']
        except KeyError:
            logger.debug(f'Data does not contain reference coordinates')
            return None

    @property
    def zbrain_coords(self):
        try:
            return self._data['zbrain_coords']
        except KeyError:
            logger.debug(f'Data does not contain ZBrain coordinates')
            return None

    @property
    def behavior(self):
        try:
            if len(self._data['behavior'].shape) == 2:
                return np.squeeze(self._data['behavior'][0, :])
            elif len(self._data['behavior'].shape) == 1:
                return np.squeeze(self._data['behavior'])
            else:
                return self._data['behavior']
        except KeyError:
            logger.debug(f'Data does not contain behavioral data')
            return None

    @property
    def spikes(self):
        try:
            return self._data['spikes']
        except KeyError:
            logger.debug(f'Data does not contain spiking data')
            return None

    @property
    def stimulus(self):
        try:
            return np.squeeze(self._data['stimulus'])
        except KeyError:
            logger.debug(f'Data does not contain stimulus data')
            return None

    @property
    def not_neuropil(self):
        try:
            return self._data['not_neuropil']
        except KeyError:
            logger.debug(f'Data does not contain neuropil information')
            return None

    @property
    def labels(self):
        try:
            return self._data['labels']
        except KeyError:
            logger.debug(f'Data does not contain ZBrainAtlas labels')
            return None

    @labels.setter
    def labels(self, value):
        # This is implemented just to make the 'old way' of loading labels work
        # It might be removed once the large SampleData file is converted
        self._data['labels'] = value

    @property
    def phase_map(self):
        try:
            return self._data['phase_map']
        except KeyError:
            return None

    @property
    def layerlags(self):
        try:
            return self._data['layerlags']
        except KeyError:
            return None

    @property
    def input(self):
        try:
            return self._data['input']
        except KeyError:
            return None

    @property
    def output(self):
        try:
            return self._data['output']
        except KeyError:
            return None

    @output.setter
    def output(self, value):
        # Seems to be required in one place in the Fishualizer (when input and output are the same)
        self._data['output'] = value

    @property
    def spatial_scale_factor(self):
        return self._spatial_scale_factor

    @spatial_scale_factor.setter
    def spatial_scale_factor(self, scaling_factors):
        try:
            x_scalef, y_scalef, z_scalef = scaling_factors
        except ValueError:
            logger.warning('Pass an iterable with three items (xscale, yscale, zscale)')
            x_scalef, y_scalef, z_scalef = (1, 1, 1)  # use default scaling
        self._spatial_scale_factor = np.array([x_scalef, y_scalef, z_scalef])

    @property
    def sel_frame(self):
        raw_coords = self._sel_coords  # current selection
        scaling = self.spatial_scale_factor  # current scaling
        scaled_coords = scaling * raw_coords  # scale coordinates
        return scaled_coords

    @sel_frame.setter
    def sel_frame(self, value):
        """
        Ability to select a property of Zecording object (eg for plotting purposes)
        One should prefer to set it either to self.coords, or self.ref_coords or any properties like this.
        It could also be set to any other function (even outside of this class) as long as it
        returns data in the proper format

        Parameters
        ----------
        value: function
            Method to call to get neuron coordinates
        """
        self._sel_coords = value

    def close(self):
        self.file.close()
        logger.info(f'File {self.path} closed')

    @property
    def available_data(self):
        if self._avail_data is None:
            property_names = [p for p in dir(Zecording) if isinstance(getattr(Zecording, p), property)]
            available = [p for p in property_names if p != 'available_data' and self.__getattribute__(p) is not None]
            self._avail_data = set(available)
        return self._avail_data

    @property
    def single_data(self):
        """Names (str) of data sets in Zecording that are a single time traces
        """
        if self._single_data is None:
            default_options = {'behavior', 'stimulus'}
            self._single_data = set()
            for name in default_options:
                if name in self.available_data:
                    self._single_data.add(name)
        return self._single_data

    @single_data.setter
    def single_data(self, value):
        self._single_data = value

    @property
    def multi_data(self):
        """Names (str) of data sets in Zecording that are multi time traces
        """
        if self._multi_data is None:
            self._multi_data = {'df', 'spikes'}
        return self._multi_data

    @multi_data.setter
    def multi_data(self, value):
        self._multi_data = value

    def compute_analysis(self, parent, func, **kwargs):
        """
        Applying some analysis function to a Zecording object
        Named parameters can be passed
        Result is stored in `func_name_res`

        Parameters
        ----------
        parent: the Fishualizer class
            which contains all its functions
        func: function
            Function to apply
            Must have signature of the form func(zecording, kwargs)
        kwargs: dict
            Additional arguments to be passed to `func`

        Returns
        -------

        res
            Result from the analysis function
        res_name: str
            Assigned name to the analysis result as added to the Zecording available data

        """
        res_name = f'{func.__name__}_res'
        parent.statusBar().showMessage(func.__name__)
        res = func(self, parent, **kwargs)
        # setattr(self, res_name, res)  # this is done in the Fishualizer.py in add_static()
        self._avail_data.add(res_name)
        return res, res_name  # return to feed into add_static

    def add_supp_single_data(self, s_name, s_data):
        """Function to add supplementary single data to rec.
        Parameters:
            - s_name, str: name of new data trace
            - s_data, float with shape of single time trace

        """
        setattr(self, s_name, s_data)
        self.available_data.add(s_name)
        self.single_data.add(s_name)

    def reverse_z_coords(self):
        """ Function to reverse the z coordinates.

        The fishualizer cannot rotate 360deg, one might want to change the orientation
        """
        if 'coords' in self.available_data:
            self.coords[:, 2] = self.coords[:, 2].max() + self.coords[:, 2].min() - self.coords[:, 2]
        if 'ref_coords' in self.available_data:
            self.ref_coords[:, 2] = self.ref_coords[:, 2].max() + self.ref_coords[:, 2].min() - self.ref_coords[:, 2]

    def __getitem__(self, dataset: str):
        try:
            return getattr(self, dataset)
        except AttributeError:
            return ValueError(f'{dataset} not a valid data set')

    def __repr__(self) -> str:
        return f'Recording from {self.path}'

    __str__ = __repr__




def create_density_map(gridfile, map_type='density_map', den_threshold=None, den_scale=5):
    """Create 4D matrix which can be used to draw a density map (by Fishualizer.draw_density_map()).

    A grid file is loaded which contains coords (n, 3) and clusters (n, 1) each.
    The resulting 4D matrix is (x, y, z, RGBA). Alternatively; one could think of
    this as 4 different 3D matrices (x, y, z). Every value indicates the Red, Green,
    Blue, Alpha value respectively for the 4 matrices. Importantly; coordinates are NOT
    encoded in the matrices. GLVolumeItem assumes a 1x1x1 grid. This is rescaled
    in this function (using the info from gridfile['coords']).

    Parameters:
    -------------
        gridfile: str
            directory where the grid file with clusters is located
        map_type; str ('density_map', 'hard_threshold')
        den_threshold: float or None
            threshold for cut-off of density map. If None, it defaults to 0 for map_type == 'density_map'
            and to 0,0005 for map_type == 'hard_threshold'
        den_scale: float, int
            if maptype == 'density_map', the density is normalized to the max value, and
            the intensity (alpha) value is subsequently linearly scaled with the normalized
            density, multiplied by density_scale_factor
    """
    hfile = h5py.File(gridfile, 'r')
    data_names = list(hfile['Data'].keys())
    data = {}
    for dn in data_names:  # extract all data sets
        data[dn] = hfile['Data'][dn].value.transpose()

    x_vals = np.unique(data['coords'][:, 0])  # x values in grid
    n_x = len(x_vals)  # number of x values
    y_vals = np.unique(data['coords'][:, 1])
    n_y = len(y_vals)
    z_vals = np.unique(data['coords'][:, 2])
    n_z = len(z_vals)
    resolution = [np.mean(np.diff(x_vals)), np.mean(np.diff(y_vals)), np.mean(
        np.diff(z_vals))]  # it is exported as a cubic grid so resolutions should be equal in all dimensions
    min_coords = data['coords'].min(axis=0)  # minimum used to translate density map in Fishualizer.draw_density_map()

    cluster_names = []
    for dn in data_names:
        if dn != 'coords':
            data[dn + '_nf'] = np.reshape(data[dn], (n_x, n_y, n_z))  # put cluster densities in new format (1D -> 3D)
            cluster_names.append(dn)
    # nf_cluster_names = [x for x in list(data.keys()) if x[-3:] == '_nf']  # list of names to use
    # colours = {nf_cluster_names[0]: [255, 0, 0, 0],
    #             nf_cluster_names[1]: [0, 255, 0, 0],
    #             nf_cluster_names[2]: [0, 0, 255, 0],
    #             nf_cluster_names[3]: [128, 128, 0, 0],
    #             nf_cluster_names[4]: [0, 128, 128, 0]}  # colours of clusters  # TODO: import color dict
    nf_cluster_names = ['positive_mixed_nf', 'negative_mixed_nf', 'posnegderiv_high95_nf']
    colours = {nf_cluster_names[0]: [255, 0, 113, 0],
               nf_cluster_names[1]: [0, 255, 157, 0],
               nf_cluster_names[2]: [184, 134, 11, 0]}  # hard coded colors of regression clusters Migault et al., 2018
    # maxnorm = {cn: data[cn].max() for cn in nf_cluster_names}  # max density per cluster (for colour normalization)
    maxnorm = {cn: 0.0005 for cn in nf_cluster_names}  # uniform max density (for colour normalization)

    dataplot = np.zeros((n_x, n_y, n_z) + (4,), dtype=np.ubyte)  # create 4D data matrix to plot (x,y,z,RGBA)

    ## Assign RGBA values in series
    if den_threshold is None:
        if map_type == 'density_map':
            den_threshold = 0  # DENSITY MAP
        elif map_type == 'hard_threshold':
            den_threshold = 0.00005  # HARD THRESHOLD MAP
    for x in range(n_x):  # loop through all coords to assign RGBA
        for y in range(n_y):
            for z in range(n_z):
                max_den = 0
                for cn in nf_cluster_names:  # check all clusters to find max one
                    if (data[cn][x, y, z] > den_threshold) and (data[cn][x, y, z] > max_den):
                        max_den = np.maximum(data[cn][x, y, z], max_den)  # DENSITY MAP
                        dataplot[x, y, z, :] = colours[cn]
                        if map_type == 'density_map':
                            dataplot[x, y, z, 3] = (max_den / maxnorm[cn] * 100) * den_scale  # DENSITY MAP
                        elif map_type == 'hard_threshold':
                            dataplot[x, y, z, 3] = 100  # HARD THRESHOLD MAP

    return dataplot, resolution, min_coords


def load_zbrain_regions(recording, zbrainfile=None):
    """Load ZBrainAtlas regions that are saved in the custom-format .h5 file.

    Parameters:
    ------------
        recording: instance of Zecording class
            Data is added to this recording
        zbrainfile: str (default None)
            directory where file is located, if None it defaults to hard-coded dir.

    Returns:
    ----------
        bool: indicating success

    """
    if zbrainfile is None:
        zbrainfile = 'Content/ZBrainAtlas_Outlines.h5'

    if zbrainfile[-3:] == '.h5':
        hfile = h5py.File(zbrainfile, 'r')
        data_names = list(hfile.keys())
        data = {}
        for dn in data_names:  # extract all data sets
            data[dn] = hfile[dn].value.transpose()

        if 'region_indices' in data_names and 'grid_coordinates' in data_names:
            data['region_indices'] = data['region_indices'].astype(
                'int') - 1  # convert from 1-indexing (matlab) to zero-indexing

            ## Below: change to correct orientation
            max_grid_coords = np.squeeze(data['resolution']) * np.squeeze([data['height'], data['width'], data['Zs']])
            long_axis_flipped = max_grid_coords[0] - data['grid_coordinates'][:, 0]
            # data['grid_coordinates'][:, 0] = long_axis_flipped.copy()
            data['grid_coordinates'][:, 0], data['grid_coordinates'][:, 1] = (data['grid_coordinates'][:, 1]).copy(), (
                data['grid_coordinates'][:, 0]).copy()
            data['grid_coordinates'] = data['grid_coordinates'] / 1000  # go to mm

            setattr(recording, 'zbrainatlas_coordinates', data['grid_coordinates'])  # add to recording
            recording.available_data.add(
                'zbrainatlas_coordinates')  # don't use [..]_coords here because coords is used for plotting
            setattr(recording, 'zbrainatlas_regions', data['region_indices'])
            recording.available_data.add('zbrainatlas_regions')
            logger.info('ZBrainAtlas succesfully added')
            return True
        else:
            logger.warning(
                f'ZBrainAtlas not loaded because region_indices and grid_coordinates were not found in the file {zbrainfile}')
            return False
