import os
from pathlib import Path
import numpy as np
from numpy import asarray
import pandas as pd
from scipy import signal as sp_sig
from mne.io import read_raw_edf as read_edf

from .montage import Montage


__all__ = ['EDF', 'EDFCollector', 'collect_files']


class EDF(object):
    SCALER = {
        'volt': lambda unit: 1e-6/unit,
        'minivolt': lambda unit: 1e-3/unit,
        'microvolt': lambda unit: 1/unit,
    }
    RESCALE_METHOD = ['min_max', 'mean_std']
    def __init__(self, raw, config_name=None, montage=None):
        self.raw = raw
        self.fs = raw.info['sfreq']
        self.nchan = raw.info['nchan']
        self.duration = raw.n_times/self.fs
        self.extras = raw._raw_extras[0]
        assert len(raw._raw_extras) == 1, (
            'Size of `_raw_extras` in given EDF is not 1. It might not '
            'be the format we supported.'
        )

        if montage and isinstance(montage, Montage):
            self.montage = montage
        else:
            self.montage = Montage(raw, config_name=config_name)

    @property
    def channels(self):
        return asarray(self.montage.names)

    @property
    def pmax(self):
        return asarray(self.extras['physical_max'])[self.montage.idx_units]

    @property
    def pmin(self):
        return asarray(self.extras['physical_min'])[self.montage.idx_units]

    @property
    def dmax(self):
        return asarray(self.extras['digital_max'])[self.montage.idx_units]

    @property
    def dmin(self):
        return asarray(self.extras['digital_min'])[self.montage.idx_units]

    @property
    def units(self):
        return asarray(self.extras['units'])[self.montage.idx_units]

    @property
    def orig_units(self):
        return self.raw._orig_units

    def set_bipolar_reference(self, val):
        # https://mne.tools/stable/generated/mne.set_bipolar_reference.html
        # self.raw.set_montage()
        raise NotImplementedError

    @classmethod
    def from_file(cls, fn, config_name=None):
        """
        Parameters
        ----------
        fn : str
            File name.
        config_name : str, default is None
            Montage configuration of file. If it is not given, all channels in
            EDF file will be used directly. (no matter they are unipolar or
            bipolar references)
        """
        return cls(read_edf(str(fn)), config_name=config_name)

    def get_data(self, t_start=0, t_stop=None, channels=None,
        unit='microvolt', rescale=None, resample_fs=None):
        """
        Parameters
        ----------
        t_start : float
            Start time of a segment. (unit: second)
        t_stop : float
            Stop time of a segment. (unit: second)
        channels : list, default is None
            Channels to read. When it is None and `auto_select` is False, all
            available channels will be read (including those non-EEG ones).
        unit : str or None, default is 'microvolt'
            Rescale data in specified magnitude. If None is given, no scale will
            be applied. Note that this might fail if not all channels are sampled
            in the same magnitude.
        rescale : one of ['min_max', 'mean_std'], default is None
            Rescale data by specified method. Note that effect of setting `unit`
            will be ignored if this argument is set. Besides, those extremes for
            'min_max' normalization used here are `pmin` and `pmax` (phsyical).
        resample_fs : int, default is None
            If this value is given, resample signal by given frequency.

        Returns
        -------
        data : ndarray, ndim: 2 (channel, signal)
        """
        if unit and unit not in self.SCALER:
            valid_units = [v for v in self.SCALER]
            raise ValueError(f'Given `unit` should be one of {valid_units}')

        if rescale and rescale not in self.RESCALE_METHOD:
            raise ValueError(f'Given `rescale` should be one of {self.RESCALE_METHOD}')

        if channels:
            missing = np.setdiff1d(channels, self.channels)
            if missing:
                raise ValueError(f'Given channels {missing} are in current montage.')

        channels = asarray(channels) if channels else self.channels

        # Decouple specified channels to those channels listed in raw data
        anodes, cathodes = self.montage.decouple(channels)
        if self.montage.is_all_unipolar:
            picks = anodes
        else:
            picks = np.array(list(set(anodes) | set(cathodes)))

        # Get signal chunck from raw data
        s_start = int(t_start*self.fs)
        s_stop = int(t_stop*self.fs) if t_stop is not None else None
        data = self.raw.get_data(picks=picks, start=s_start, stop=s_stop)

        # Reorder and composite signal according to montage
        data = self.montage.reorder_data_by_channels(data, picks, channels)

        if resample_fs and not np.isclose(self.fs, resample_fs):
            data = self.resample(data, self.fs, resample_fs)

        idx_channels = np.nonzero(channels[:, None] == self.channels)[1]
        selected_units = self.montage.units[idx_channels]

        # Replace 1D array with a scalar to slightly improve performance of
        # multiplication
        if np.allclose(selected_units, selected_units[0]):
            selected_units = selected_units[0]

        if unit and rescale is None:
            scale = self.SCALER[unit](selected_units)
            return data*scale
        elif rescale:
            if rescale == 'min_max':
                pmin = self.pmin[idx_channels]*selected_units
                pmax = self.pmax[idx_channels]*selected_units
                return self.normalize(data, pmin, pmax)
            elif rescale == 'mean_std':
                return self.standardize(data)
            else:
                raise ValueError(f'Unknown rescale method: {rescale}')
        else:
            return data

    # TODO: move this method to a new class (because this class doesn't have
    # a map of annotaion, e.g. {'bckg': 0, 'tcsz': 1} ...)
    # TODO: considering not replace string annotation to digital? Because annotation
    # map might be different in EDFs.
    def anno_to_samples(self, t_start, t_stop, sample_length, channels=None):
        """ Convert interval of annotations to discrete samples.
        e.g. For a 10-seconds segment of signal `x`, given an annotation:
            {'seiz': [3.0, 6.0]}
            >>> x.anno_to_samples(0, 8, 1)
            []
        """
        pass

    @staticmethod
    def resample(data, fs_ori, fs_new, window='boxcar'):
        """
        Parameters
        ----------
        data : ndarray
            Data to be resampled.
        fs_ori : int
            Original sampling frequency.
        fs_new : int
            New sampling frequency.
        window : str
            Same functionality declared in `mne.io.BaseRaw.resample`.
            A frequency-domain window to use in resampling.
        """
        if data.ndim != 2:
            raise ValueError('Dimension of data should be 2: (channels, samples).')
        duration = data.shape[1] / fs_ori
        num = int(duration * fs_new)
        return sp_sig.resample(data, num, axis=1, window=window)

    @staticmethod
    def normalize(data, pmin, pmax):
        data = asarray(data)
        pmin, pmax = asarray(pmin), asarray(pmax)
        if data.ndim != 2:
            ValueError('Dimension of given data should be 2.')
        assert pmin.ndim == pmax.ndim, (
            f'Dimension of `pmin` ({pmin.ndim}) and `pmax` ({pmin.ndim}) is not the same'
        )
        assert pmin.ndim <= data.ndim, (
            'Dimension of `pmin` and `pmax` should be less or equal to 2.'
        )
        if pmin.ndim == 0 or pmin.ndim == 2:
            return (data - pmin)/(pmax - pmin)
        else:
            return (data - pmin[..., None])/(pmax - pmin)[..., None]

    @staticmethod
    def standardize(data, axis=1):
        """ Standardize data. Default axis is 1 (i.e. applying along channel) """
        data = asarray(data)
        if data.ndim != 2:
            ValueError('Dimension of given data should be 2.')
        assert axis < 2, 'Given axis should be less than 2'
        mean, std = np.mean(data, axis=axis), np.std(data, axis=axis)
        if axis == 0:
            return (data - mean)/std
        else:
            return (data - mean[..., None])/std[..., None]


class EDFCollector(object):
    @classmethod
    def from_dir(cls, dir_name, depth=4):
        """ Collect *.edf files from given directory.
        Note that given directory should be the same pattern given by official dataset.
        (TUH EEG v1.1.0 convention)

        e.g.
        Filename:
        "002/00000258/s002_2003_07_21/00000258_s002_t000.edf"

        Components:
            002: 3-digit identifier
            00000258: patient number
            s002_2003_07_21: number of session followed by date (YYYY_MM_DD)
            00000258_s002_t000.edf: EEG file

        Parameters
        ----------
        dir_name : str
            Name of directory.
        depth : int. optional
            Depth of recursive search.

        Returns
        -------
        file_list : list
            List of `pathlib.Path` object.
        """
        print('Collecting edf files from: "%s", ...' % dir_name)

        file_list = []
        collect_files(dir_name, file_list, '*.edf', depth=depth)

        print('\x1b[1ACollecting edf files from: "%s", DONE' % dir_name)
        return file_list


def collect_files(root, collection, pattern, depth=1):
    current_depth = depth - 1
    if current_depth == 0:
        collection.extend(Path(root).glob(pattern))
        return

    for v in os.scandir(root):
        if not v.is_dir: continue
        collect_files(v, collection, pattern, current_depth)
