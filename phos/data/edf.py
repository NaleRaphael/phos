import os
from pathlib import Path
import numpy as np
from numpy import asarray
import pandas as pd
from scipy import signal as sp_sig
from mne.io import read_raw_edf as read_edf

from .montage import Montage
from .annotation import AnnotationProcessor


__all__ = ['EDF', 'EDFCollector', 'collect_files']


class EDF(object):
    SCALER = {
        'volt': lambda unit: 1e-6/unit,
        'minivolt': lambda unit: 1e-3/unit,
        'microvolt': lambda unit: 1/unit,
    }
    RESCALE_METHOD = ['min_max', 'mean_std']

    def __init__(self, raw, config_name=None, montage=None):
        """
        Parameters
        ----------
        raw : mne.io.edf.edf.RawEDF
            Raw edf object.
        config_name : str, default is None
            Configuration of montage. Possible values are listed in
            `montage.DEFAULT_MONTAGE_DICT`. For further details, please
            checkout the docstring of `Montage` class.
        montage : Montage, default is None
            Montage for current EDF file.
        """
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

    @classmethod
    def from_file(cls, fn, config_name=None, verbose=False):
        """
        Parameters
        ----------
        fn : str
            File name.
        config_name : str, default is None
            Montage configuration of file. If it is not given, all channels in
            EDF file will be used directly. (no matter they are unipolar or
            bipolar references)
        verbose : bool, default is False
        """
        return cls(read_edf(str(fn), verbose=verbose), config_name=config_name)

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
            Channels to read. If not specified, all channels defined in montaged
            will be used. (including those non-EEG ones)
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
                raise ValueError(f'Given channels {missing} are not in current montage.')

        channels = asarray(channels) if channels else self.channels

        # Decompose specified channels to those channels listed in raw data
        anodes, cathodes = self.montage.decompose(channels)
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
    # TODO: considering not replacing string annotation to digital? Because
    # annotation map might be different in EDFs.
    def get_annotations(self, t_start, t_stop, sample_length, channels=None,
        cache_raw=True, allow_annotation_overlapping=False,
        select_overlapping='right'):
        """ Convert interval of annotations to discrete samples.
        e.g. For a 10-seconds segment of signal `x`, given an annotation:
            {'seiz': [3.0, 6.0]}
            >>> x.get_annotations(0, 8, 1)
            [0, 0, 0, 1, 1, 1, 0, 0]

        Parameters
        ----------
        t_start : float
            Start time of a segment. (unit: second)
        t_stop : float
            Stop time of a segment. (unit: second)
        sample_length : float
            Length of a sample. (unit: second)
        channels : list, default is None
            Channels to read. If not specified, all channels defined in montaged
            will be used. (including those non-EEG ones)
        cache_raw : bool, default is True
            Cache content of label file.
        allow_annotation_overlapping : bool, default is False
            Raise an error if there is overlapping intervals of annotations.
        select_overlapping: str, default is 'right'
            Method for selecting annotations in overlapping region. e.g. Given
            label content: [(0, 'A', 0., 12.), (0, 'B', 8., 20.), ...] in the
            format of (channel, annotation, t_start, t_stop). There is an
            overlapping at time interval [8., 12.]. While time interval of
            required sample is [9., 13.], and `sample_length` is 4.:
            - 'right':
                Compare middle point with `t_start` of each overlapping intervals.
                Middle point of given interval is (9. + 13.)/2 = 11., Which is
                closer to the `t_start` of second interval with annotation 'B'.
                Therefore, `[['B']]` is returned.
            - 'left':
                Compare middle point with `t_stop` of each overlapping intervals.
                In this case, middle point is closer to the `t_stop` of first
                interval. Therefore, `[['A']]` is returned.
        """
        # NOTE: In TUSZ dataset, annotations are annotated in according to montage
        # rather than raw channels.
        if channels is not None:
            channels = np.asarray(channels)
            missing = np.setdiff1d(channels, self.channels)
            if missing:
                raise ValueError(f'Given channels {missing} are not in current montage.')
        else:
            channels = self.channels
        idx_channels = self.montage.channel_to_index(channels)

        fn_lbl = Path(self.raw.filenames[0]).with_suffix('.lbl.npy')
        ary_lbl = np.load(fn_lbl)
        assert ary_lbl.dtype == AnnotationProcessor.OUTPUT_FMT, (f'Incorrect format of'
            ' given label file, should be {AnnotationProcessor.OUTPUT_FMT}'
        )
        if cache_raw:
            self.ary_lbl = ary_lbl

        n_sample = int(np.ceil((t_stop - t_start)/sample_length))
        result = []

        for i, idx_ch in enumerate(idx_channels):
            ch_info = ary_lbl[ary_lbl['ch'] == idx_ch]
            anno = ch_info['anno']

            ts, te = ch_info['ts'][:, None], ch_info['te'][:, None]
            mid_points = np.arange(t_start, t_stop, sample_length) + sample_length/2
            samples = np.tile(mid_points, (ch_info.size, 1))
            mask = ((ts <= samples) & (samples <= te)).astype(int)

            n_anno = np.sum(mask, axis=0)
            n_anno_required = np.ones(n_sample, dtype=int)

            # Check whether there is any sample annotated by multiple annotations, which
            # indicates there might be an overlapping region across time intervals.
            # - Found intervals with multiple annotations (e.g. overlapping interval)
            if np.any(n_anno > n_anno_required):
                idx_bad_samples = np.where(n_anno > n_anno_required)[0]
                if not allow_annotation_overlapping:
                    msg = (
                        'Multiple annotations found at %s-th sample.\n'
                        'Information:\n'
                        '  (t_start, t_stop, sample_length): (%s, %s, %s)\n'
                        '  Label file: %s'
                        % (idx_bad_samples, t_start, t_stop, sample_length, fn_lbl)
                    )
                    raise RuntimeError(msg)
                else:
                    # handle overlapping
                    if select_overlapping == 'right':
                        dist = mid_points - ts
                    else:
                        dist = te - mid_points
                    idx_min_dist = np.argmin(dist[idx_bad_samples], axis=0)
                    new_mask = np.zeros_like(mask[:, idx_bad_samples])
                    new_mask[idx_min_dist, idx_bad_samples] = 1
                    mask = new_mask

            # - Found intervals without defined annotations
            if np.any(n_anno < n_anno_required):
                msg = (
                    'Found intervals without definition.\n'
                    'Information:\n'
                    '  (t_start, t_stop, sample_length): (%s, %s, %s)\n'
                    '  Label file: %s'
                    % (idx_bad_samples, t_start, t_stop, sample_length, fn_lbl)
                )
                raise RuntimeError(msg)

            # shape of desired output: (1, n_sample)
            # shape of mask: (anno.size, n_sample)
            result.append(np.matmul(anno[None, :], mask).reshape(-1))

        result = np.stack(result)
        return result

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
