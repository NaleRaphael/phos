from bisect import bisect_right
from pathlib import Path
import re

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

from .data import (
    EDF, Montage, EDFCollector, AnnotationProcessor,
)


rand = np.random.rand
searchsorted = np.searchsorted

# Regex for finding montage config from file path
# current available: tcp_ar, tcp_le, tcp_ar_a
regex_montage_config = re.compile(r'_(tcp_\w+)')


class TUSZDataset(Dataset):
    _DATASET_KINDS = ['train', 'dev', 'eval']
    _DATASET_ENTRIES = ['01_tcp_ar', '02_tcp_le', '03_tcp_ar_a']
    def __init__(self, datasets, kind, sample_length):
        if not all([isinstance(ds, EDFDataset) for ds in datasets]):
            raise TypeError('Some datasets are not `EDFDataset`.')
        self.datasets = datasets
        self.kind = kind
        self.sample_length = sample_length
        self._files = pd.concat([ds.files for ds in datasets], ignore_index=True)
        self._durations = pd.concat([ds.durations for ds in datasets], ignore_index=True)
        self.cumulative_sizes = np.cumsum([len(ds) for ds in datasets])
        self.cumsum = np.cumsum(self._durations.values)

    @property
    def files(self):
        return self._files

    @property
    def durations(self):
        return self._durations

    @classmethod
    def load_from(
        cls, root, kind, sample_length, force_update=False, csv_file=None,
        extra_dir=None, aug_shift=False, desired_fs=None, return_labels=False
    ):
        """
        Parameters
        ----------
        root : str
            Root directory of dataset. Structure of sub-directories should follow
            the one of official TUSZ dataset.
        kind : str
            Kind of dataset.
        sample_length : int
            Length of a sample. (unit: second)
        force_update : bool
            See also the definition in `EDFDataset.from_dir()`.
        csv_file : str, pathlib.Path or None
            A csv file that contains names and paths of files in dataset.
        extra_dir : str, pathlib.Path or None
            Directory of files containing extra information for dataset preprocessing.
            - calibration_[kind].csv:
                Calibration interval of data.
                Fields: ['filename', 'start', 'stop']
        aug_shift : bool, default is False
            See also the definition in `EDFDataset.__init__()`.
        desired_fs : int, default is None
            See also the definition in `EDFDataset.__init__()`.
        return_labels : bool, default is False
            See also the definition in `EDFDataset.__init__()`.
        """
        if kind not in cls._DATASET_KINDS:
            raise ValueError('Given `kind` should be one of %s' % cls._DATASET_KINDS)
        if return_labels and kind == 'eval':
            raise ValueError('Annotations are not available in "eval" dataset.')

        if extra_dir:
            fn_calb = 'calibration_%s.csv' % kind
            calb_file = Path(extra_dir, fn_calb)

            # Extract required calibration info automatically
            if not calb_file.exists():
                from .data import extract_calibration_info
                extract_calibration_info(extra_dir)
        else:
            calb_file = None

        datasets = []

        for entry in cls._DATASET_ENTRIES:
            csv_file = Path(extra_dir, 'edfs_%s.csv' % entry)
            ds = EDFDataset.from_dir(
                Path(root, entry), sample_length,
                force_update=force_update, csv_file=csv_file,
                calb_file=calb_file, aug_shift=aug_shift,
                desired_fs=desired_fs, return_labels=return_labels
            )
            datasets.append(ds)

        return cls(datasets, kind, sample_length)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, idx):
        # impl taken from `torch.ConcatDataset`
        if isinstance(idx, (tuple, list)):
            channels, idx = idx
        else:
            channels = None
        if idx < 0:
            if -idx > len(self):
                raise ValueError('absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][channels, sample_idx]


class EDFDataset(Dataset):
    HEADER = ['filename', 'duration']
    CALB_HEADER = ['filename', 'start', 'stop']
    def __init__(
        self, content, sample_length, aug_shift=False, desired_fs=None,
        return_labels=False
    ):
        """
        Parameters
        ----------
        content : pd.DataFrame
        sample_length : int
            Length of a sample. (unit: second)
        aug_shift : bool, default is False
            As a simple augmentation, signal will be shifted with a random offset
            while retrieving it.
        desired_fs : int, default is None
            If this value is given, retrieved signal segment will be resampled by
            this frequency if its sampling frequency is not equal to this one.
        return_labels : bool, default is False
            If true, annotation of each sample will be returned.
        """
        if not isinstance(content, pd.DataFrame):
            raise TypeError('Given `content` should be a `pd.DataFrame`.')
        if not all(content.columns == self.HEADER):
            raise ValueError('Header of `content` should be %s' % self.HEADER)
        if isinstance(content.filename[0], str):
            content.filename = content.filename.apply(lambda x: Path(x))
        self.content = content
        self.sample_length = sample_length
        self.aug_shift = aug_shift
        self.desired_fs = desired_fs
        self.return_labels = return_labels
        self.cumsum = np.cumsum(content.duration.values)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, idx):
        return self.get_item(idx)

    @property
    def files(self):
        return self.content['filename']

    @property
    def durations(self):
        return self.content['duration']

    @classmethod
    def from_dir(
        cls, dir_name, sample_length, force_update=False, csv_file=None,
        calb_file=None, aug_shift=False, desired_fs=None, return_labels=False
    ):
        print('Loading EDF files from "%s" ...' % dir_name)
        if not force_update and csv_file and Path(csv_file).exists():
            print(
                'List of EDFDataset has been created. '
                'Loading from it at: "%s"' % (Path(csv_file).as_posix())
            )
            return cls(
                pd.read_csv(csv_file), sample_length, aug_shift=aug_shift,
                desired_fs=desired_fs, return_labels=return_labels
            )

        edf_list = EDFCollector.from_dir(dir_name)

        cls.process_annotation(
            edf_list, force_update=force_update, calb_file=calb_file
        )
        durations = cls.collect_durations(edf_list)

        df = pd.DataFrame(zip(edf_list, durations), columns=cls.HEADER)
        df['filename'] = df['filename'].apply(lambda x: x.absolute().as_posix())
        df.to_csv(csv_file, header=cls.HEADER, index=False)

        return cls(
            df, sample_length, aug_shift=aug_shift, desired_fs=None,
            return_labels=return_labels
        )

    @classmethod
    def process_annotation(cls, edf_list, force_update=False, calb_file=None):
        print('Processing annotation files ...')
        anno_proc = AnnotationProcessor()

        if calb_file:
            calb_df = pd.read_csv(calb_file, header=0)
            assert set(calb_df.columns) == set(cls.CALB_HEADER), (
                'Fields in `calb_file` should be %s' % cls.CALB_HEADER
            )
            calb_map = {
                r['filename']: [r['start'], r['stop']]
                for i, r in calb_df.iterrows()
            }
        else:
            calb_map = {}

        for fn in tqdm(edf_list, ascii=True):
            fn_label_npy = fn.absolute().with_suffix('.lbl.npy')
            if fn_label_npy.exists() and not force_update: continue

            anno = anno_proc.convert(fn.absolute().with_suffix('.lbl'))
            fn_base = fn.name.split('.')[0]
            if fn_base in calb_map:
                ts, te = calb_map[fn_base]
                anno_calb = {'anno': 27, 'ts': ts, 'te': te, 'prob': 1.}

                # Add annotations "calb" (index: 27) which indicate those segments
                # of signal calibration. See also "_DOCS/02_annot.pdf" - Table 1.
                anno = anno_proc.add_annotation(anno, anno_calb, sort_by='ch')
            np.save(fn_label_npy, anno)

    @classmethod
    def collect_durations(cls, edf_list):
        print('Collecting duration of files ...')
        durations = np.empty(len(edf_list))
        for i, fn in tqdm(enumerate(edf_list), total=len(edf_list), ascii=True):
            durations[i] = EDF.from_file(str(fn.absolute())).duration
        return durations


    def get_item(self, idx):
        if isinstance(idx, (tuple, list)):
            channels, idx = idx
        else:
            channels = None
        index, offset = self.get_index_offset(idx)
        return self.get_chunk(index, offset, channels=channels)

    def get_index_offset(self, idx):
        """ Get real index of files and time offset in that EDF file.

        Parameters
        ----------
        idx : int
            Index passed from `dataset.__getitem__(idx)`.

        Returns
        -------
        index : int
            Index of selected file.
        offset : float
            Time offset of selected file. (unit: second)
        """
        # Impl taken from `jukebox.FilesAudioDataset` for virtualizing mulitple
        # time series datum as a huge stream. Note that our implement is a bit
        # different from to the one in jukebox. Because the offset we calculating
        # here is in the unit of second.
        sl = self.sample_length
        half_interval = sl/2

        # Generate random shift, range: [-0.5, 0.5]*sample_length
        shift = (rand()-0.5)*sl if self.aug_shift else 0.
        offset = idx*sl + shift
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], (
            f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        )
        index = searchsorted(self.cumsum, midpoint)

        # Get start and end of selected file
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index]
        assert start <= midpoint <= end, (
            f'Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}'
        )

        # Clip the offset in range
        if offset > end - sl:
            offset = max(start, offset - half_interval)
        elif offset < start:
            offset = min(end - sl, offset + half_interval)
        assert start <= offset <= end - sl, (
            f'Offset {offset} not in [{start}, {end - sl}]. '
            f'End: {end}, SL: {sl}, Index: {index}'
        )

        offset = offset - start
        return index, offset

    def get_chunk(self, index, offset, channels=None):
        fn = str(self.files[index])
        match = regex_montage_config.search(fn)
        if match is None:
            raise RuntimeError('Failed to parse config of montage from file path.')
        config_name = match.groups()[0]
        edf = EDF.from_file(fn, config_name=config_name)
        t_start, t_stop = offset, offset + self.sample_length
        chunk = edf.get_data(
            t_start=t_start, t_stop=t_stop, channels=channels,
            resample_fs=self.desired_fs
        )
        if self.return_labels:
            anno = edf.get_annotations(
                t_start, t_stop, self.sample_length, channels=channels
            )
            return chunk, anno
        else:
            return chunk
