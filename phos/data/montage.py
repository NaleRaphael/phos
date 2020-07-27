import numpy as np
from numpy import asarray
from mne.io.edf.edf import RawEDF


__all__ = ['Montage']


# Format: [ch_name], [anode], [cathode]
DEFAULT_MONTAGE_DICT = {}


class Channel(object):
    __slots__ = ('name', 'anode', 'cathode')
    def __init__(self, name, anode, cathode=None):
        self.name = name
        self.anode = anode
        self.cathode = cathode

    def __repr__(self):
        return f'"{self.name}": "{self.anode}" -- "{self.cathode}"'

_Ch = Channel


def check_initialized(value_to_check):
    def outter_wrapper(func):
        def wrapper(*args, **kwargs):
            if not value_to_check:
                func()
        return wrapper
    return outter_wrapper


def cache_montage(func):
    memo = {}
    def wrapper(*args, **kwargs):
        update_cache = kwargs.get('update_cache', False)
        if func.__name__ in memo and not update_cache:
            return memo[func.__name__]
        result = func(*args, **kwargs)
        memo[func.__name__] = result
        return result
    return wrapper


@check_initialized(DEFAULT_MONTAGE_DICT)
def init_default_bipolar_montage():
    DEFAULT_MONTAGE_DICT['tcp_ar'] = {
        0:  _Ch('FP1-F7', 'EEG FP1-REF', 'EEG F7-REF'),
        1:  _Ch('F7-T3',  'EEG F7-REF',  'EEG T3-REF'),
        2:  _Ch('T3-T5',  'EEG T3-REF',  'EEG T5-REF'),
        3:  _Ch('T5-O1',  'EEG T5-REF',  'EEG O1-REF'),
        4:  _Ch('FP2-F8', 'EEG FP2-REF', 'EEG F8-REF'),
        5:  _Ch('F8-T4',  'EEG F8-REF',  'EEG T4-REF'),
        6:  _Ch('T4-T6',  'EEG T4-REF',  'EEG T6-REF'),
        7:  _Ch('T6-O2',  'EEG T6-REF',  'EEG O2-REF'),
        8:  _Ch('A1-T3',  'EEG A1-REF',  'EEG T3-REF'),
        9:  _Ch('T3-C3',  'EEG T3-REF',  'EEG C3-REF'),
        10: _Ch('C3-CZ',  'EEG C3-REF',  'EEG CZ-REF'),
        11: _Ch('CZ-C4',  'EEG CZ-REF',  'EEG C4-REF'),
        12: _Ch('C4-T4',  'EEG C4-REF',  'EEG T4-REF'),
        13: _Ch('T4-A2',  'EEG T4-REF',  'EEG A2-REF'),
        14: _Ch('FP1-F3', 'EEG FP1-REF', 'EEG F3-REF'),
        15: _Ch('F3-C3',  'EEG F3-REF',  'EEG C3-REF'),
        16: _Ch('C3-P3',  'EEG C3-REF',  'EEG P3-REF'),
        17: _Ch('P3-O1',  'EEG P3-REF',  'EEG O1-REF'),
        18: _Ch('FP2-F4', 'EEG FP2-REF', 'EEG F4-REF'),
        19: _Ch('F4-C4',  'EEG F4-REF',  'EEG C4-REF'),
        20: _Ch('C4-P4',  'EEG C4-REF',  'EEG P4-REF'),
        21: _Ch('P4-O2',  'EEG P4-REF',  'EEG O2-REF'),
    }

    DEFAULT_MONTAGE_DICT['tcp_le'] = {
        0:  _Ch('FP1-F7', 'EEG FP1-LE', 'EEG F7-LE'),
        1:  _Ch('F7-T3',  'EEG F7-LE',  'EEG T3-LE'),
        2:  _Ch('T3-T5',  'EEG T3-LE',  'EEG T5-LE'),
        3:  _Ch('T5-O1',  'EEG T5-LE',  'EEG O1-LE'),
        4:  _Ch('FP2-F8', 'EEG FP2-LE', 'EEG F8-LE'),
        5:  _Ch('F8-T4',  'EEG F8-LE',  'EEG T4-LE'),
        6:  _Ch('T4-T6',  'EEG T4-LE',  'EEG T6-LE'),
        7:  _Ch('T6-O2',  'EEG T6-LE',  'EEG O2-LE'),
        8:  _Ch('A1-T3',  'EEG A1-LE',  'EEG T3-LE'),
        9:  _Ch('T3-C3',  'EEG T3-LE',  'EEG C3-LE'),
        10: _Ch('C3-CZ',  'EEG C3-LE',  'EEG CZ-LE'),
        11: _Ch('CZ-C4',  'EEG CZ-LE',  'EEG C4-LE'),
        12: _Ch('C4-T4',  'EEG C4-LE',  'EEG T4-LE'),
        13: _Ch('T4-A2',  'EEG T4-LE',  'EEG A2-LE'),
        14: _Ch('FP1-F3', 'EEG FP1-LE', 'EEG F3-LE'),
        15: _Ch('F3-C3',  'EEG F3-LE',  'EEG C3-LE'),
        16: _Ch('C3-P3',  'EEG C3-LE',  'EEG P3-LE'),
        17: _Ch('P3-O1',  'EEG P3-LE',  'EEG O1-LE'),
        18: _Ch('FP2-F4', 'EEG FP2-LE', 'EEG F4-LE'),
        19: _Ch('F4-C4',  'EEG F4-LE',  'EEG C4-LE'),
        20: _Ch('C4-P4',  'EEG C4-LE',  'EEG P4-LE'),
        21: _Ch('P4-O2',  'EEG P4-LE',  'EEG O2-LE'),
    }

    DEFAULT_MONTAGE_DICT['tcp_ar_a'] = {
        0:  _Ch('FP1-F7', 'EEG FP1-REF', 'EEG F7-REF'),
        1:  _Ch('F7-T3',  'EEG F7-REF',  'EEG T3-REF'),
        2:  _Ch('T3-T5',  'EEG T3-REF',  'EEG T5-REF'),
        3:  _Ch('T5-O1',  'EEG T5-REF',  'EEG O1-REF'),
        4:  _Ch('FP2-F8', 'EEG FP2-REF', 'EEG F8-REF'),
        5:  _Ch('F8-T4',  'EEG F8-REF',  'EEG T4-REF'),
        6:  _Ch('T4-T6',  'EEG T4-REF',  'EEG T6-REF'),
        7:  _Ch('T6-O2',  'EEG T6-REF',  'EEG O2-REF'),
        8:  _Ch('T3-C3',  'EEG T3-REF',  'EEG C3-REF'),
        9:  _Ch('C3-CZ',  'EEG C3-REF',  'EEG CZ-REF'),
        10: _Ch('CZ-C4',  'EEG CZ-REF',  'EEG C4-REF'),
        11: _Ch('C4-T4',  'EEG C4-REF',  'EEG T4-REF'),
        12: _Ch('FP1-F3', 'EEG FP1-REF', 'EEG F3-REF'),
        13: _Ch('F3-C3',  'EEG F3-REF',  'EEG C3-REF'),
        14: _Ch('C3-P3',  'EEG C3-REF',  'EEG P3-REF'),
        15: _Ch('P3-O1',  'EEG P3-REF',  'EEG O1-REF'),
        16: _Ch('FP2-F4', 'EEG FP2-REF', 'EEG F4-REF'),
        17: _Ch('F4-C4',  'EEG F4-REF',  'EEG C4-REF'),
        18: _Ch('C4-P4',  'EEG C4-REF',  'EEG P4-REF'),
        19: _Ch('P4-O2',  'EEG P4-REF',  'EEG O2-REF'),
    }


def find_common_channels(montage_dict):
    from functools import reduce

    ref_name_sets = []
    for montage in DEFAULT_MONTAGE_DICT.values():
        ref_name_sets.append(set([ch.name for ch in montage.values()]))
    return reduce(lambda res, ele: res & ele, ref_name_sets)


@cache_montage
def get_common_montage_from_config(config_name, update_cache=False):
    montage_dict = DEFAULT_MONTAGE_DICT[config_name]
    common_channels = find_common_channels(montage_dict)
    return {idx: ch for idx, ch in montage_dict.items() if ch.name in common_channels}


class Montage(object):
    def __init__(self, raw, config_name=None, channels=None, use_common_montage=True):
        """
        Parameters
        ----------
        raw : mne.io.edf.edf.RawEDF
        config_name : str, default is None
            Configuration of montage, possible values are listed in `DEFAULT_MONTAGE_DICT`.
        channels : array_like of Channel objects or tuple with 3 fields:
                   [channel_name, anode, cathode], default is None
            If this value and `config_name` are both None, `raw.ch_names` will
            be used.
        use_common_montage: bool, default is True
            If `config_name` is given and this value is True, common channels
            in those montages defined in `DEFAULT_MONTAGE_DICT` will be used.
            Otherwise, channels of montage defined by `config_name` will be used.
        """
        if not isinstance(raw, RawEDF):
            raise TypeError(f'Given raw object should be an instance of {RawEDF}.')

        assert len(raw._raw_extras) == 1, (
            'Size of `_raw_extras` in given EDF is not 1. It might not '
            'be the format we supported.'
        )

        if config_name:
            # Late initialization
            init_default_bipolar_montage()
            if config_name not in DEFAULT_MONTAGE_DICT:
                raise ValueError(f'No corresponding montage found for {config_name}.')
            if use_common_montage:
                self.channels = get_common_montage_from_config(config_name)
            else:
                self.channels = DEFAULT_MONTAGE_DICT[config_name]
        elif channels:
            if all([isinstance(v, _Ch) for v in channels]):
                self.channels = channels
            else:
                try:
                    channels = asarray(channels)
                    if channels.ndim != 2 or channels.shape[-1] != 3:
                        raise ValueError('Incorrect format of channels')
                    self.channels = {i: _Ch(*v) for i, v in enumerate(channels)}
                except:
                    raise ValueError('Unknown format of channels.')
        else:
            # No given `config_name` and `channels`, use channels in raw data directly.
            # In this case, all channels will be taken as uniploar referece.
            self.channels = {idx: _Ch(name, name) for idx, name in enumerate(raw.ch_names)}

        raw_ch_names = asarray(raw.ch_names)
        anodes, cathodes = self.anodes, self.cathodes
        idx_anodes = np.nonzero(anodes[:, None] == raw_ch_names)[1]
        idx_cathodes = np.nonzero(cathodes[:, None] == raw_ch_names)[1]

        if idx_anodes.size != anodes.size:
            missing = np.setdiff1d(anodes, raw_ch_names)
            raise ValueError(f'Missing anodes in raw: {missing}')

        if not self.is_all_unipolar and idx_cathodes.size != cathodes.size:
            missing = np.setdiff1d(cathodes, raw_ch_names)
            raise ValueError(f'Missing cathodes in raw: {missing}')

        # Check whether units (magnitude) and orig_units (e.g. voltage) of
        # anodes and cathodes are the same
        if not self.is_all_unipolar:
            try:
                self._check_units(raw, idx_anodes, idx_cathodes)
            except ValueError:
                raise

        self._units = asarray(raw._raw_extras[0]['units'])[idx_anodes]
        self.idx_units = idx_anodes

    def __len__(self):
        return len(self.channels)

    @property
    def names(self):
        return np.array([v.name for v in self.channels.values()])

    @property
    def anodes(self):
        return np.array([v.anode for v in self.channels.values()])

    @property
    def cathodes(self):
        return np.array([v.cathode for v in self.channels.values()])

    @property
    def units(self):
        return self._units

    @property
    def is_all_unipolar(self):
        # If this montage is formed by unipolar channels, all cathodes should
        # be None. Therefore, `any(self.cathodes)` is False, and this property
        # returns True.
        return not any(self.cathodes)

    def _check_units(self, raw, idx_anodes, idx_cathodes):
        raw_ch_names = asarray(raw.ch_names)
        raw_units = asarray(raw._raw_extras[0]['units'])     # magnitude
        raw_o_units = asarray([raw._orig_units[ch] for ch in raw_ch_names])

        units_anodes = raw_units[idx_anodes]
        units_cathodes = raw_units[idx_cathodes]
        cmp_units = np.isclose(units_anodes, units_cathodes)
        if not all(cmp_units):
            mismatched = self.names[np.nonzero(~cmp_units)]
            xa, xc = units_anodes[~cmp_units], units_cathodes[~cmp_units]
            raise ValueError(f'Unit mismatched in channels: {mismatched}, '
                'In anodes: {xa}; In cathodes: {xc}')

        o_units_anodes = raw_o_units[idx_anodes]
        o_units_cathodes = raw_o_units[idx_cathodes]
        cmp_o_units = (o_units_anodes == o_units_cathodes)
        if not all(cmp_o_units):
            mismatched = self.names[np.nonzero(~cmp_o_units)]
            xa, xc = o_units_anodes[~cmp_o_units], o_units_cathodes[~cmp_o_units]
            raise ValueError(f'orig_units mismatched in channels: {mismatched}, '
                'In anodes: {xa}; In cathodes: {xc}')

    def channel_to_index(self, ch_names):
        """ Convert names of channels to indices in current montage. """
        return np.nonzero(np.asarray(ch_names)[:, None] == self.names)[1]

    def decompose(self, ch_names):
        """ Decompose given channels to their own composition.

        Parameters
        ----------
        ch_names : array_like of str
            Channels of current montage to be decomposed.

        Returns
        -------
        anodes_selected : ndarray of str
            Anodes of given channels.
        cathodes_selected : ndarray of str
            Cathodes of given channels. If channels are unipolar references,
            returned values will be None.
        """
        ch_names = asarray(ch_names)
        names, anodes, cathodes = self.names, self.anodes, self.cathodes
        idx_selected = np.nonzero(ch_names[:, None] == names)[1]
        if idx_selected.size != ch_names.size:
            missing = [v for v in ch_names if v not in names]
            raise ValueError('Some of given channels does not exist.')
        anodes_selcted = anodes[idx_selected]
        cathodes_selected = cathodes[idx_selected]
        return anodes_selcted, cathodes_selected

    def reorder_data_by_channels(self, data, raw_ch_names, ch_names):
        """
        Parameters
        ----------
        data : ndarray, shape: (channel, signal)
        raw_ch_names : ndarray, or list of str
            Channel names of given data.
        ch_names : ndarray, list of str
            Channel names to composite those components listed in `raw_ch_names`.
            (composition should be listed in `self.channels`)
        """
        data = asarray(data)

        if data.ndim != 2:
            raise ValueError('Dimension of given data should be 2 (channel, signal).')
        for v in ch_names:
            missing = [v for v in ch_names if v not in self.names]
            if missing:
                raise ValueError(f'Channels not found: {missing}')

        raw_ch_names = np.asarray(raw_ch_names)
        anodes, cathodes = self.decompose(ch_names)

        # Locate anodes in raw_ch_names
        idx_anodes = np.nonzero(anodes[:, None] == raw_ch_names)[1]
        assert all(raw_ch_names[idx_anodes] == anodes), (
            f'Anodes mismatched. Given composition: "{ch_names}"; '
            f'desired: "{anodes}", found: "{raw_ch_names[idx_anodes]}"'
        )

        if self.is_all_unipolar:
            return data[idx_anodes]

        # Locate cathodes in raw_ch_names
        idx_cathodes = np.nonzero(cathodes[:, None] == raw_ch_names)[1]
        assert all(raw_ch_names[idx_cathodes] == cathodes), (
            f'Cathodes mismatched. Given composition: "{ch_names}"; '
            f'desired: "{cathodes}", found: "{raw_ch_names[idx_cathodes]}"'
        )
        return data[idx_anodes] - data[idx_cathodes]
