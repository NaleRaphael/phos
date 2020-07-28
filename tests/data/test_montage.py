import pytest
import numpy.testing as npt
from mne.io import read_raw_edf
from phos.data import Montage

from .fixtures import case_dict


class TestMontage:
    def test_create__default_montage(self):
        # In TUSZ dataset, channels are uniploar referece.
        case = case_dict['normal']
        raw_edf = read_raw_edf(case['fn_edf'])
        montage = Montage(raw_edf)
        assert set(montage.names) == set(raw_edf.ch_names)
        assert set(montage.anodes) == set(raw_edf.ch_names)
        assert all(montage.cathodes == None)

    def test_create__custom_montage(self):
        case = case_dict['normal']
        raw_edf = read_raw_edf(case['fn_edf'])
        channels = [
            ('FP1-F7', 'EEG FP1-REF', 'EEG F7-REF'),
            ('F7-T3', 'EEG F7-REF', 'EEG T3-REF')
        ]
        montage = Montage(raw_edf, channels=channels)
        assert all(montage.names == [v[0] for v in channels])
        assert all(montage.anodes == [v[1] for v in channels])
        assert all(montage.cathodes == [v[2] for v in channels])

    def test_decompose(self):
        case = case_dict['normal']
        raw_edf = read_raw_edf(case['fn_edf'])
        channels = [
            ('FP1-F7', 'EEG FP1-REF', 'EEG F7-REF'),
            ('F7-T3', 'EEG F7-REF', 'EEG T3-REF')
        ]

        montage = Montage(raw_edf, channels=channels)
        anodes, cathodes = montage.decompose(['FP1-F7'])
        npt.assert_equal(anodes, ['EEG FP1-REF'])
        npt.assert_equal(cathodes, ['EEG F7-REF'])

        # Note that order of given channels is reversed
        anodes, cathodes = montage.decompose(['F7-T3', 'FP1-F7'])
        npt.assert_equal(anodes, ['EEG F7-REF', 'EEG FP1-REF'])
        npt.assert_equal(cathodes, ['EEG T3-REF', 'EEG F7-REF'])
