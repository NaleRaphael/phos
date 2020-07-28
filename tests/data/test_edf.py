import pytest
import numpy as np
import numpy.testing as npt

from mne.io.edf.edf import RawEDF
from phos.data import EDF, Montage

from .fixtures import case_dict


class TestEDF:
    def test_load__montage(self):
        case = case_dict['normal']
        edf = EDF.from_file(case['fn_edf'], config_name=case['montage_config'])
        assert edf is not None
        assert edf.raw and isinstance(edf.raw, RawEDF)

    def test_load__default_channels(self):
        # In TUSZ dataset, channels are uniploar referece.
        case = case_dict['normal']
        edf = EDF.from_file(case['fn_edf'])
        assert set(edf.channels) == set(edf.raw.info.ch_names)

    def test_get_data__all_channels(self):
        case = case_dict['normal']
        edf = EDF.from_file(case['fn_edf'], config_name=case['montage_config'])

        t_start, t_stop = 0, 1
        chunk = edf.get_data(t_start, t_stop)
        assert chunk.shape == (len(edf.channels), int(edf.fs * (t_stop - t_start)))

    def test_get_data__specific_channels(self):
        case = case_dict['normal']
        edf = EDF.from_file(case['fn_edf'], config_name=case['montage_config'])

        t_start, t_stop = 0, 1
        selected_channels = edf.channels[:2]

        # Note that units of raw EEG channels are microvolt, but magnitude of them are
        # in 1e-6 already. So that we should specify the argument `unit` to 'volt' since
        # there is an unit coverter in our implementation of `EDF.get_data()`.
        chunk = edf.get_data(t_start, t_stop, channels=selected_channels, unit='volt')
        assert chunk.shape == (len(selected_channels), int(edf.fs * (t_stop - t_start)))

        anodes, cathodes = edf.montage.decompose(selected_channels)
        n_start, n_stop = int(t_start * edf.fs), int(t_stop * edf.fs)
        sig_anodes = edf.raw.get_data(picks=anodes, start=n_start, stop=n_stop)
        sig_cathodes = edf.raw.get_data(picks=cathodes, start=n_start, stop=n_stop)
        sig_diff = sig_anodes - sig_cathodes

        assert sig_diff.shape == chunk.shape
        npt.assert_allclose(sig_diff, chunk)

    def test_get_data__resample(self):
        case = case_dict['normal']
        edf = EDF.from_file(case['fn_edf'], config_name=case['montage_config'])

        t_start, t_stop = 0, 1
        selected_channels = edf.channels[:2]
        resample_fs = int(edf.fs / 2)
        chunk = edf.get_data(
            t_start, t_stop, channels=selected_channels, resample_fs=resample_fs
        )
        assert chunk.shape == (len(selected_channels), int(resample_fs * (t_stop - t_start)))

    def test_get_annotations__all_channels(self, mocker):
        case = case_dict['normal']
        edf = EDF.from_file(case['fn_edf'], config_name=case['montage_config'])

        def fake_label_file(*args, **kwargs):
            dtype = [
                ('ch', 'u1'), ('anno', 'u1'), ('ts', '<f2'), ('te', '<f2'), ('prob', '<f2')
            ]
            # In given 2-second signal, (anno, [t_start, t_stop], prob) are:
            # [(1, [0., 1.], 1.), (2, [1., 2.], 1.)] in all channels
            content = [[(ch, 1, 0., 1., 1.), (ch, 2, 1., 2., 1.)] for ch in range(edf.n_channels)]
            return np.array(content, dtype=dtype).reshape(-1)

        mocker.patch.object(np, 'load', side_effect=fake_label_file)

        t_start, t_stop, sample_length = 0, 2, 1
        anno = edf.get_annotations(t_start, t_stop, sample_length)
        assert anno.shape == (edf.n_channels, int((t_stop - t_start)/sample_length))
        # Returned annotations should be [1, 2] in all channels.
        npt.assert_equal(anno, np.tile([1, 2], edf.n_channels).reshape(edf.n_channels, -1))

        t_start, t_stop, sample_length = 0, 1, 1
        anno = edf.get_annotations(t_start, t_stop, sample_length)
        assert anno.shape == (edf.n_channels, int((t_stop - t_start)/sample_length))
        # Returned annotations should be [1] in all channels.
        npt.assert_equal(anno, np.tile([1], edf.n_channels).reshape(edf.n_channels, -1))

        t_start, t_stop, sample_length = 0.5, 1.5, 1
        anno = edf.get_annotations(t_start, t_stop, sample_length)
        assert anno.shape == (edf.n_channels, int((t_stop - t_start)/sample_length))
        # Returned annotations should be [1] in all channels.
        npt.assert_equal(anno, np.tile([1], edf.n_channels).reshape(edf.n_channels, -1))

    def test_get_annotations__specific_channels(self, mocker):
        case = case_dict['normal']
        edf = EDF.from_file(case['fn_edf'], config_name=case['montage_config'])

        def fake_label_file(*args, **kwargs):
            dtype = [
                ('ch', 'u1'), ('anno', 'u1'), ('ts', '<f2'), ('te', '<f2'), ('prob', '<f2')
            ]
            # In given 2-second signal, (anno, [t_start, t_stop], prob) are:
            # [(1, [0., 1.], 1.), (2, [1., 2.], 1.)] in all channels
            content = [[(ch, 1, 0., 1., 1.), (ch, 2, 1., 2., 1.)] for ch in range(edf.n_channels)]
            return np.array(content, dtype=dtype).reshape(-1)

        mocker.patch.object(np, 'load', side_effect=fake_label_file)

        t_start, t_stop, sample_length = 0, 2, 1
        selected_channels = edf.montage.names[:2]
        anno = edf.get_annotations(t_start, t_stop, sample_length, channels=selected_channels)
        assert anno.shape == (len(selected_channels), int((t_stop - t_start)/sample_length))
        # Returned annotations should be [1, 2] in all channels.
        npt.assert_equal(anno, np.tile([1, 2], len(selected_channels)).reshape(len(selected_channels), -1))

    def test_get_annotations__overlapping(self, mocker):
        case = case_dict['normal']
        edf = EDF.from_file(case['fn_edf'], config_name=case['montage_config'])

        def fake_label_file(*args, **kwargs):
            dtype = [
                ('ch', 'u1'), ('anno', 'u1'), ('ts', '<f2'), ('te', '<f2'), ('prob', '<f2')
            ]
            # In given 2-second signal, (anno, [t_start, t_stop], prob) are:
            # [(1, [0., 1.5], 1.), (2, [0.5, 2.], 1.)] in all channels
            # Note that there is an overlapping segment: [0.5, 1.5]
            content = [[(ch, 1, 0., 1.5, 1.), (ch, 2, 0.5, 2., 1.)] for ch in range(edf.n_channels)]
            return np.array(content, dtype=dtype).reshape(-1)

        mocker.patch.object(np, 'load', side_effect=fake_label_file)

        t_start, t_stop, sample_length = 0, 2, 1

        # Exception should be raised when `allow_annotation_overlapping` is False
        with pytest.raises(RuntimeError) as ex:
            anno = edf.get_annotations(t_start, t_stop, sample_length)

        anno = edf.get_annotations(t_start, t_stop, sample_length, allow_annotation_overlapping=True)
        assert anno.shape == (edf.n_channels, int((t_stop - t_start)/sample_length))
        # Returned annotations should be [1, 2] in all channels.
        npt.assert_equal(anno, np.tile([1, 2], edf.n_channels).reshape(edf.n_channels, -1))

        t_start, t_stop, sample_length = 0.5, 1.5, 1
        anno = edf.get_annotations(t_start, t_stop, sample_length, allow_annotation_overlapping=True)
        assert anno.shape == (edf.n_channels, int((t_stop - t_start)/sample_length))
        # Returned annotations should be [2] in all channels.
        npt.assert_equal(anno, np.tile([2], edf.n_channels).reshape(edf.n_channels, -1))

        t_start, t_stop, sample_length = 0.5, 1.5, 1
        anno = edf.get_annotations(
            t_start, t_stop, sample_length, allow_annotation_overlapping=True,
            select_overlapping='left'
        )
        # Returned annotations should be [2] in all channels.
        assert anno.shape == (edf.n_channels, int((t_stop - t_start)/sample_length))
        npt.assert_equal(anno, np.tile([1], edf.n_channels).reshape(edf.n_channels, -1))
