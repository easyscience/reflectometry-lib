# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for PolarizedDataSet and spin-channel detection.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.data import detect_polarization_channel
from easyreflectometry.data.polarized import _channel_from_filename


def _dataset(name: str = 'series') -> DataSet1D:
    return DataSet1D(name=name, x=np.array([0.01, 0.02]), y=np.array([1.0, 0.5]))


class TestPolarizedDataSet:
    def test_requires_at_least_one_channel(self):
        with pytest.raises(ValueError):
            PolarizedDataSet(channels={})

    def test_rejects_invalid_channel_key(self):
        with pytest.raises(ValueError):
            PolarizedDataSet(channels={'xx': _dataset()})

    def test_rejects_non_dataset_values(self):
        with pytest.raises(ValueError):
            PolarizedDataSet(channels={'pp': np.array([1.0])})

    def test_channels_in_canonical_order(self):
        p = PolarizedDataSet(channels={'mm': _dataset('a'), 'pm': _dataset('b'), 'pp': _dataset('c')})
        assert p.available_channels == [
            PolarizationChannel.PP,
            PolarizationChannel.PM,
            PolarizationChannel.MM,
        ]

    def test_getitem_and_contains_accept_strings_and_enums(self):
        pp = _dataset('up-up')
        p = PolarizedDataSet(channels={'pp': pp, 'mm': _dataset('down-down')})
        assert p['pp'] is pp
        assert p[PolarizationChannel.PP] is pp
        assert 'pp' in p
        assert PolarizationChannel.MM in p
        assert 'pm' not in p
        assert 'xx' not in p
        assert len(p) == 2

    def test_model_propagates_to_channel_datasets(self):
        pp = _dataset()
        mm = _dataset()
        p = PolarizedDataSet(channels={'pp': pp, 'mm': mm})
        assert p.is_simulation
        marker = object()
        p.model = marker
        assert p.is_experiment
        assert pp.model is marker
        assert mm.model is marker

    def test_channels_view_is_read_only(self):
        p = PolarizedDataSet(channels={'pp': _dataset()})
        with pytest.raises(TypeError):
            p.channels[PolarizationChannel.MM] = _dataset()
        with pytest.raises(TypeError):
            del p.channels[PolarizationChannel.PP]

    def test_set_channel_validates_propagates_and_reorders(self):
        p = PolarizedDataSet(channels={'mm': _dataset()})
        marker = object()
        p.model = marker

        with pytest.raises(ValueError):
            p.set_channel('xx', _dataset())
        with pytest.raises(ValueError):
            p.set_channel('pp', np.array([1.0]))

        pp = _dataset('up-up')
        p.set_channel('pp', pp)
        # Canonical order restored, model propagated to the new dataset.
        assert p.available_channels == [PolarizationChannel.PP, PolarizationChannel.MM]
        assert p['pp'] is pp
        assert pp.model is marker

        replacement = _dataset('up-up-2')
        p.set_channel(PolarizationChannel.PP, replacement)
        assert p['pp'] is replacement

    def test_remove_channel_guards(self):
        p = PolarizedDataSet(channels={'pp': _dataset(), 'mm': _dataset()})
        with pytest.raises(ValueError):
            p.remove_channel('pm')  # not present
        p.remove_channel('pp')
        assert p.available_channels == [PolarizationChannel.MM]
        with pytest.raises(ValueError):
            p.remove_channel('mm')  # last channel cannot be removed
        assert p.available_channels == [PolarizationChannel.MM]


class TestChannelDetection:
    @pytest.mark.parametrize(
        'filename,expected',
        [
            ('sample_uu.dat', PolarizationChannel.PP),
            ('sample_pp.ort', PolarizationChannel.PP),
            ('sample-up-up.txt', PolarizationChannel.PP),
            ('sample_up.dat', PolarizationChannel.PP),
            ('run12_plus.txt', PolarizationChannel.PP),
            ('sample_dd.dat', PolarizationChannel.MM),
            ('sample_down.dat', PolarizationChannel.MM),
            ('sample-down-down.txt', PolarizationChannel.MM),
            ('sample_ud.dat', PolarizationChannel.PM),
            ('sample_up_down.dat', PolarizationChannel.PM),
            ('sample_pm.ort', PolarizationChannel.PM),
            ('sample_du.dat', PolarizationChannel.MP),
            ('sample_down_up.dat', PolarizationChannel.MP),
            ('sample_mp.ort', PolarizationChannel.MP),
            ('nothing_here.dat', None),
            ('d2o_layer.dat', None),
        ],
    )
    def test_filename_heuristics(self, filename, expected):
        assert _channel_from_filename(filename) == expected

    @pytest.mark.parametrize(
        'polarization,expected',
        [
            ('pp', PolarizationChannel.PP),
            ('mm', PolarizationChannel.MM),
            ('pm', PolarizationChannel.PM),
            ('mp', PolarizationChannel.MP),
            # Partially-analysed observables are not spin channels: 'po' measures
            # pp + pm (incident plus, no outgoing analysis), 'mo' measures mp + mm.
            ('po', None),
            ('mo', None),
            ('op', None),
            ('om', None),
            ('unpolarized', None),
        ],
    )
    def test_orso_header_detection(self, polarization, expected):
        orso_dataset = SimpleNamespace(
            info=SimpleNamespace(
                data_source=SimpleNamespace(
                    measurement=SimpleNamespace(
                        instrument_settings=SimpleNamespace(polarization=polarization),
                    )
                )
            )
        )
        with patch('orsopy.fileio.orso.load_orso', return_value=[orso_dataset]):
            assert detect_polarization_channel('whatever.ort') == expected

    def test_header_takes_precedence_over_filename(self):
        orso_dataset = SimpleNamespace(
            info=SimpleNamespace(
                data_source=SimpleNamespace(
                    measurement=SimpleNamespace(
                        instrument_settings=SimpleNamespace(polarization='mm'),
                    )
                )
            )
        )
        with patch('orsopy.fileio.orso.load_orso', return_value=[orso_dataset]):
            assert detect_polarization_channel('sample_uu.ort') == PolarizationChannel.MM

    def test_unreadable_file_falls_back_to_filename(self):
        # No such file: the ORSO branch raises internally and the name decides.
        assert detect_polarization_channel('no_such_file_dd.ort') == PolarizationChannel.MM

    def test_unpolarized_header_suppresses_filename_fallback(self):
        # A header that explicitly declares a non-channel polarization wins over
        # channel-looking filename tokens.
        orso_dataset = SimpleNamespace(
            info=SimpleNamespace(
                data_source=SimpleNamespace(
                    measurement=SimpleNamespace(
                        instrument_settings=SimpleNamespace(polarization='unpolarized'),
                    )
                )
            )
        )
        with patch('orsopy.fileio.orso.load_orso', return_value=[orso_dataset]):
            assert detect_polarization_channel('sample_uu.ort') is None
