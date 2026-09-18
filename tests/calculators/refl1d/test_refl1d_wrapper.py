# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Refl1d wrapper.
"""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from easyreflectometry.calculators.polarization import PolarizationChannel
from easyreflectometry.calculators.refl1d.wrapper import Refl1dWrapper
from easyreflectometry.calculators.refl1d.wrapper import _build_sample
from easyreflectometry.calculators.refl1d.wrapper import _get_oversampling_q
from easyreflectometry.calculators.refl1d.wrapper import _get_polarized_probe
from easyreflectometry.calculators.refl1d.wrapper import _get_probe


class TestRefl1d(unittest.TestCase):
    def test_init(self):
        p = Refl1dWrapper()
        assert_equal(list(p.storage.keys()), ['material', 'layer', 'item', 'model'])
        assert_equal(issubclass(p.storage['material'].__class__, dict), True)
        assert p._magnetism is False

    def test_set_magnetism(self):
        p = Refl1dWrapper()
        p.magnetism = True
        assert p._magnetism is True

    def test_reset_storage(self):
        p = Refl1dWrapper()
        p.storage['material']['a'] = 1
        assert_equal(p.storage['material']['a'], 1)
        p.reset_storage()
        assert_equal(p.storage['material'], {})

    def test_create_material(self):
        p = Refl1dWrapper()
        p.create_material('Si')
        assert_equal(list(p.storage['material'].keys()), ['Si'])
        assert_almost_equal(p.storage['material']['Si'].rho.value, 0.0)
        assert_almost_equal(p.storage['material']['Si'].irho.value, 0.0)
        assert_equal(p.storage['material']['Si'].name, 'Si')

    def test_update_material(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        assert_equal(list(p.storage['material'].keys()), ['B'])
        assert_almost_equal(p.storage['material']['B'].rho.value, 6.908)
        assert_almost_equal(p.storage['material']['B'].irho.value, -0.278)

    def test_get_material_value(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        assert_equal(list(p.storage['material'].keys()), ['B'])
        assert_almost_equal(p.get_material_value('B', 'rho'), 6.908)
        assert_almost_equal(p.get_material_value('B', 'irho'), -0.278)

    def test_create_layer(self):
        p = Refl1dWrapper()
        p.create_layer('Si')
        assert_equal(list(p.storage['layer'].keys()), ['Si'])
        assert_almost_equal(p.storage['layer']['Si'].thickness.value, 0)
        assert_almost_equal(p.storage['layer']['Si'].interface.value, 0)

    def test_update_layer(self):
        p = Refl1dWrapper()
        p.create_layer('Si')
        p.update_layer('Si', thickness=10, interface=5)
        assert_almost_equal(p.storage['layer']['Si'].thickness.value, 10)
        assert_almost_equal(p.storage['layer']['Si'].interface.value, 5)

    def test_update_magnetic_layer(self):
        p = Refl1dWrapper()
        p.magnetism = True
        p.create_layer('Si')
        p.update_layer('Si', magnetism_rhoM=5, magnetism_thetaM=10)
        assert_almost_equal(p.storage['layer']['Si'].magnetism.thetaM.value, 10)
        assert_almost_equal(p.storage['layer']['Si'].magnetism.rhoM.value, 5)

    def test_get_layer_value(self):
        p = Refl1dWrapper()
        p.create_layer('Si')
        p.update_layer('Si', thickness=10, interface=5)
        assert_almost_equal(p.get_layer_value('Si', 'thickness'), 10)
        assert_almost_equal(p.get_layer_value('Si', 'interface'), 5)

    def test_magnetic_get_layer_value(self):
        p = Refl1dWrapper()
        p.magnetism = True
        p.create_layer('Si')
        p.update_layer('Si', magnetism_rhoM=5, magnetism_thetaM=10)
        assert_almost_equal(p.get_layer_value('Si', 'magnetism_thetaM'), 10)
        assert_almost_equal(p.get_layer_value('Si', 'magnetism_rhoM'), 5)

    def test_create_item(self):
        p = Refl1dWrapper()
        p.create_item('SiNi')
        assert_equal(list(p.storage['item'].keys()), ['SiNi'])
        assert_almost_equal(p.storage['item']['SiNi'].repeat.value, 1)

    def test_update_item(self):
        p = Refl1dWrapper()
        p.create_item('SiNi')
        p.update_item('SiNi', repeat=10)
        assert_almost_equal(p.storage['item']['SiNi'].repeat.value, 10)

    def test_get_item_value(self):
        p = Refl1dWrapper()
        p.create_item('SiNi')
        p.update_item('SiNi', repeat=10)
        assert_almost_equal(p.get_item_value('SiNi', 'repeat'), 10)

    def test_create_model(self):
        p = Refl1dWrapper()
        p.create_model('MyModel')
        assert_equal(p.storage['model']['MyModel'], {'scale': 1, 'bkg': 0, 'items': []})

    def test_update_model(self):
        p = Refl1dWrapper()
        p.create_model('MyModel')
        p.update_model('MyModel', scale=2, bkg=1e-3)
        assert_almost_equal(p.storage['model']['MyModel']['scale'], 2)
        assert_almost_equal(p.storage['model']['MyModel']['bkg'], 1e-3)

    def test_get_model_value(self):
        p = Refl1dWrapper()
        p.create_model('MyModel')
        p.update_model('MyModel', scale=2, bkg=1e-3)
        assert_almost_equal(p.get_model_value('MyModel', 'scale'), 2)
        assert_almost_equal(p.get_model_value('MyModel', 'bkg'), 1e-3)

    def test_assign_material_to_layer(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        assert_almost_equal(p.storage['layer']['B_layer'].material.rho.value, 6.908)
        assert_almost_equal(p.storage['layer']['B_layer'].material.irho.value, -0.278)

    def test_add_layer_to_item(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        assert_equal(len(p.storage['item']['B_item'].stack), 0)
        p.add_layer_to_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item'].stack), 1)
        assert_equal(p.storage['item']['B_item'][0].name, 'B_layer')

    def test_add_item(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        p.create_model('MyModel')
        assert_equal(len(p.storage['model']['MyModel']['items']), 0)
        p.add_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel']['items']), 1)
        assert_equal(p.storage['model']['MyModel']['items'][0].name, 'B_item')

    def test_remove_layer_from_item(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item'].stack), 1)
        p.remove_layer_from_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item'].stack), 0)

    def test_remove_item(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        p.create_model('MyModel')
        p.add_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel']['items']), 1)
        p.remove_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel']['items']), 0)

    def test_calculate(self):
        p = Refl1dWrapper()
        p.create_material('Material1')
        p.update_material('Material1', rho=0.000, irho=0.000)
        p.create_material('Material2')
        p.update_material('Material2', rho=2.000, irho=0.000)
        p.create_material('Material3')
        p.update_material('Material3', rho=4.000, irho=0.000)
        p.create_model('MyModel')
        p.update_model('MyModel', bkg=1e-7)
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thickness=10, interface=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', interface=1.0)
        p.create_item('Item')
        p.add_layer_to_item('Layer1', 'Item')
        p.add_layer_to_item('Layer2', 'Item')
        p.add_layer_to_item('Layer3', 'Item')
        p.add_item('Item', 'MyModel')
        q = np.linspace(0.001, 0.3, 10)
        expected = [
            9.9949e-01,
            1.0842e-02,
            1.4709e-04,
            2.1277e-05,
            5.2902e-06,
            1.6347e-06,
            5.7605e-07,
            2.3775e-07,
            1.3093e-07,
            1.0520e-07,
        ]
        assert_almost_equal(p.calculate(q, 'MyModel'), expected, decimal=4)

    def test_calculate_three_items(self):
        p = Refl1dWrapper()
        p.create_material('Material1')
        p.update_material('Material1', rho=0.000, irho=0.000)
        p.create_material('Material2')
        p.update_material('Material2', rho=2.000, irho=0.000)
        p.create_material('Material3')
        p.update_material('Material3', rho=4.000, irho=0.000)
        p.create_model('MyModel')
        p.update_model('MyModel', bkg=1e-7)
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thickness=10, interface=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', interface=1.0)
        p.create_item('Item1')
        p.add_layer_to_item('Layer1', 'Item1')
        p.create_item('Item2')
        p.add_layer_to_item('Layer2', 'Item2')
        p.add_layer_to_item('Layer1', 'Item2')
        p.create_item('Item3')
        p.add_layer_to_item('Layer3', 'Item3')
        p.add_item('Item1', 'MyModel')
        p.add_item('Item2', 'MyModel')
        p.add_item('Item3', 'MyModel')
        p.update_item('Item2', repeat=10)
        q = np.linspace(0.001, 0.3, 10)
        expected = [
            9.9949e-01,
            8.7414e-03,
            1.1850e-04,
            5.4758e-06,
            6.3826e-06,
            1.0777e-06,
            1.0968e-06,
            4.5635e-07,
            3.4120e-07,
            2.7505e-07,
        ]
        assert_almost_equal(p.calculate(q, 'MyModel'), expected, decimal=4)

    def test_sld_profile(self):
        p = Refl1dWrapper()
        p.create_material('Material1')
        p.update_material('Material1', rho=0.000, irho=0.000)
        p.create_material('Material2')
        p.update_material('Material2', rho=2.000, irho=0.000)
        p.create_material('Material3')
        p.update_material('Material3', rho=4.000, irho=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thickness=10, interface=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', interface=1.0)
        p.create_item('Item')
        p.add_layer_to_item('Layer1', 'Item')
        p.add_layer_to_item('Layer2', 'Item')
        p.add_layer_to_item('Layer3', 'Item')
        p.add_item('Item', 'MyModel')
        assert_almost_equal(p.sld_profile('MyModel')[1][0], 0)
        assert_almost_equal(p.sld_profile('MyModel')[1][-1], 4)


def test_get_oversampling():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)

    # Then
    oversampling = _get_oversampling_q(q_array=q, dq_array=dq, oversampling_factor=5)

    # Expect
    assert len(oversampling) == 50
    assert oversampling[0] == 0.965
    assert oversampling[-1] == 10.35


def test_get_probe():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_probe(q_array=q, dq_array=dq, model_name=model_name, storage=storage)

    # Then
    assert all(probe.Q == q)
    assert all(probe.calc_Q == q)
    assert all(probe.dQ == dq)
    assert probe.intensity.value == 10
    assert probe.background.value == 20


def test_get_probe_oversampling():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_probe(q_array=q, dq_array=dq, model_name=model_name, storage=storage, oversampling_factor=2)

    # Then
    assert len(probe.calc_Q) == len(q)


def test_get_polarized_probe():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_polarized_probe(q_array=q, dq_array=dq, model_name=model_name, storage=storage)

    # Then
    assert all(probe.Q == q)
    assert all(probe.calc_Q == q)
    assert all(probe.dQ == dq)
    assert len(probe.calc_Q) == len(q)
    assert len(probe.xs) == 4
    for cross_section in probe.xs:
        assert cross_section is not None
        assert cross_section.intensity.value == 10
        assert cross_section.background.value == 20
        assert len(cross_section.calc_Q) == len(q)


def test_get_polarized_probe_oversampling():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_polarized_probe(q_array=q, dq_array=dq, model_name=model_name, storage=storage, oversampling_factor=2)

    # Then
    for cross_section in probe.xs:
        assert len(cross_section.calc_Qo) == 2 * len(q)


Q_POLARIZED = np.linspace(0.005, 0.3, 100)


def _sample_wrapper(rho: float, magnetic: bool, rhoM: float = 0.0, thetaM: float = 270.0) -> Refl1dWrapper:
    """Vacuum | 100 A layer of `rho` (optionally magnetic) | Si substrate.

    Magnetic values may be set via `update_layer` at any time (also one key at a
    time); they are stored per layer and attached to the slabs whenever magnetism
    is enabled.
    """
    p = Refl1dWrapper()
    if magnetic:
        p.magnetism = True
    p.create_material('Vacuum')
    p.update_material('Vacuum', rho=0.0, irho=0.0)
    p.create_material('MaterialMag')
    p.update_material('MaterialMag', rho=rho, irho=0.0)
    p.create_material('Si')
    p.update_material('Si', rho=2.047, irho=0.0)
    p.create_model('MyModel')
    p.create_layer('Superphase')
    p.assign_material_to_layer('Vacuum', 'Superphase')
    p.create_layer('LayerMag')
    p.assign_material_to_layer('MaterialMag', 'LayerMag')
    p.update_layer('LayerMag', thickness=100, interface=0)
    if magnetic:
        p.update_layer('LayerMag', magnetism_rhoM=rhoM, magnetism_thetaM=thetaM)
    p.create_layer('Subphase')
    p.assign_material_to_layer('Si', 'Subphase')
    p.create_item('Item')
    p.add_layer_to_item('Superphase', 'Item')
    p.add_layer_to_item('LayerMag', 'Item')
    p.add_layer_to_item('Subphase', 'Item')
    p.add_item('Item', 'MyModel')
    return p


def test_calculate_polarized_shape():
    p = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=45)

    channels = p.calculate_polarized(Q_POLARIZED, 'MyModel')

    assert list(channels.keys()) == ['pp', 'pm', 'mp', 'mm']
    for reflectivity in channels.values():
        assert isinstance(reflectivity, np.ndarray)
        assert len(reflectivity) == len(Q_POLARIZED)
        assert np.all(np.isfinite(reflectivity))


def test_calculate_follows_selected_channel():
    p = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=45)
    channels = p.calculate_polarized(Q_POLARIZED, 'MyModel')

    for channel in PolarizationChannel:
        p.polarization_channel = channel
        assert_allclose(p.calculate(Q_POLARIZED, 'MyModel'), channels[channel.value], rtol=1e-10)


def test_calculate_polarized_zero_magnetic_sld():
    # Polarized calculation with zero magnetic SLD (magnetism enabled, rhoM=0):
    # the non-spin-flip channels degenerate to the unpolarized result and the
    # spin-flip channels vanish.
    p = _sample_wrapper(rho=4.0, magnetic=True, rhoM=0.0, thetaM=270)
    unpolarized = _sample_wrapper(rho=4.0, magnetic=False)

    channels = p.calculate_polarized(Q_POLARIZED, 'MyModel')
    reference = unpolarized.calculate(Q_POLARIZED, 'MyModel')

    assert_allclose(channels['pp'], reference, rtol=1e-5)
    assert_allclose(channels['mm'], reference, rtol=1e-5)
    # Tolerance pinned from the observed numerics of refl1d 1.0.0 (machine noise).
    assert np.max(channels['pm']) < 1e-16
    assert np.max(channels['mp']) < 1e-16


def test_calculate_polarized_channel_ordering():
    # Pins the pp/mm halves of POLARIZATION_CHANNEL_TO_INDEX with physics, guarding
    # against a pp/mm swap: with the moment collinear with the neutron polarization
    # axis there is no spin flip and the non-spin-flip channels see rho +/- rhoM.
    # refl1d returns cross-sections in probe._xs_names order ['mm','mp','pm','pp']
    # (refl1d/probe/probe.py; magnetic_amplitude returns (--,-+,+-,++)). With the
    # default guide field (Aguide=270), thetaM=270 is the moment-parallel-to-field
    # orientation: spin-up ('pp') sees rho + rhoM, spin-down ('mm') sees rho - rhoM.
    # thetaM=90 (anti-parallel) swaps the two; both are spin-flip-free.
    rho, rhoM = 4.0, 2.0
    plus = _sample_wrapper(rho=rho + rhoM, magnetic=False)
    minus = _sample_wrapper(rho=rho - rhoM, magnetic=False)
    reflectivity_plus = plus.calculate(Q_POLARIZED, 'MyModel')
    reflectivity_minus = minus.calculate(Q_POLARIZED, 'MyModel')

    aligned = _sample_wrapper(rho=rho, magnetic=True, rhoM=rhoM, thetaM=270)
    channels = aligned.calculate_polarized(Q_POLARIZED, 'MyModel')
    assert_allclose(channels['pp'], reflectivity_plus, rtol=1e-4, atol=1e-9)
    assert_allclose(channels['mm'], reflectivity_minus, rtol=1e-4, atol=1e-9)
    assert np.max(channels['pm']) < 1e-16
    assert np.max(channels['mp']) < 1e-16

    anti_aligned = _sample_wrapper(rho=rho, magnetic=True, rhoM=rhoM, thetaM=90)
    channels = anti_aligned.calculate_polarized(Q_POLARIZED, 'MyModel')
    assert_allclose(channels['pp'], reflectivity_minus, rtol=1e-4, atol=1e-9)
    assert_allclose(channels['mm'], reflectivity_plus, rtol=1e-4, atol=1e-9)


def test_channel_index_map_matches_refl1d_xs_names():
    # The index map must agree with refl1d's own cross-section labels: the wrapper
    # extracts Experiment.reflectivity() results in probe.xs order, which is
    # PolarizedNeutronProbe._xs_names. This pins the convention at the source so a
    # refl1d-side reordering (or a wrapper-side swap) fails loudly.
    from refl1d import names as refl1d_names

    from easyreflectometry.calculators.polarization import POLARIZATION_CHANNEL_TO_INDEX

    xs_names = refl1d_names.PolarizedNeutronQProbe._xs_names
    assert len(xs_names) == 4
    for channel, index in POLARIZATION_CHANNEL_TO_INDEX.items():
        assert xs_names[index] == channel.value


def test_calculate_polarized_spin_flip():
    # Moment perpendicular to the neutron polarization (thetaM=0) produces spin flip;
    # a collinear moment (thetaM=90, see test_calculate_polarized_channel_ordering)
    # produces essentially none.
    aligned = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=90)
    perpendicular = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=0)

    channels_aligned = aligned.calculate_polarized(Q_POLARIZED, 'MyModel')
    channels_perpendicular = perpendicular.calculate_polarized(Q_POLARIZED, 'MyModel')

    assert np.max(channels_perpendicular['pm']) > 1e3 * np.max(channels_aligned['pm'])
    assert np.max(channels_perpendicular['pm']) > 1e-6  # absolute sanity floor
    # pm and mp are identical by symmetry for a non-chiral, non-absorptive sample,
    # so this cannot distinguish them: the pm=1 / mp=2 indices rest on the refl1d
    # docstring alone ("a sequence pp, pm, mp and mm").
    assert_allclose(channels_perpendicular['pm'], channels_perpendicular['mp'], rtol=1e-10)


def test_calculate_polarized_scale_and_background():
    # Intensity and background must reach every cross-section:
    # R_out = scale * R + bkg, channel by channel.
    scale, bkg = 2.0, 1e-6
    plain = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=45)
    scaled = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=45)
    scaled.update_model('MyModel', scale=scale, bkg=bkg)

    channels_plain = plain.calculate_polarized(Q_POLARIZED, 'MyModel')
    channels_scaled = scaled.calculate_polarized(Q_POLARIZED, 'MyModel')

    for key in ['pp', 'pm', 'mp', 'mm']:
        assert_allclose(channels_scaled[key], scale * channels_plain[key] + bkg, rtol=1e-8)


def test_calculate_polarized_requires_magnetism():
    p = _sample_wrapper(rho=4.0, magnetic=False)
    with pytest.raises(ValueError):
        p.calculate_polarized(Q_POLARIZED, 'MyModel')


def test_polarization_channel_normalization():
    p = Refl1dWrapper()
    p.magnetism = True

    p.polarization_channel = PolarizationChannel.MM
    assert p.polarization_channel is PolarizationChannel.MM
    p.polarization_channel = 'pm'
    assert p.polarization_channel is PolarizationChannel.PM

    for bad in ['MM', 'xx', None]:
        with pytest.raises(ValueError):
            p.polarization_channel = bad


def test_polarization_channel_requires_magnetism():
    p = Refl1dWrapper()
    with pytest.raises(ValueError):
        p.polarization_channel = 'mm'
    # pp is always allowed
    p.polarization_channel = 'pp'
    assert p.polarization_channel is PolarizationChannel.PP


def test_disabling_magnetism_resets_channel():
    p = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=45)
    unpolarized = _sample_wrapper(rho=4.0, magnetic=False)
    p.polarization_channel = 'mm'

    p.magnetism = False

    # The transition is complete: channel back to pp, slab Magnetism objects
    # stripped, and the plain (unpolarized) calculation path works.
    assert p.polarization_channel is PolarizationChannel.PP
    assert all(layer.magnetism is None for layer in p.storage['layer'].values())
    assert_allclose(p.calculate(Q_POLARIZED, 'MyModel'), unpolarized.calculate(Q_POLARIZED, 'MyModel'), rtol=1e-10)

    # Re-enabling restores the stored magnetic values (rhoM/thetaM survive the
    # toggle so the wrapper stays in sync with model parameters that still hold them).
    p.magnetism = True
    assert p.polarization_channel is PolarizationChannel.PP
    restored = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=45)
    channels = p.calculate_polarized(Q_POLARIZED, 'MyModel')
    reference = restored.calculate_polarized(Q_POLARIZED, 'MyModel')
    for channel in ('pp', 'pm', 'mp', 'mm'):
        assert_allclose(channels[channel], reference[channel], rtol=1e-10)


def test_polarized_reflectivities_guards_malformed_output():
    p = _sample_wrapper(rho=4.0, magnetic=True, rhoM=2.0, thetaM=45)
    q = Q_POLARIZED

    # Fewer than four cross-sections
    with patch('easyreflectometry.calculators.refl1d.wrapper.names.Experiment') as mock_experiment:
        mock_experiment.return_value.reflectivity.return_value = [(q, np.ones(len(q)))] * 3
        with pytest.raises(RuntimeError, match='expected 4'):
            p.calculate_polarized(q, 'MyModel')

    # Wrong-length cross-section: index 1 is 'mp' in refl1d's (mm, mp, pm, pp) order.
    with patch('easyreflectometry.calculators.refl1d.wrapper.names.Experiment') as mock_experiment:
        mock_experiment.return_value.reflectivity.return_value = [
            (q, np.ones(len(q))),
            (q, np.ones(len(q) - 1)),
            (q, np.ones(len(q))),
            (q, np.ones(len(q))),
        ]
        with pytest.raises(RuntimeError, match='malformed mp'):
            p.calculate_polarized(q, 'MyModel')

    # Non-finite values: index 3 is 'pp' in refl1d's (mm, mp, pm, pp) order.
    with patch('easyreflectometry.calculators.refl1d.wrapper.names.Experiment') as mock_experiment:
        bad = np.ones(len(q))
        bad[0] = np.nan
        mock_experiment.return_value.reflectivity.return_value = [(q, np.ones(len(q)))] * 3 + [(q, bad)]
        with pytest.raises(RuntimeError, match='malformed pp'):
            p.calculate_polarized(q, 'MyModel')


def test_polarization_channel_survives_reset_storage():
    # reset_storage leaves _magnetism and the resolution function alone;
    # the selected channel behaves consistently.
    p = Refl1dWrapper()
    p.magnetism = True
    p.polarization_channel = 'mm'
    p.reset_storage()
    assert p.polarization_channel is PolarizationChannel.MM
    assert p._magnetism is True


@patch('easyreflectometry.calculators.refl1d.wrapper.names.Stack')
@patch('easyreflectometry.calculators.refl1d.wrapper.Repeat')
def test_build_sample(mock_repeat, mock_stack):
    # When
    mock_item_1 = MagicMock()
    mock_item_1.repeat = MagicMock()
    mock_item_1.repeat.value = 1
    mock_item_1.stack = ['1a', '1b']
    mock_item_2 = MagicMock()
    mock_item_2.repeat = MagicMock()
    mock_item_2.repeat.value = 2
    mock_item_2.stack = ['2a', '2b']
    model_name = 'model_name'
    mock_stack.__or__ = MagicMock()

    storage = {'model': {model_name: {'items': []}}}
    storage['model'][model_name]['items'].append(mock_item_1)
    storage['model'][model_name]['items'].append(mock_item_2)

    # Then
    _ = _build_sample(model_name=model_name, storage=storage)

    # Expect
    assert mock_stack.call_count == 2
    assert mock_repeat.call_count == 1
    # TODO do asserts on sample returned by _build_sample
    # will probably use other build_sample function in future
    # difficult to test current implementation
