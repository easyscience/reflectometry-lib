# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Refnx calculator.
"""

import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from easyreflectometry.calculators.refl1d.calculator import Refl1d


class TestRefl1d(unittest.TestCase):
    def test_init(self):
        p = Refl1d()
        assert_equal(list(p._wrapper.storage.keys()), ['material', 'layer', 'item', 'model'])
        assert_equal(p._material_link['sld'], 'rho')
        assert_equal(p._material_link['isld'], 'irho')
        assert_equal(p._layer_link['thickness'], 'thickness')
        assert_equal(p._layer_link['roughness'], 'interface')
        assert_equal(p._item_link['repetitions'], 'repeat')
        assert_equal(p._model_link['scale'], 'scale')
        assert_equal(p._model_link['background'], 'bkg')
        assert_equal(p.name, 'refl1d')

    def test_reflectity_profile(self):
        p = Refl1d()
        p._wrapper.create_material('Material1')
        p._wrapper.update_material('Material1', rho=0.000, irho=0.000)
        p._wrapper.create_material('Material2')
        p._wrapper.update_material('Material2', rho=2.000, irho=0.000)
        p._wrapper.create_material('Material3')
        p._wrapper.update_material('Material3', rho=4.000, irho=0.000)
        p._wrapper.create_model('MyModel')
        p._wrapper.update_model('MyModel', bkg=1e-7)
        p._wrapper.create_layer('Layer1')
        p._wrapper.assign_material_to_layer('Material1', 'Layer1')
        p._wrapper.create_layer('Layer2')
        p._wrapper.assign_material_to_layer('Material2', 'Layer2')
        p._wrapper.update_layer('Layer2', thickness=10, interface=1.0)
        p._wrapper.create_layer('Layer3')
        p._wrapper.assign_material_to_layer('Material3', 'Layer3')
        p._wrapper.update_layer('Layer3', interface=1.0)
        p._wrapper.create_item('Item')
        p._wrapper.add_layer_to_item('Layer1', 'Item')
        p._wrapper.add_layer_to_item('Layer2', 'Item')
        p._wrapper.add_layer_to_item('Layer3', 'Item')
        p._wrapper.add_item('Item', 'MyModel')
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
        assert_almost_equal(p.reflectity_profile(q, 'MyModel'), expected, decimal=4)

    def test_calculate2(self):
        p = Refl1d()
        p._wrapper.create_material('Material1')
        p._wrapper.update_material('Material1', rho=0.000, irho=0.000)
        p._wrapper.create_material('Material2')
        p._wrapper.update_material('Material2', rho=2.000, irho=0.000)
        p._wrapper.create_material('Material3')
        p._wrapper.update_material('Material3', rho=4.000, irho=0.000)
        p._wrapper.create_model('MyModel')
        p._wrapper.update_model('MyModel', bkg=1e-7)
        p._wrapper.create_layer('Layer1')
        p._wrapper.assign_material_to_layer('Material1', 'Layer1')
        p._wrapper.create_layer('Layer2')
        p._wrapper.assign_material_to_layer('Material2', 'Layer2')
        p._wrapper.update_layer('Layer2', thickness=10, interface=1.0)
        p._wrapper.create_layer('Layer3')
        p._wrapper.assign_material_to_layer('Material3', 'Layer3')
        p._wrapper.update_layer('Layer3', interface=1.0)
        p._wrapper.create_item('Item1')
        p._wrapper.add_layer_to_item('Layer1', 'Item1')
        p._wrapper.create_item('Item2')
        p._wrapper.add_layer_to_item('Layer2', 'Item2')
        p._wrapper.add_layer_to_item('Layer1', 'Item2')
        p._wrapper.create_item('Item3')
        p._wrapper.add_layer_to_item('Layer3', 'Item3')
        p._wrapper.add_item('Item1', 'MyModel')
        p._wrapper.add_item('Item2', 'MyModel')
        p._wrapper.add_item('Item3', 'MyModel')
        p._wrapper.update_item('Item2', repeat=10)
        q = np.linspace(0.001, 0.3, 10)
        actual = p.reflectity_profile(q, 'MyModel')
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
        assert_almost_equal(actual, expected, decimal=4)

    def test_calculate_magnetic(self):
        p = Refl1d()
        p.include_magnetism = True
        p._wrapper.create_material('Material1')
        p._wrapper.update_material('Material1', rho=0.000, irho=0.000)
        p._wrapper.create_material('Material2')
        p._wrapper.update_material('Material2', rho=2.000, irho=0.000)
        p._wrapper.create_material('Material3')
        p._wrapper.update_material('Material3', rho=4.000, irho=0.000)
        p._wrapper.create_model('MyModel')
        p._wrapper.update_model('MyModel', bkg=1e-7)
        p._wrapper.create_layer('Layer1')
        p._wrapper.assign_material_to_layer('Material1', 'Layer1')
        p._wrapper.create_layer('Layer2')
        p._wrapper.assign_material_to_layer('Material2', 'Layer2')
        p._wrapper.update_layer('Layer2', thickness=10, interface=1.0)
        p._wrapper.create_layer('Layer3')
        p._wrapper.assign_material_to_layer('Material3', 'Layer3')
        p._wrapper.update_layer('Layer3', interface=1.0)
        p._wrapper.create_item('Item1')
        p._wrapper.add_layer_to_item('Layer1', 'Item1')
        p._wrapper.create_item('Item2')
        p._wrapper.add_layer_to_item('Layer2', 'Item2')
        p._wrapper.add_layer_to_item('Layer1', 'Item2')
        p._wrapper.create_item('Item3')
        p._wrapper.add_layer_to_item('Layer3', 'Item3')
        p._wrapper.add_item('Item1', 'MyModel')
        p._wrapper.add_item('Item2', 'MyModel')
        p._wrapper.add_item('Item3', 'MyModel')
        q = np.linspace(0.001, 0.3, 10)
        actual = p.reflectity_profile(q, 'MyModel')
        expected = [
            9.99491251e-01,
            1.08413641e-02,
            1.46824402e-04,
            2.11783999e-05,
            5.24616472e-06,
            1.61422945e-06,
            5.66961121e-07,
            2.34269519e-07,
            1.30026616e-07,
            1.05139655e-07,
        ]
        assert_almost_equal(actual, expected, decimal=4)

    def test_polarized_reflectivity_profiles(self):
        p = Refl1d()
        p.include_magnetism = True
        p._wrapper.create_material('Material1')
        p._wrapper.update_material('Material1', rho=0.000, irho=0.000)
        p._wrapper.create_material('Material2')
        p._wrapper.update_material('Material2', rho=4.000, irho=0.000)
        p._wrapper.create_material('Material3')
        p._wrapper.update_material('Material3', rho=2.047, irho=0.000)
        p._wrapper.create_model('MyModel')
        p._wrapper.create_layer('Layer1')
        p._wrapper.assign_material_to_layer('Material1', 'Layer1')
        p._wrapper.create_layer('Layer2')
        p._wrapper.assign_material_to_layer('Material2', 'Layer2')
        p._wrapper.update_layer('Layer2', thickness=100, interface=0)
        p._wrapper.update_layer('Layer2', magnetism_rhoM=2, magnetism_thetaM=45)
        p._wrapper.create_layer('Layer3')
        p._wrapper.assign_material_to_layer('Material3', 'Layer3')
        p._wrapper.create_item('Item')
        p._wrapper.add_layer_to_item('Layer1', 'Item')
        p._wrapper.add_layer_to_item('Layer2', 'Item')
        p._wrapper.add_layer_to_item('Layer3', 'Item')
        p._wrapper.add_item('Item', 'MyModel')
        q = np.linspace(0.005, 0.3, 50)

        channels = p.polarized_reflectivity_profiles(q, 'MyModel')

        assert_equal(list(channels.keys()), ['pp', 'pm', 'mp', 'mm'])
        for reflectivity in channels.values():
            assert_equal(len(reflectivity), len(q))

        # reflectity_profile follows the selected channel
        for key in ['pp', 'pm', 'mp', 'mm']:
            p.polarization_channel = key
            assert_equal(p.polarization_channel.value, key)
            assert_almost_equal(p.reflectity_profile(q, 'MyModel'), channels[key])

    def test_sld_profile(self):
        p = Refl1d()
        p._wrapper.create_material('Material1')
        p._wrapper.update_material('Material1', rho=0.000, irho=0.000)
        p._wrapper.create_material('Material2')
        p._wrapper.update_material('Material2', rho=2.000, irho=0.000)
        p._wrapper.create_material('Material3')
        p._wrapper.update_material('Material3', rho=4.000, irho=0.000)
        p._wrapper.create_model('MyModel')
        p._wrapper.update_model('MyModel', bkg=1e-7, dq=5.0)
        p._wrapper.create_layer('Layer1')
        p._wrapper.assign_material_to_layer('Material1', 'Layer1')
        p._wrapper.create_layer('Layer2')
        p._wrapper.assign_material_to_layer('Material2', 'Layer2')
        p._wrapper.update_layer('Layer2', thickness=10, interface=1.0)
        p._wrapper.create_layer('Layer3')
        p._wrapper.assign_material_to_layer('Material3', 'Layer3')
        p._wrapper.update_layer('Layer3', interface=1.0)
        p._wrapper.create_item('Item')
        p._wrapper.add_layer_to_item('Layer1', 'Item')
        p._wrapper.add_layer_to_item('Layer2', 'Item')
        p._wrapper.add_layer_to_item('Layer3', 'Item')
        p._wrapper.add_item('Item', 'MyModel')
        assert_almost_equal(p.sld_profile('MyModel')[1][0], 0)
        assert_almost_equal(p.sld_profile('MyModel')[1][-1], 4)
