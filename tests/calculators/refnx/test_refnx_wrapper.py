# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Refnx wrapper.
"""

import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from refnx import reflect

from easyreflectometry.calculators.refnx.wrapper import RefnxWrapper
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import PercentageFwhm


class TestRefnx(unittest.TestCase):
    def test_init(self):
        p = RefnxWrapper()
        assert_equal(list(p.storage.keys()), ['material', 'layer', 'item', 'model'])
        assert_equal(issubclass(p.storage['material'].__class__, dict), True)

    def test_set_magnetism(self):
        p = RefnxWrapper()
        with pytest.raises(NotImplementedError):
            p.magnetism = True
        assert p._magnetism is False

    def test_calculate_polarized_not_supported(self):
        p = RefnxWrapper()
        with pytest.raises(NotImplementedError):
            p.calculate_polarized(np.linspace(0.01, 0.3, 10), 'MyModel')

    def test_polarization_channel_requires_magnetism(self):
        p = RefnxWrapper()
        with pytest.raises(ValueError):
            p.polarization_channel = 'mm'

    def test_reset_storage(self):
        p = RefnxWrapper()
        p.storage['material']['a'] = 1
        assert_equal(p.storage['material']['a'], 1)
        p.reset_storage()
        assert_equal(p.storage['material'], {})

    def test_create_material(self):
        p = RefnxWrapper()
        p.create_material('Si')
        assert_equal(list(p.storage['material'].keys()), ['Si'])
        assert_almost_equal(p.storage['material']['Si'].real.value, 0.0)
        assert_almost_equal(p.storage['material']['Si'].imag.value, 0.0)
        assert_equal(p.storage['material']['Si'].name, 'Si')

    def test_update_material(self):
        p = RefnxWrapper()
        p.create_material('B')
        p.update_material('B', real=6.908, imag=-0.278)
        assert_equal(list(p.storage['material'].keys()), ['B'])
        assert_almost_equal(p.storage['material']['B'].real.value, 6.908)
        assert_almost_equal(p.storage['material']['B'].imag.value, -0.278)

    def test_get_material_value(self):
        p = RefnxWrapper()
        p.create_material('B')
        p.update_material('B', real=6.908, imag=-0.278)
        assert_equal(list(p.storage['material'].keys()), ['B'])
        assert_almost_equal(p.get_material_value('B', 'real'), 6.908)
        assert_almost_equal(p.get_material_value('B', 'imag'), -0.278)

    def test_create_layer(self):
        p = RefnxWrapper()
        p.create_layer('Si')
        assert_equal(list(p.storage['layer'].keys()), ['Si'])
        assert_almost_equal(p.storage['layer']['Si'].thick.value, 0)
        assert_almost_equal(p.storage['layer']['Si'].rough.value, 0)
        assert_almost_equal(p.storage['layer']['Si'].sld.real.value, 0)
        assert_almost_equal(p.storage['layer']['Si'].sld.imag.value, 0)

    def test_update_layer(self):
        p = RefnxWrapper()
        p.create_layer('Si')
        p.update_layer('Si', thick=10, rough=5)
        assert_almost_equal(p.storage['layer']['Si'].thick.value, 10)
        assert_almost_equal(p.storage['layer']['Si'].rough.value, 5)

    def test_get_layer_value(self):
        p = RefnxWrapper()
        p.create_layer('Si')
        p.update_layer('Si', thick=10, rough=5)
        assert_almost_equal(p.get_layer_value('Si', 'thick'), 10)
        assert_almost_equal(p.get_layer_value('Si', 'rough'), 5)

    def test_create_item(self):
        p = RefnxWrapper()
        p.create_item('SiNi')
        assert_equal(list(p.storage['item'].keys()), ['SiNi'])
        assert_almost_equal(p.storage['item']['SiNi'].repeats.value, 1)

    def test_update_item(self):
        p = RefnxWrapper()
        p.create_item('SiNi')
        p.update_item('SiNi', repeats=10)
        assert_almost_equal(p.storage['item']['SiNi'].repeats.value, 10)

    def test_get_item_value(self):
        p = RefnxWrapper()
        p.create_item('SiNi')
        p.update_item('SiNi', repeats=10)
        assert_almost_equal(p.get_item_value('SiNi', 'repeats'), 10)

    def test_create_model(self):
        p = RefnxWrapper()
        p.create_model('MyModel')
        assert_equal(isinstance(p.storage['model']['MyModel'], reflect.ReflectModel), True)

    def test_update_model(self):
        p = RefnxWrapper()
        p.create_model('MyModel')
        p.update_model('MyModel', scale=2, bkg=1e-3, dq=2.0)
        assert_almost_equal(p.storage['model']['MyModel'].scale.value, 2)
        assert_almost_equal(p.storage['model']['MyModel'].bkg.value, 1e-3)
        assert_almost_equal(p.storage['model']['MyModel'].dq.value, 2.0)

    def test_get_model_value(self):
        p = RefnxWrapper()
        p.create_model('MyModel')
        p.update_model('MyModel', scale=2, bkg=1e-3, dq=2.0)
        assert_almost_equal(p.get_model_value('MyModel', 'scale'), 2)
        assert_almost_equal(p.get_model_value('MyModel', 'bkg'), 1e-3)
        assert_almost_equal(p.get_model_value('MyModel', 'dq'), 2.0)

    def test_assign_material_to_layer(self):
        p = RefnxWrapper()
        p.create_material('B')
        p.update_material('B', real=6.908, imag=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        assert_almost_equal(p.storage['layer']['B_layer'].sld.real.value, 6.908)
        assert_almost_equal(p.storage['layer']['B_layer'].sld.imag.value, -0.278)

    def test_add_layer_to_item(self):
        p = RefnxWrapper()
        p.create_material('B')
        p.update_material('B', real=6.908, imag=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        assert_equal(len(p.storage['item']['B_item']), 0)
        p.add_layer_to_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item']), 1)
        assert_equal(p.storage['item']['B_item'][0].name, 'B_layer')

    def test_add_item(self):
        p = RefnxWrapper()
        p.create_material('B')
        p.update_material('B', real=6.908, imag=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        p.create_model('MyModel')
        assert_equal(len(p.storage['model']['MyModel'].structure.components), 0)
        p.add_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel'].structure.components), 1)
        assert_equal(p.storage['model']['MyModel'].structure.components[0].name, 'B_item')

    def test_remove_layer_from_item(self):
        p = RefnxWrapper()
        p.create_material('B')
        p.update_material('B', real=6.908, imag=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item']), 1)
        p.remove_layer_from_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item']), 0)

    def test_remove_item(self):
        p = RefnxWrapper()
        p.create_material('B')
        p.update_material('B', real=6.908, imag=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        p.create_model('MyModel')
        p.add_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel'].structure.components), 1)
        p.remove_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel'].structure.components), 0)

    def test_calculate(self):
        p = RefnxWrapper()
        p.create_material('Material1')
        p.update_material('Material1', real=0.000, imag=0.000)
        p.create_material('Material2')
        p.update_material('Material2', real=2.000, imag=0.000)
        p.create_material('Material3')
        p.update_material('Material3', real=4.000, imag=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thick=10, rough=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', rough=1.0)
        p.create_item('Item')
        p.add_layer_to_item('Layer1', 'Item')
        p.add_layer_to_item('Layer2', 'Item')
        p.add_layer_to_item('Layer3', 'Item')
        p.add_item('Item', 'MyModel')
        q = np.linspace(0.001, 0.3, 10)
        expected = [
            9.99956517e-01,
            2.16286891e-03,
            1.14086254e-04,
            1.93031759e-05,
            4.94188894e-06,
            1.54191953e-06,
            5.45592112e-07,
            2.26619392e-07,
            1.26726993e-07,
            1.01842852e-07,
        ]
        assert_almost_equal(p.calculate(q, 'MyModel'), expected)

    def test_calculate_three_items(self):
        p = RefnxWrapper()
        p.create_material('Material1')
        p.update_material('Material1', real=0.000, imag=0.000)
        p.create_material('Material2')
        p.update_material('Material2', real=2.000, imag=0.000)
        p.create_material('Material3')
        p.update_material('Material3', real=4.000, imag=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thick=10, rough=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', rough=1.0)
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
        p.update_item('Item2', repeats=10)
        q = np.linspace(0.001, 0.3, 10)
        expected = [
            9.9995652e-01,
            1.7096697e-05,
            1.2253047e-04,
            2.4026928e-06,
            6.7117546e-06,
            8.3209877e-07,
            1.1512901e-06,
            4.1468151e-07,
            3.4981523e-07,
            2.5424356e-07,
        ]
        assert_almost_equal(p.calculate(q, 'MyModel'), expected)

    def test_sld_profile(self):
        p = RefnxWrapper()
        p.create_material('Material1')
        p.update_material('Material1', real=0.000, imag=0.000)
        p.create_material('Material2')
        p.update_material('Material2', real=2.000, imag=0.000)
        p.create_material('Material3')
        p.update_material('Material3', real=4.000, imag=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thick=10, rough=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', rough=1.0)
        p.create_item('Item')
        p.add_layer_to_item('Layer1', 'Item')
        p.add_layer_to_item('Layer2', 'Item')
        p.add_layer_to_item('Layer3', 'Item')
        p.add_item('Item', 'MyModel')
        assert_almost_equal(p.sld_profile('MyModel')[1][0], 0)
        assert_almost_equal(p.sld_profile('MyModel')[1][-1], 4)

    # Tests from https://github.com/reflectivity/analysis/tree/master/validation/test/unpolarised
    def test_calculate_github_test0(self):
        p = RefnxWrapper()
        p.create_material('Material1')
        p.update_material('Material1', real=2.070, imag=0.000)
        p.create_material('Material2')
        p.update_material('Material2', real=3.450, imag=0.100)
        p.create_material('Material3')
        p.update_material('Material3', real=5.000, imag=0.01)
        p.create_material('Material4')
        p.update_material('Material4', real=6.000, imag=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thick=100, rough=3.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', thick=200, rough=1.0)
        p.create_layer('Layer4')
        p.assign_material_to_layer('Material4', 'Layer4')
        p.update_layer('Layer4', rough=5.0)
        p.create_item('Item1')
        p.add_layer_to_item('Layer1', 'Item1')
        p.create_item('Item2')
        p.add_layer_to_item('Layer2', 'Item2')
        p.create_item('Item3')
        p.add_layer_to_item('Layer3', 'Item3')
        p.create_item('Item4')
        p.add_layer_to_item('Layer4', 'Item4')
        p.add_item('Item1', 'MyModel')
        p.add_item('Item2', 'MyModel')
        p.add_item('Item3', 'MyModel')
        p.add_item('Item4', 'MyModel')
        p.set_resolution_function(PercentageFwhm(0))
        p.update_model('MyModel', bkg=0)
        q = np.array([
            5.000000000000000104e-03,
            3.717499999999999971e-02,
            5.449999999999999983e-02,
            1.005349999999999994e-01,
            2.955650000000000222e-01,
        ])
        expected = [
            9.665000503913141472e-01,
            3.486325360684768590e-04,
            8.540420179439664689e-05,
            5.815959818366009312e-06,
            4.999742968030015832e-08,
        ]
        assert_almost_equal(p.calculate(q, 'MyModel'), expected)

    def test_calculate_github_test2(self):
        p = RefnxWrapper()
        p.create_material('Material1')
        p.update_material('Material1', real=0.000, imag=0.000)
        p.create_material('Material2')
        p.update_material('Material2', real=6.360, imag=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', rough=3.0)
        p.create_item('Item1')
        p.add_layer_to_item('Layer1', 'Item1')
        p.create_item('Item2')
        p.add_layer_to_item('Layer2', 'Item2')
        p.add_item('Item1', 'MyModel')
        p.add_item('Item2', 'MyModel')
        p.set_resolution_function(PercentageFwhm(0))
        p.update_model('MyModel', bkg=0)
        q = np.array([
            5.000000000000000104e-03,
            7.564500000000000390e-02,
            1.433050000000000157e-01,
            2.368350000000000177e-01,
            5.920499999999999652e-01,
        ])
        expected = [
            1.000000000000000222e00,
            1.964576414578978456e-04,
            1.280698699505669096e-05,
            1.234290141526865827e-06,
            2.222536631965092181e-09,
        ]
        assert_almost_equal(p.calculate(q, 'MyModel'), expected)

    def test_calculate_github_test4_constant_resolution(self):
        p = RefnxWrapper()
        p.create_material('Material1')
        p.update_material('Material1', real=2.070, imag=0.000)
        p.create_material('Material2')
        p.update_material('Material2', real=3.450, imag=0.100)
        p.create_material('Material3')
        p.update_material('Material3', real=5.000, imag=0.01)
        p.create_material('Material4')
        p.update_material('Material4', real=6.000, imag=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thick=100, rough=3.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', thick=200, rough=1.0)
        p.create_layer('Layer4')
        p.assign_material_to_layer('Material4', 'Layer4')
        p.update_layer('Layer4', rough=5.0)
        p.create_item('Item1')
        p.add_layer_to_item('Layer1', 'Item1')
        p.create_item('Item2')
        p.add_layer_to_item('Layer2', 'Item2')
        p.create_item('Item3')
        p.add_layer_to_item('Layer3', 'Item3')
        p.create_item('Item4')
        p.add_layer_to_item('Layer4', 'Item4')
        p.add_item('Item1', 'MyModel')
        p.add_item('Item2', 'MyModel')
        p.add_item('Item3', 'MyModel')
        p.add_item('Item4', 'MyModel')
        p.set_resolution_function(PercentageFwhm(5))
        p.update_model('MyModel', bkg=0)
        assert_allclose(p.calculate(test4_dat[:, 0], 'MyModel'), test4_dat[:, 1], rtol=0.03)

    def test_calculate_github_test4_spline_resolution(self):
        p = RefnxWrapper()
        p.create_material('Material1')
        p.update_material('Material1', real=2.070, imag=0.000)
        p.create_material('Material2')
        p.update_material('Material2', real=3.450, imag=0.100)
        p.create_material('Material3')
        p.update_material('Material3', real=5.000, imag=0.01)
        p.create_material('Material4')
        p.update_material('Material4', real=6.000, imag=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thick=100, rough=3.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', thick=200, rough=1.0)
        p.create_layer('Layer4')
        p.assign_material_to_layer('Material4', 'Layer4')
        p.update_layer('Layer4', rough=5.0)
        p.create_item('Item1')
        p.add_layer_to_item('Layer1', 'Item1')
        p.create_item('Item2')
        p.add_layer_to_item('Layer2', 'Item2')
        p.create_item('Item3')
        p.add_layer_to_item('Layer3', 'Item3')
        p.create_item('Item4')
        p.add_layer_to_item('Layer4', 'Item4')
        p.add_item('Item1', 'MyModel')
        p.add_item('Item2', 'MyModel')
        p.add_item('Item3', 'MyModel')
        p.add_item('Item4', 'MyModel')
        p.update_model('MyModel', bkg=0)
        sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        p.set_resolution_function(LinearSpline(test4_dat[:, 0], sigma_to_fwhm * test4_dat[:, 3]))
        assert_allclose(p.calculate(test4_dat[:, 0], 'MyModel'), test4_dat[:, 1], rtol=0.03)


# https://github.com/reflectivity/analysis/tree/master/validation/test/unpolarised
# validation/test/unpolarised/data/test4.dat
test4_dat_raw = """
4.999999999999999237e-03 9.660499468321636085e-01 0.000000000000000000e+00 1.061652250360023761e-04
5.208965929570049704e-03 9.646283312283907563e-01 0.000000000000000000e+00 1.106022080235347408e-04
5.426665211084315613e-03 9.631435299315659337e-01 0.000000000000000000e+00 1.152246266659623608e-04
5.653462839144248603e-03 9.615923806647335148e-01 0.000000000000000000e+00 1.200402309100852380e-04
5.889739062638551202e-03 9.599715341545977942e-01 0.000000000000000000e+00 1.250570945976711191e-04
6.135890022268412555e-03 9.582774463689063271e-01 0.000000000000000000e+00 1.302836290020575514e-04
6.392328414716996060e-03 9.565063736552937845e-01 0.000000000000000000e+00 1.357285969304924749e-04
6.659484184576673627e-03 9.546543728325473932e-01 0.000000000000000000e+00 1.414011274158563051e-04
6.937805245194089872e-03 9.527173094019888433e-01 0.000000000000000000e+00 1.473107310223976691e-04
7.227758229641680597e-03 9.506908787954021500e-01 0.000000000000000000e+00 1.534673157911454707e-04
7.529829273074612438e-03 9.485706483542604150e-01 0.000000000000000000e+00 1.598812038517289096e-04
7.844524827784976270e-03 9.463521322252024248e-01 0.000000000000000000e+00 1.665631487284599923e-04
8.172372512319661317e-03 9.440309187595408158e-01 0.000000000000000000e+00 1.735243533696914056e-04
8.513921996085575816e-03 9.416028824956215182e-01 0.000000000000000000e+00 1.807764889306791788e-04
8.869745920925360336e-03 9.390645344556776131e-01 0.000000000000000000e+00 1.883317143414410305e-04
9.240440861208627787e-03 9.364136032254166686e-01 0.000000000000000000e+00 1.962026966924171388e-04
9.626628324048536189e-03 9.336500112051923095e-01 0.000000000000000000e+00 2.044026324721134858e-04
1.002895579132056722e-02 9.307775499924635376e-01 0.000000000000000000e+00 2.129452696923334900e-04
1.044809780523061650e-02 9.278068446083937992e-01 0.000000000000000000e+00 2.218449309380942302e-04
1.088475709925237553e-02 9.247608214119300563e-01 0.000000000000000000e+00 2.311165373808706110e-04
1.133966577633027141e-02 9.216853828676729865e-01 0.000000000000000000e+00 2.407756337954316216e-04
1.181358653632318123e-02 9.186719597214783040e-01 0.000000000000000000e+00 2.508384146222077419e-04
1.230731395474697620e-02 9.159110487636203946e-01 0.000000000000000000e+00 2.613217511188890515e-04
1.282167581495980910e-02 9.138467828784078151e-01 0.000000000000000000e+00 2.722432196467755149e-04
1.335753449602358582e-02 9.108355606753477662e-01 0.000000000000000000e+00 2.836211311393017432e-04
1.391578841856870238e-02 7.335981045780506360e-01 0.000000000000000000e+00 2.954745618021484190e-04
1.449737355108597463e-02 2.213862971885717790e-01 0.000000000000000000e+00 3.078233850964063527e-04
1.510326497917135677e-02 5.065074610089120161e-02 0.000000000000000000e+00 3.206883050584202191e-04
1.573447854035441712e-02 1.590368472895162949e-02 0.000000000000000000e+00 3.340908910121754400e-04
1.639207252725145142e-02 4.800075501038165819e-03 0.000000000000000000e+00 3.480536137324245511e-04
1.707714946189881067e-02 1.223948799745163416e-03 0.000000000000000000e+00 3.625998831191868982e-04
1.779085794424128605e-02 5.046568494808504907e-04 0.000000000000000000e+00 3.777540874467854378e-04
1.853439457787469519e-02 8.404754652332931120e-04 0.000000000000000000e+00 3.935416342532259133e-04
1.930900597627142543e-02 1.458717108645946623e-03 0.000000000000000000e+00 4.099889929384741916e-04
2.011599085285246946e-02 2.022184226696178543e-03 0.000000000000000000e+00 4.271237391430496191e-04
2.095670219841026111e-02 2.397327154528816270e-03 0.000000000000000000e+00 4.449746009813422727e-04
2.183254954953296745e-02 2.551634050707026792e-03 0.000000000000000000e+00 4.635715072071680543e-04
2.274500135183342431e-02 2.503723878205208874e-03 0.000000000000000000e+00 4.829456373923148504e-04
2.369558742194500037e-02 2.297014419596041072e-03 0.000000000000000000e+00 5.031294742022117319e-04
2.468590151241202915e-02 1.984702877815211226e-03 0.000000000000000000e+00 5.241568578563629999e-04
2.571760398377520920e-02 1.620535097096735370e-03 0.000000000000000000e+00 5.460630428648573405e-04
2.679242458833201376e-02 1.252789959601383627e-03 0.000000000000000000e+00 5.688847571360784120e-04
2.791216537023925512e-02 9.202954264280633354e-04 0.000000000000000000e+00 5.926602635547127294e-04
2.907870368682024739e-02 6.499790081461440973e-04 0.000000000000000000e+00 6.174294241333008141e-04
3.029399535614193950e-02 4.557902347035073598e-04 0.000000000000000000e+00 6.432337668448841412e-04
3.156007793613933854e-02 3.389679007057118062e-04 0.000000000000000000e+00 6.701165552488013702e-04
3.287907414078505841e-02 2.896305199626667578e-04 0.000000000000000000e+00 6.981228610263705585e-04
3.425319539903139837e-02 2.895827802138814912e-04 0.000000000000000000e+00 7.272996395480660261e-04
3.568474556249201513e-02 3.160833603894238047e-04 0.000000000000000000e+00 7.576958085988904973e-04
3.717612476807939659e-02 3.461421601729252370e-04 0.000000000000000000e+00 7.893623303939302548e-04
3.872983346207417577e-02 3.607471650414172397e-04 0.000000000000000000e+00 8.223522970216000675e-04
4.034847659237329048e-02 3.483107825318375100e-04 0.000000000000000000e+00 8.567210194578371010e-04
4.203476797594541542e-02 3.066260485666312411e-04 0.000000000000000000e+00 8.925261203004783862e-04
4.379153484881635749e-02 2.427846473412773824e-04 0.000000000000000000e+00 9.298276303793059786e-04
4.562172260621279868e-02 1.708603072276521735e-04 0.000000000000000000e+00 9.686880894037319147e-04
4.752839974081164709e-02 1.076850793981012493e-04 0.000000000000000000e+00 1.009172650816869167e-03
4.951476298737475523e-02 6.765012337432836328e-05 0.000000000000000000e+00 1.051349191031792632e-03
5.158414268239425865e-02 5.796797455235872996e-05 0.000000000000000000e+00 1.095288423233128482e-03
5.374000834773436097e-02 7.600076548404877867e-05 0.000000000000000000e+00 1.141064015934773043e-03
5.598597450763168060e-02 1.098724565425083957e-04 0.000000000000000000e+00 1.188752716492522165e-03
5.832580674880616378e-02 1.425800498622337880e-04 0.000000000000000000e+00 1.238434479778678654e-03
6.076342803384361668e-02 1.584158826364232800e-04 0.000000000000000000e+00 1.290192602234388778e-03
6.330292527843461858e-02 1.492439980337561915e-04 0.000000000000000000e+00 1.344113861524451250e-03
6.594855620349690528e-02 1.177519033425392445e-04 0.000000000000000000e+00 1.400288662028740172e-03
6.870475647367019212e-02 7.577842573871102763e-05 0.000000000000000000e+00 1.458811186414187651e-03
7.157614713415108576e-02 3.813747620286129380e-05 0.000000000000000000e+00 1.519779553541433575e-03
7.456754235833716604e-02 1.502718568743208691e-05 0.000000000000000000e+00 1.583295982970901261e-03
7.768395751926998605e-02 7.432986794932621371e-06 0.000000000000000000e+00 1.649466966344109589e-03
8.093061759840886049e-02 8.479436602724126006e-06 0.000000000000000000e+00 1.718403445927546482e-03
8.431296594583481685e-02 9.720659036587660457e-06 0.000000000000000000e+00 1.790221000618471897e-03
8.783667340657071165e-02 7.475971384271801476e-06 0.000000000000000000e+00 1.865040039724485304e-03
9.150764782831966038e-02 4.045160813957839975e-06 0.000000000000000000e+00 1.942986004841762470e-03
9.533204396656239088e-02 3.178505467676818043e-06 0.000000000000000000e+00 2.024189580170434287e-03
9.931627380361947310e-02 4.953666565173862248e-06 0.000000000000000000e+00 2.108786911619698213e-03
1.034670172989808762e-01 5.966968494098515925e-06 0.000000000000000000e+00 2.196919835070051605e-03
1.077912335889252976e-01 4.395731401080907728e-06 0.000000000000000000e+00 2.288736114175310681e-03
1.122961726542076877e-01 2.838663311476676714e-06 0.000000000000000000e+00 2.384389688103147591e-03
1.169893874753767798e-01 4.376365975608713520e-06 0.000000000000000000e+00 2.484040929629491301e-03
1.218787466961013116e-01 7.365431957590666160e-06 0.000000000000000000e+00 2.587856914019505839e-03
1.269724478157380210e-01 7.444758004751446953e-06 0.000000000000000000e+00 2.696011699145979719e-03
1.322790309332581016e-01 4.174506384415917617e-06 0.000000000000000000e+00 2.808686617314733527e-03
1.378073930655767942e-01 1.378897368540984724e-06 0.000000000000000000e+00 2.926070579286359132e-03
1.435668030642915372e-01 8.179962055944333663e-07 0.000000000000000000e+00 3.048360391003989995e-03
1.495669171558374755e-01 7.196203711450000268e-07 0.000000000000000000e+00 3.175761083558122696e-03
1.558177951311167375e-01 4.074965533304633985e-07 0.000000000000000000e+00 3.308486256941745463e-03
1.623299172117425582e-01 5.697106402976437542e-07 0.000000000000000000e+00 3.446758438172057518e-03
1.691142016211787946e-01 6.318259741325602534e-07 0.000000000000000000e+00 3.590809454379265478e-03
1.761820228902321317e-01 5.884226632043859758e-07 0.000000000000000000e+00 3.740880821487923939e-03
1.835452309275898974e-01 1.086074476142989135e-06 0.000000000000000000e+00 3.897224149142521597e-03
1.912161708873765797e-01 1.080905569118702644e-06 0.000000000000000000e+00 4.060101562556204732e-03
1.992077038670377609e-01 4.319349827800466143e-07 0.000000000000000000e+00 4.229786141989877457e-03
2.075332284702559316e-01 1.748578430054353337e-07 0.000000000000000000e+00 4.406562380598564102e-03
2.162067032710481007e-01 1.052279192866519288e-07 0.000000000000000000e+00 4.590726661412603340e-03
2.252426702167101280e-01 1.207279972283447134e-07 0.000000000000000000e+00 4.782587754253420607e-03
2.346562790088451700e-01 1.538042650955484403e-07 0.000000000000000000e+00 4.982467333417002313e-03
2.444633125033515431e-01 2.521547835091313121e-07 0.000000000000000000e+00 5.190700516992979158e-03
2.546802131719588802e-01 2.588416113893713325e-07 0.000000000000000000e+00 5.407636428723615156e-03
2.653241106696743179e-01 9.730928539681922884e-08 0.000000000000000000e+00 5.633638783344635526e-03
2.764128505543612668e-01 3.779049549799957986e-08 0.000000000000000000e+00 5.869086496389332337e-03
2.879650242066011945e-01 3.707321909920906896e-08 0.000000000000000000e+00 6.114374319478338426e-03
2.999999999999999334e-01 6.224636414855785008e-08 0.000000000000000000e+00 6.369913502160142078e-03
"""

test4_dat_rows = test4_dat_raw.split('\n')[1:-1]
test4_dat = np.array([np.array(row.split()) for row in test4_dat_rows], dtype=float)
