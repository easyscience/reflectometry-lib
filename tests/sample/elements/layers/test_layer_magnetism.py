# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for LayerMagnetism and magnetic-parameter plumbing through the calculators.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from easyreflectometry.calculators.factory import CalculatorFactory
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.005, 0.3, 50)


def _magnetic_model(magnetism: LayerMagnetism | None) -> Model:
    vacuum = Material(sld=0, isld=0, name='Vacuum')
    material = Material(sld=4.0, isld=0, name='Sld 4')
    si = Material(sld=2.047, isld=0, name='Si')
    superphase = Layer(material=vacuum, thickness=0, roughness=0, name='Vacuum Superphase')
    layer = Layer(material=material, thickness=100, roughness=0, magnetism=magnetism, name='Sld 4 Layer')
    subphase = Layer(material=si, thickness=0, roughness=0, name='Si Subphase')
    sample = Sample(Multilayer(superphase), Multilayer(layer), Multilayer(subphase), name='Sample')
    model = Model(sample=sample, scale=1, background=0, name='Magnetic Model')
    model.resolution_function = PercentageFwhm(0)
    return model


class TestLayerMagnetism:
    def test_default_construction(self):
        magnetism = LayerMagnetism()
        assert_equal(magnetism.name, 'EasyLayerMagnetism')
        assert_equal(magnetism.rho_m.value, 0.0)
        assert_equal(magnetism.rho_m.fixed, True)
        assert_equal(magnetism.theta_m.value, 270.0)
        assert_equal(magnetism.theta_m.min, 0.0)
        assert_equal(magnetism.theta_m.max, 360.0)
        assert_equal(magnetism.theta_m.fixed, True)

    def test_construction_with_values(self):
        magnetism = LayerMagnetism(rho_m=2.5, theta_m=45.0, name='FeMoment')
        assert_equal(magnetism.name, 'FeMoment')
        assert_equal(magnetism.rho_m.value, 2.5)
        assert_equal(magnetism.theta_m.value, 45.0)

    def test_dict_round_trip(self):
        magnetism = LayerMagnetism(rho_m=1.5, theta_m=90.0)
        magnetism_dict = magnetism.as_dict()
        reloaded = LayerMagnetism.from_dict(magnetism_dict)
        assert_equal(reloaded.rho_m.value, 1.5)
        assert_equal(reloaded.theta_m.value, 90.0)
        assert sorted(magnetism.as_dict()) == sorted(reloaded.as_dict())


class TestLayerWithMagnetism:
    def test_layer_default_is_non_magnetic(self):
        layer = Layer()
        assert layer.magnetism is None

    def test_layer_dict_round_trip_with_magnetism(self):
        layer = Layer(magnetism=LayerMagnetism(rho_m=2.0, theta_m=45.0), name='MagneticLayer')
        layer_dict = layer.as_dict()
        reloaded = Layer.from_dict(layer_dict)
        assert reloaded.magnetism is not None
        assert_equal(reloaded.magnetism.rho_m.value, 2.0)
        assert_equal(reloaded.magnetism.theta_m.value, 45.0)
        assert sorted(layer.as_dict()) == sorted(reloaded.as_dict())

    def test_magnetic_parameters_are_fittable_variables(self):
        magnetism = LayerMagnetism(rho_m=2.0, theta_m=45.0)
        layer = Layer(magnetism=magnetism)
        variable_names = [variable.name for variable in layer.get_all_variables()]
        assert 'rho_m' in variable_names
        assert 'theta_m' in variable_names


class TestMagnetismThroughCalculator:
    def _interface(self, name: str) -> CalculatorFactory:
        interface = CalculatorFactory()
        interface.switch(name)
        return interface

    def test_model_interface_enables_magnetism_and_binds_parameters(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        interface = self._interface('refl1d')
        model.interface = interface
        calculator = interface()

        assert model.has_magnetism is True
        assert calculator.include_magnetism is True
        layer = model.sample[1].layers[0]
        wrapper = calculator._wrapper
        assert wrapper.get_layer_value(layer.unique_name, 'magnetism_rhoM') == 2.0
        assert wrapper.get_layer_value(layer.unique_name, 'magnetism_thetaM') == 45.0

    def test_magnetic_parameter_change_changes_reflectivity(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = self._interface('refl1d')

        before = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)
        model.sample[1].layers[0].magnetism.rho_m = 4.0
        after = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)

        assert not np.allclose(before['pp'], after['pp'])
        assert not np.allclose(before['mm'], after['mm'])

    def test_model_parameters_match_direct_wrapper_values(self):
        # Setting rho_m/theta_m through the model must reproduce the reflectivity
        # obtained by setting magnetism_rhoM/thetaM directly on the wrapper.
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = self._interface('refl1d')
        via_model = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)

        reference_model = _magnetic_model(None)
        interface = self._interface('refl1d')
        reference_model.interface = interface
        calculator = interface()
        calculator.include_magnetism = True
        layer_name = reference_model.sample[1].layers[0].unique_name
        calculator._wrapper.update_layer(layer_name, magnetism_rhoM=2.0, magnetism_thetaM=45.0)
        via_wrapper = interface.polarized_reflectivity_profiles(Q, reference_model.unique_name)

        for channel in ('pp', 'pm', 'mp', 'mm'):
            assert_allclose(via_model[channel], via_wrapper[channel], rtol=1e-10)

    def test_magnetism_added_after_interface_is_bound(self):
        model = _magnetic_model(None)
        model.interface = self._interface('refl1d')
        calculator = model.interface()
        assert calculator.include_magnetism is False

        layer = model.sample[1].layers[0]
        layer.magnetism = LayerMagnetism(rho_m=2.0, theta_m=45.0)

        assert calculator.include_magnetism is True
        wrapper = calculator._wrapper
        assert wrapper.get_layer_value(layer.unique_name, 'magnetism_rhoM') == 2.0
        # The parameter is live: a change propagates to the backend.
        layer.magnetism.rho_m = 3.0
        assert wrapper.get_layer_value(layer.unique_name, 'magnetism_rhoM') == 3.0

    def test_removing_last_magnetism_disables_calculator_magnetism(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = self._interface('refl1d')
        calculator = model.interface()
        layer = model.sample[1].layers[0]
        detached = layer.magnetism

        layer.magnetism = None

        # Model and calculator agree again: no magnetism anywhere.
        assert layer.magnetism is None
        assert model.has_magnetism is False
        assert calculator.include_magnetism is False
        assert calculator._wrapper.get_layer_value(layer.unique_name, 'magnetism_rhoM') == 0.0
        # The plain (unpolarized) calculation path works.
        reflectivity = calculator.reflectity_profile(Q, model.unique_name)
        assert len(reflectivity) == len(Q)
        # The detached parameters no longer reach the backend.
        detached.rho_m = 5.0
        assert calculator._wrapper.get_layer_value(layer.unique_name, 'magnetism_rhoM') == 0.0

    def test_removing_one_of_two_magnetic_layers_keeps_magnetism(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = self._interface('refl1d')
        calculator = model.interface()
        subphase = model.sample[2].layers[0]
        subphase.magnetism = LayerMagnetism(rho_m=1.0, theta_m=270.0)

        model.sample[1].layers[0].magnetism = None

        # One magnetic layer remains: the polarized path stays on, its values intact.
        assert model.has_magnetism is True
        assert calculator.include_magnetism is True
        assert calculator._wrapper.get_layer_value(subphase.unique_name, 'magnetism_rhoM') == 1.0

    def test_magnetic_layer_with_refnx_raises(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        with pytest.raises(NotImplementedError):
            model.interface = self._interface('refnx')

    def test_supports_magnetism_capability_flag(self):
        assert self._interface('refl1d')().supports_magnetism is True
        assert self._interface('refnx')().supports_magnetism is False

    def test_project_exposes_calculator_capability(self):
        from easyreflectometry.project import Project

        project = Project()
        project.calculator = 'refl1d'
        assert project.calculator_supports_magnetism is True
        project.calculator = 'refnx'
        assert project.calculator_supports_magnetism is False

    def test_fit_recovers_rho_m_from_synthetic_data(self):
        from easyreflectometry.data import DataSet1D
        from easyreflectometry.fitting import MultiFitter

        # Synthesize noiseless pp data from a model with a known moment.
        truth = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        truth.interface = self._interface('refl1d')
        reflectivity = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)['pp']
        # `ye` holds variances.
        data = DataSet1D(name='synthetic_pp', x=Q, y=reflectivity, ye=(0.01 * reflectivity) ** 2)

        # Fit a model starting from the wrong moment; only rho_m is free.
        model = _magnetic_model(LayerMagnetism(rho_m=1.0, theta_m=270.0))
        model.interface = self._interface('refl1d')
        rho_m = model.sample[1].layers[0].magnetism.rho_m
        rho_m.fixed = False
        rho_m.min = 0.0
        rho_m.max = 5.0

        fitter = MultiFitter(model)
        result = fitter.fit_single_data_set_1d(data)

        assert result.success
        assert_allclose(rho_m.value, 2.5, atol=0.01)
