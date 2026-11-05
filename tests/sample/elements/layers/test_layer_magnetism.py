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
        variable_names = [variable.display_name for variable in layer.get_all_variables()]
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
        # The parameter is live: a change reaches the backend at the next evaluation.
        layer.magnetism.rho_m = 3.0
        calculator._push_values()
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


FE_RHO_M = 5.0420761  # 1e-6 / angstrom^2: 7.874 g/cm^3, 2.2 muB per atom


class TestRhoMCoupling:
    """`Layer.rho_m_coupled`: the layer's magnetic SLD follows a `MaterialMagnetic`."""

    @staticmethod
    def _coupled_model():
        from easyreflectometry.sample import MaterialMagnetic

        model = _magnetic_model(None)
        film = model.sample[1].layers[0]
        film.material = MaterialMagnetic(name='Fe')
        film.rho_m_coupled = True
        interface = CalculatorFactory()
        interface.switch('refl1d')
        model.interface = interface
        return model, film

    @staticmethod
    def _backend_rho_m(film):
        # Values reach the backend when the model is evaluated.
        film.interface()._push_values()
        return film.interface()._wrapper.get_layer_value(film.unique_name, 'magnetism_rhoM')

    def test_coupling_attaches_magnetism_and_follows_the_material(self):
        model, film = self._coupled_model()

        assert film.rho_m_coupled
        assert film.magnetism.theta_m.value == 270.0
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M)
        film.material.magnetic_moment = 1.1
        assert self._backend_rho_m(film) == pytest.approx(FE_RHO_M / 2)

    def test_material_without_rho_m_cannot_couple(self):
        layer = Layer(material=Material(sld=4.0), magnetism=LayerMagnetism())
        with pytest.raises(ValueError, match='no magnetic SLD'):
            layer.rho_m_coupled = True

    def test_uncoupling_keeps_the_value_and_is_idempotent(self):
        _, film = self._coupled_model()

        film.rho_m_coupled = False
        film.rho_m_coupled = False

        rho_m = film.magnetism.rho_m
        assert rho_m.independent and rho_m.fixed
        assert rho_m.value == pytest.approx(FE_RHO_M)
        film.material.magnetic_moment = 1.1
        assert rho_m.value == pytest.approx(FE_RHO_M)

    @pytest.mark.parametrize('replace', ['property', 'assign_material'])
    def test_replacing_the_material_rewires(self, replace):
        from easyreflectometry.sample import MaterialMagnetic

        _, film = self._coupled_model()
        old = film.material
        nickel = MaterialMagnetic(chemical_structure='Ni', density=8.908, magnetic_moment=0.6, name='Ni')

        if replace == 'property':
            film.material = nickel
        else:
            nickel.interface = film.interface  # `assign_material` expects a bound material
            film.assign_material(nickel)

        assert film.rho_m_coupled
        assert film.magnetism.rho_m.value == pytest.approx(nickel.rho_m.value)
        old.magnetic_moment = 0.1
        assert film.magnetism.rho_m.value == pytest.approx(nickel.rho_m.value)

    def test_non_magnetic_material_uncouples_with_a_warning(self):
        _, film = self._coupled_model()
        with pytest.warns(UserWarning, match='no longer coupled'):
            film.material = Material(sld=4.0)
        assert not film.rho_m_coupled
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M)

    def test_removing_magnetism_uncouples(self):
        _, film = self._coupled_model()
        removed = film.magnetism

        film.magnetism = None

        assert not film.rho_m_coupled
        assert removed.rho_m.independent

    def test_replacing_magnetism_keeps_the_coupling(self):
        _, film = self._coupled_model()
        old = film.magnetism

        film.magnetism = LayerMagnetism(theta_m=0.0)
        film.material.magnetic_moment = 1.1

        assert film.rho_m_coupled
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M / 2)
        assert self._backend_rho_m(film) == pytest.approx(FE_RHO_M / 2)
        # The detached object neither follows the material nor reaches the backend.
        assert old.rho_m.independent
        old.rho_m.value = 9.0
        assert self._backend_rho_m(film) == pytest.approx(FE_RHO_M / 2)

    def test_unconstrain_reads_as_uncoupled_and_stays_so(self):
        from easyreflectometry.constraints import unconstrain
        from easyreflectometry.sample import MaterialMagnetic

        _, film = self._coupled_model()
        unconstrain(film.magnetism.rho_m)

        assert not film.rho_m_coupled
        film.material = MaterialMagnetic(name='Fe2')
        assert film.magnetism.rho_m.independent

    def test_round_trip_keeps_the_coupling(self):
        model, film = self._coupled_model()

        restored = Model.from_dict(model.as_dict())
        restored_film = restored.sample[1].layers[0]
        restored_film.material.magnetic_moment = 1.1

        assert restored_film.rho_m_coupled
        assert restored_film.magnetism.rho_m.value == pytest.approx(FE_RHO_M / 2)

    def test_project_round_trip_shares_one_material(self, tmp_path):
        from easyscience import global_object

        from easyreflectometry.model import ModelCollection
        from easyreflectometry.project import Project

        model, film = self._coupled_model()
        second = Layer(material=film.material, thickness=50, name='second', rho_m_coupled=True)
        model.sample[1].layers.append(second)
        project = Project()
        project.set_path_project_parent(tmp_path)
        project.calculator = 'refl1d'
        project.models = ModelCollection(model)
        project.save_as_json()

        global_object.registry.clear()
        reloaded = Project()
        reloaded.load_from_json(project.path_json)
        first, other = reloaded.models[0].sample[1].layers
        first.material.magnetic_moment = 1.1

        assert first.material is other.material
        assert first.rho_m_coupled and other.rho_m_coupled
        assert other.magnetism.rho_m.value == pytest.approx(FE_RHO_M / 2)

    def test_fit_recovers_the_moment(self):
        from easyreflectometry.data import DataSet1D
        from easyreflectometry.fitting import MultiFitter

        truth, _ = self._coupled_model()
        reflectivity = truth.interface().polarized_reflectivity_profiles(Q, truth.unique_name)['pp']
        data = DataSet1D(name='synthetic_pp', x=Q, y=reflectivity, ye=(0.01 * reflectivity) ** 2)

        model, film = self._coupled_model()
        moment = film.material.magnetic_moment
        moment.value = 1.0
        moment.fixed = False  # density stays fixed: the data constrain density x moment

        result = MultiFitter(model).fit_single_data_set_1d(data)

        assert result.success
        assert_allclose(moment.value, 2.2, atol=0.01)

    def test_material_moment_never_reaches_an_uncoupled_layer(self):
        from easyreflectometry.sample import MaterialMagnetic

        model = _magnetic_model(LayerMagnetism(rho_m=1.0))
        uncoupled = model.sample[1].layers[0]
        iron = MaterialMagnetic(name='Fe')
        uncoupled.material = iron
        coupled = Layer(material=iron, thickness=50, name='coupled', magnetism=LayerMagnetism())
        model.sample[1].layers.append(coupled)
        interface = CalculatorFactory()
        interface.switch('refl1d')
        model.interface = interface

        # Coupling a bound layer takes the material's value at once.
        coupled.rho_m_coupled = True
        assert coupled.magnetism.rho_m.value == pytest.approx(FE_RHO_M)
        assert iron.rho_m.value == pytest.approx(FE_RHO_M)

        iron.magnetic_moment = 1.1

        interface()._push_values()  # values reach the backend when the model is evaluated
        wrapper = interface()._wrapper
        assert wrapper.get_layer_value(coupled.unique_name, 'magnetism_rhoM') == pytest.approx(FE_RHO_M / 2)
        assert wrapper.get_layer_value(uncoupled.unique_name, 'magnetism_rhoM') == pytest.approx(1.0)


class TestRhoMUserConstraint:
    """A user constraint on a coupled ``rho_m`` replaces the coupling; it is not the coupling."""

    @staticmethod
    def _constrained_film(scaled_to_material: bool):
        from easyscience.variable import Parameter

        from easyreflectometry.constraints import constrain
        from easyreflectometry.sample import MaterialMagnetic

        iron = MaterialMagnetic(name='Fe')
        film = Layer(material=iron, thickness=100, name='film', rho_m_coupled=True)
        if scaled_to_material:
            source = iron.rho_m
            constrain(film.magnetism.rho_m, '0.5 * m', m=source)
        else:
            source = Parameter(display_name='external', value=1.0, unit='1/angstrom^2')
            constrain(film.magnetism.rho_m, 'x', x=source)
        return film, source

    @pytest.mark.parametrize('scaled_to_material', [False, True])
    def test_reads_as_uncoupled(self, scaled_to_material):
        film, _ = self._constrained_film(scaled_to_material)
        assert not film.rho_m_coupled

    def test_survives_a_material_change(self):
        from easyreflectometry.sample import MaterialMagnetic

        film, external = self._constrained_film(scaled_to_material=False)
        film.material = MaterialMagnetic(chemical_structure='Ni', density=8.908, magnetic_moment=0.6, name='Ni')

        external.value = 2.0
        assert film.magnetism.rho_m.value == pytest.approx(2.0)
        assert not film.rho_m_coupled

    def test_export_writes_the_layer_value(self):
        from easyreflectometry.orso_utils import _layer_material_to_orso
        from easyreflectometry.special.calculations import MAGNETIC_SCATTERING_LENGTH

        film, _ = self._constrained_film(scaled_to_material=False)
        moment = _layer_material_to_orso(film).magnetic_moment.magnitude
        assert moment * film.material.number_density * MAGNETIC_SCATTERING_LENGTH * 1e6 == pytest.approx(1.0)

    def test_explicit_coupling_replaces_the_constraint(self):
        film, external = self._constrained_film(scaled_to_material=False)

        film.rho_m_coupled = True

        assert film.rho_m_coupled
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M)
        external.value = 2.0
        assert film.magnetism.rho_m.value == pytest.approx(FE_RHO_M)

    def test_saved_as_uncoupled(self):
        film, _ = self._constrained_film(scaled_to_material=False)
        assert film.as_dict()['rho_m_coupled'] is False
