# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for explicit-channel calculation, the polarized reflectivity cache,
per-channel experiment loading, and simultaneous multi-channel fitting.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.calculators.refl1d import wrapper as refl1d_wrapper
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.005, 0.3, 50)


@pytest.fixture(autouse=True)
def _isolated_global_object():
    """Leave the easyscience object map clean for the next test file.

    Several tests here build a `Project` (and therefore a 'project_models'
    collection); a leftover registration makes the *next* module's first
    `Project()` fail with 'Object name project_models already exists'.
    """
    global_object.map._clear()
    yield
    global_object.map._clear()


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


def _refl1d_interface() -> CalculatorFactory:
    interface = CalculatorFactory()
    interface.switch('refl1d')
    return interface


def _polarized_data(channels: dict[str, np.ndarray], model=None) -> PolarizedDataSet:
    datasets = {
        channel: DataSet1D(name=channel, x=Q, y=reflectivity, ye=(0.01 * reflectivity) ** 2)
        for channel, reflectivity in channels.items()
    }
    return PolarizedDataSet(name='synthetic', channels=datasets, model=model)


class TestCalculateChannel:
    def test_channels_match_calculate_polarized(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()

        reference = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
        for channel in ('pp', 'pm', 'mp', 'mm'):
            assert_allclose(
                calculator.reflectivity_profile_channel(Q, model.unique_name, channel),
                reference[channel],
                rtol=1e-12,
            )

    def test_pp_without_magnetism_falls_back_to_unpolarized(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        calculator = model.interface()

        assert_allclose(
            calculator.reflectivity_profile_channel(Q, model.unique_name, 'pp'),
            calculator.reflectity_profile(Q, model.unique_name),
            rtol=1e-12,
        )

    def test_spin_flip_without_magnetism_raises(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        with pytest.raises(ValueError):
            model.interface().reflectivity_profile_channel(Q, model.unique_name, 'pm')

    def test_fit_func_for_channel(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()

        reference = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)
        fit_func = model.interface.fit_func_for_channel('mm')
        assert_allclose(fit_func(Q, model.unique_name), reference['mm'], rtol=1e-12)

    def test_magnetism_inside_a_repeating_multilayer_raises_instead_of_silently_wrong(self):
        """refl1d itself does not support repeated magnetic slabs (`profile.repeat`).

        `magnetism.ipynb` documents this as a known limitation; this test pins
        that the failure is loud (`NotImplementedError`) rather than a silent
        wrong answer, so the tutorial's claim stays checked against behaviour.
        """
        from easyreflectometry.sample import RepeatingMultilayer

        vacuum = Material(sld=0, isld=0, name='Vacuum')
        material = Material(sld=4.0, isld=0, name='Fe')
        si = Material(sld=2.047, isld=0, name='Si')
        superphase = Layer(material=vacuum, thickness=0, roughness=0, name='Vacuum Superphase')
        layer = Layer(
            material=material,
            thickness=50,
            roughness=2,
            magnetism=LayerMagnetism(rho_m=5.0, theta_m=40.0),
            name='Fe film',
        )
        subphase = Layer(material=si, thickness=0, roughness=3, name='Si Subphase')
        repeated = RepeatingMultilayer(layer, repetitions=3, name='Repeated Fe')
        sample = Sample(Multilayer(superphase), repeated, Multilayer(subphase), name='Repeated magnetic sample')
        model = Model(sample=sample, scale=1, background=0, name='Repeated magnetic model')
        model.resolution_function = PercentageFwhm(0)
        model.interface = _refl1d_interface()

        with pytest.raises(NotImplementedError, match='[Rr]epeat'):
            model.interface().polarized_reflectivity_profiles(Q, model.unique_name)


class TestPolarizedCache:
    def test_repeated_calculation_hits_cache(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()

        with patch.object(refl1d_wrapper.names, 'Experiment', wraps=refl1d_wrapper.names.Experiment) as experiment:
            first = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            second = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            assert experiment.call_count == 1
            # All four channels through calculate_channel: still no new evaluation.
            for channel in ('pp', 'pm', 'mp', 'mm'):
                calculator.reflectivity_profile_channel(Q, model.unique_name, channel)
            assert experiment.call_count == 1
        for channel in ('pp', 'pm', 'mp', 'mm'):
            assert_allclose(first[channel], second[channel], rtol=1e-15)

    def test_parameter_change_invalidates_cache(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()

        with patch.object(refl1d_wrapper.names, 'Experiment', wraps=refl1d_wrapper.names.Experiment) as experiment:
            before = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            model.sample[1].layers[0].magnetism.rho_m = 3.0
            after = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            assert experiment.call_count == 2
        assert not np.allclose(before['mm'], after['mm'])

    def test_q_dtype_is_normalized_before_keying(self):
        # Keying happens after normalization to float64 plus explicit shape, so
        # byte-identical arrays of different dtype/shape can never collide. A
        # float32 grid whose values are exactly representable normalizes to the
        # same key as its float64 twin and shares the cache entry.
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()
        q_pow2 = np.array([0.03125, 0.0625, 0.125, 0.25])  # exact in float32

        with patch.object(refl1d_wrapper.names, 'Experiment', wraps=refl1d_wrapper.names.Experiment) as experiment:
            first = calculator.polarized_reflectivity_profiles(q_pow2, model.unique_name)
            second = calculator.polarized_reflectivity_profiles(q_pow2.astype(np.float32), model.unique_name)
            assert experiment.call_count == 1
        for channel in ('pp', 'pm', 'mp', 'mm'):
            assert len(first[channel]) == len(q_pow2)
            assert_allclose(second[channel], first[channel], rtol=1e-15)

    def test_different_q_grids_coexist_within_one_state(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()
        q_other = np.linspace(0.01, 0.2, 30)

        with patch.object(refl1d_wrapper.names, 'Experiment', wraps=refl1d_wrapper.names.Experiment) as experiment:
            calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            calculator.polarized_reflectivity_profiles(q_other, model.unique_name)
            assert experiment.call_count == 2
            # Both grids now cached for the same model state.
            calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            calculator.polarized_reflectivity_profiles(q_other, model.unique_name)
            assert experiment.call_count == 2


class TestLoadPolarizedExperiment:
    @staticmethod
    def _write_channel_file(directory, name: str) -> str:
        path = directory / name
        q = np.linspace(0.01, 0.2, 20)
        reflectivity = np.exp(-q * 30)
        error = 0.01 * reflectivity
        np.savetxt(path, np.column_stack([q, reflectivity, error]))
        return str(path)

    def test_load_polarized_experiment(self, tmp_path):
        pp_path = self._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        new_index = project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})

        # The index is returned so a GUI can make the new experiment current.
        assert new_index == 0
        experiment = project.experiments[0]
        assert isinstance(experiment, PolarizedDataSet)
        assert experiment.available_channels == [PolarizationChannel.PP, PolarizationChannel.MM]
        assert experiment.name == 'Polarized experiment 0'
        assert experiment.model is project.models[0]
        assert experiment['pp'].model is project.models[0]
        assert len(experiment['pp'].x) == 20

    def test_second_polarized_experiment_gets_the_next_index(self, tmp_path):
        pp_path = self._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        first = project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})
        second = project.load_polarized_experiment({'pp': pp_path})

        assert (first, second) == (0, 1)
        assert len(project.experiments) == 2

    def test_multi_dataset_file_is_rejected(self, tmp_path):
        import os

        multi_path = os.path.join(os.path.dirname(__file__), '_static', 'test_example2.ort')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        with pytest.raises(ValueError, match='multiple datasets'):
            project.load_polarized_experiment({'pp': multi_path, 'mm': mm_path})
        assert len(project.experiments) == 0

    def test_suggest_polarized_channel_assignment(self, tmp_path):
        pp_path = self._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')
        unknown_path = self._write_channel_file(tmp_path, 'sample_other.txt')

        project = Project()
        suggestion = project.suggest_polarized_channel_assignment([pp_path, mm_path, unknown_path])

        assert suggestion[str(pp_path)] == PolarizationChannel.PP
        assert suggestion[str(mm_path)] == PolarizationChannel.MM
        assert suggestion[str(unknown_path)] is None

    def test_project_as_dict_and_from_dict_round_trip_a_polarized_experiment(self, tmp_path):
        pp_path = self._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()
        project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})
        original = project.experiments[0]

        # Must not raise (PolarizedDataSet has no .x/.y/.ye of its own) and must
        # be plain-JSON-serializable, since that is what `save_as_json` does with it.
        project_dict = project.as_dict(include_materials_not_in_model=True)
        json.dumps(project_dict)

        global_object.map._clear()
        reloaded_project = Project()
        reloaded_project.from_dict(project_dict)
        reloaded = reloaded_project.experiments[0]

        assert isinstance(reloaded, PolarizedDataSet)
        assert reloaded.name == original.name
        assert reloaded.available_channels == original.available_channels
        assert reloaded.model is reloaded_project.models[0]
        for channel in original.available_channels:
            assert reloaded[channel].name == original[channel].name
            assert reloaded[channel].model is reloaded_project.models[0]
            assert_allclose(reloaded[channel].x, original[channel].x)
            assert_allclose(reloaded[channel].y, original[channel].y)
            assert_allclose(reloaded[channel].ye, original[channel].ye)

    def test_project_save_as_json_and_load_from_json_round_trip_a_polarized_experiment(self, tmp_path):
        pp_path = self._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')

        project = Project()
        project.set_path_project_parent(tmp_path)
        project.calculator = 'refl1d'
        project.default_model()
        project._info['name'] = 'Polarized round trip'
        project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})

        project.save_as_json()
        assert project.path_json.exists()

        global_object.map._clear()
        reloaded_project = Project()
        reloaded_project.load_from_json(project.path_json)
        reloaded = reloaded_project.experiments[0]

        assert isinstance(reloaded, PolarizedDataSet)
        assert reloaded.available_channels == [PolarizationChannel.PP, PolarizationChannel.MM]


class TestChannelAwareExperimentAccessors:
    """`experimental_data_for_model_at_index(index, channel=…)` and friends."""

    @staticmethod
    def _polarized_project(tmp_path) -> Project:
        pp_path = TestLoadPolarizedExperiment._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = TestLoadPolarizedExperiment._write_channel_file(tmp_path, 'sample_dd.txt')
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()
        project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})
        return project

    def test_without_channel_returns_the_whole_experiment(self, tmp_path):
        project = self._polarized_project(tmp_path)

        experiment = project.experimental_data_for_model_at_index(0)

        assert isinstance(experiment, PolarizedDataSet)
        assert project.experiment_is_polarized_at_index(0) is True
        assert project.experiment_channels_at_index(0) == [PolarizationChannel.PP, PolarizationChannel.MM]

    def test_channel_returns_that_channel_dataset(self, tmp_path):
        project = self._polarized_project(tmp_path)
        experiment = project.experiments[0]

        for channel in ('pp', PolarizationChannel.MM):
            data = project.experimental_data_for_model_at_index(0, channel=channel)
            assert isinstance(data, DataSet1D)
            assert data is experiment[channel]

    def test_unmeasured_channel_raises_key_error(self, tmp_path):
        project = self._polarized_project(tmp_path)

        with pytest.raises(KeyError, match='was not measured'):
            project.experimental_data_for_model_at_index(0, channel='pm')

    def test_unknown_channel_raises_value_error(self, tmp_path):
        project = self._polarized_project(tmp_path)

        with pytest.raises(ValueError, match='Unknown spin channel'):
            project.experimental_data_for_model_at_index(0, channel='xx')

    def test_channel_on_unpolarized_experiment_raises_value_error(self, tmp_path):
        path = TestLoadPolarizedExperiment._write_channel_file(tmp_path, 'sample.txt')
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()
        project.load_experiment_for_model_at_index(path, 0)

        assert project.experiment_is_polarized_at_index(0) is False
        assert project.experiment_channels_at_index(0) == []
        with pytest.raises(ValueError, match='not polarized'):
            project.experimental_data_for_model_at_index(0, channel='pp')

    def test_missing_experiment_raises_index_error(self):
        project = Project()
        project.default_model()

        assert project.experiment_is_polarized_at_index(0) is False
        with pytest.raises(IndexError):
            project.experimental_data_for_model_at_index(0, channel='pp')

    def test_model_data_per_channel_differs_for_magnetic_model(self):
        global_object.map._clear()
        model = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=40.0))
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(model)
        q_range = np.linspace(0.01, 0.2, 25)

        pp = project.model_data_for_model_at_index(0, q_range=q_range, channel='pp')
        mm = project.model_data_for_model_at_index(0, q_range=q_range, channel='mm')
        pm = project.model_data_for_model_at_index(0, q_range=q_range, channel='pm')

        assert pp.name.endswith('(pp) for Model 0')
        # Each cross-section is genuinely different — this is what a per-channel
        # display/report must show instead of one curve repeated four times.
        assert not np.allclose(pp.y, mm.y)
        assert not np.allclose(pp.y, pm.y)

    def test_spin_flip_channel_of_non_magnetic_model_raises(self):
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        with pytest.raises(ValueError, match='requires magnetism'):
            project.model_data_for_model_at_index(0, channel='pm')


class TestFitPolarized:
    @pytest.mark.slow
    def test_two_channel_nsf_fit_recovers_rho_m(self):
        truth = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        truth.interface = _refl1d_interface()
        reference = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)

        model = _magnetic_model(LayerMagnetism(rho_m=1.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        rho_m = model.sample[1].layers[0].magnetism.rho_m
        rho_m.fixed = False
        rho_m.min = 0.0
        rho_m.max = 5.0

        data = _polarized_data({'pp': reference['pp'], 'mm': reference['mm']}, model=model)
        fitter = MultiFitter(model)
        results = fitter.fit_polarized(data)

        assert list(results.keys()) == ['pp', 'mm']
        assert all(result.success for result in results.values())
        assert_allclose(rho_m.value, 2.5, atol=0.01)

    @pytest.mark.slow
    def test_four_channel_fit_recovers_rho_m_and_theta_m(self):
        truth = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=45.0))
        truth.interface = _refl1d_interface()
        reference = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)

        model = _magnetic_model(LayerMagnetism(rho_m=1.5, theta_m=60.0))
        model.interface = _refl1d_interface()
        magnetism = model.sample[1].layers[0].magnetism
        magnetism.rho_m.fixed = False
        magnetism.rho_m.min = 0.0
        magnetism.rho_m.max = 5.0
        magnetism.theta_m.fixed = False
        magnetism.theta_m.min = 0.0
        magnetism.theta_m.max = 90.0

        data = _polarized_data(dict(reference), model=model)
        fitter = MultiFitter(model)
        results = fitter.fit_polarized(data)

        assert list(results.keys()) == ['pp', 'pm', 'mp', 'mm']
        assert all(result.success for result in results.values())
        assert_allclose(magnetism.rho_m.value, 2.5, atol=0.02)
        assert_allclose(magnetism.theta_m.value, 45.0, atol=0.5)

    @pytest.mark.slow
    def test_shared_structural_parameter_fitted_across_channels(self):
        truth = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        truth.interface = _refl1d_interface()
        reference = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)

        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        thickness = model.sample[1].layers[0].thickness
        thickness.value = 90.0
        thickness.fixed = False
        thickness.min = 50.0
        thickness.max = 150.0

        data = _polarized_data({'pp': reference['pp'], 'mm': reference['mm']}, model=model)
        fitter = MultiFitter(model)
        results = fitter.fit_polarized(data)

        assert all(result.success for result in results.values())
        assert_allclose(thickness.value, 100.0, atol=0.1)

    def test_fit_polarized_requires_matching_model(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        other = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        other.interface = _refl1d_interface()

        data = _polarized_data({'pp': np.ones_like(Q)}, model=other)
        fitter = MultiFitter(model)
        with pytest.raises(ValueError, match='must be the model'):
            fitter.fit_polarized(data)

    def test_fit_polarized_requires_matching_channel_models(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        other = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        other.interface = _refl1d_interface()

        data = _polarized_data({'pp': np.ones_like(Q), 'mm': np.ones_like(Q)}, model=model)
        # Rebind one channel dataset behind the experiment's back.
        data['mm'].model = other

        fitter = MultiFitter(model)
        with pytest.raises(ValueError, match="'mm' channel dataset"):
            fitter.fit_polarized(data)

    def test_fit_polarized_requires_single_model(self):
        model_a = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        model_b = _magnetic_model(None)
        interface = _refl1d_interface()
        model_a.interface = interface
        model_b.interface = interface

        data = _polarized_data({'pp': np.ones_like(Q)}, model=model_a)
        fitter = MultiFitter(model_a, model_b)
        with pytest.raises(ValueError):
            fitter.fit_polarized(data)


class TestMultiFitterForExperiments:
    """`MultiFitter.for_experiments` — one fit function per dataset, channels expanded."""

    @staticmethod
    def _unpolarized_data(model, name='plain') -> DataSet1D:
        reflectivity = np.exp(-Q * 30)
        dataset = DataSet1D(name=name, x=Q, y=reflectivity, ye=(0.01 * reflectivity) ** 2)
        dataset.model = model
        return dataset

    def test_polarized_experiment_expands_to_one_function_per_channel(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        reference = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)
        data = _polarized_data(dict(reference), model=model)

        fitter = MultiFitter.for_experiments([data])

        assert fitter.fit_channels == [
            PolarizationChannel.PP,
            PolarizationChannel.PM,
            PolarizationChannel.MP,
            PolarizationChannel.MM,
        ]
        assert fitter.fit_datasets == [data[channel] for channel in data.available_channels]
        assert len(fitter._fit_func) == 4
        # Each function evaluates its own cross-section, not four copies of one.
        curves = [func(Q) for func in fitter._fit_func]
        for index, channel in enumerate(data.available_channels):
            assert_allclose(curves[index], reference[channel.value], rtol=1e-9)

    def test_unpolarized_experiment_keeps_one_function(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        data = self._unpolarized_data(model)

        fitter = MultiFitter.for_experiments([data])

        assert fitter.fit_channels == [None]
        assert fitter.fit_datasets == [data]
        assert_allclose(fitter._fit_func[0](Q), model.interface.fit_func(Q, model.unique_name), rtol=1e-9)

    def test_mixed_experiments_share_one_fitter(self):
        """A polarized and an ordinary experiment fitted together, two models."""
        magnetic = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        plain = _magnetic_model(None)
        interface = _refl1d_interface()
        magnetic.interface = interface
        plain.interface = interface
        reference = magnetic.interface.polarized_reflectivity_profiles(Q, magnetic.unique_name)
        polarized = _polarized_data({'pp': reference['pp'], 'mm': reference['mm']}, model=magnetic)
        unpolarized = self._unpolarized_data(plain)

        fitter = MultiFitter.for_experiments([polarized, unpolarized])

        assert fitter.fit_channels == [PolarizationChannel.PP, PolarizationChannel.MM, None]
        # Both models' parameters are enumerated, so they are fitted together.
        assert len(fitter._models) == 2
        assert len(fitter.easy_science_multi_fitter._fit_functions) == 3

    def test_repeated_model_is_registered_once(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        first = self._unpolarized_data(model, name='a')
        second = self._unpolarized_data(model, name='b')

        fitter = MultiFitter.for_experiments([first, second])

        assert len(fitter._models) == 1
        assert len(fitter.fit_datasets) == 2

    def test_experiment_without_model_is_rejected(self):
        dataset = DataSet1D(name='orphan', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q))
        dataset.model = None

        with pytest.raises(ValueError, match='no model'):
            MultiFitter.for_experiments([dataset])

    def test_empty_experiment_list_is_rejected(self):
        with pytest.raises(ValueError, match='At least one experiment'):
            MultiFitter.for_experiments([])

    @pytest.mark.slow
    def test_prepared_fitter_recovers_rho_m_when_run(self):
        """The fitter is usable exactly like `fit_polarized`, but caller-driven."""
        truth = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        truth.interface = _refl1d_interface()
        reference = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)

        model = _magnetic_model(LayerMagnetism(rho_m=1.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        rho_m = model.sample[1].layers[0].magnetism.rho_m
        rho_m.fixed = False
        rho_m.min = 0.0
        rho_m.max = 5.0
        data = _polarized_data({'pp': reference['pp'], 'mm': reference['mm']}, model=model)

        fitter = MultiFitter.for_experiments([data])
        x = [np.asarray(dataset.x) for dataset in fitter.fit_datasets]
        y = [np.asarray(dataset.y) for dataset in fitter.fit_datasets]
        weights = [1.0 / np.sqrt(np.asarray(dataset.ye)) for dataset in fitter.fit_datasets]
        results = fitter.easy_science_multi_fitter.fit(x, y, weights=weights)

        assert all(result.success for result in results)
        assert_allclose(rho_m.value, 2.5, atol=0.01)


class TestRecordFitResults:
    """Results produced by a caller-driven fit can be handed back to the fitter."""

    def _fitter_and_results(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        reflectivity = np.exp(-Q * 30)
        dataset = DataSet1D(name='plain', x=Q, y=reflectivity, ye=(0.01 * reflectivity) ** 2)
        dataset.model = model
        fitter = MultiFitter.for_experiments([dataset])
        results = fitter.easy_science_multi_fitter.fit(
            [np.asarray(dataset.x)], [np.asarray(dataset.y)], weights=[1.0 / np.sqrt(np.asarray(dataset.ye))]
        )
        return fitter, list(results)

    def test_metrics_are_none_before_recording(self):
        fitter, _results = self._fitter_and_results()

        # `easy_science_multi_fitter.fit` bypasses MultiFitter entirely.
        assert fitter.chi2 is None
        assert fitter.reduced_chi is None

    def test_recording_makes_the_metrics_available(self):
        fitter, results = self._fitter_and_results()

        fitter.record_fit_results(results)

        assert fitter.chi2 == pytest.approx(sum(r.chi2 for r in results))
        assert fitter.reduced_chi is not None
        # The classical metrics need the original arrays, which FitResults lacks.
        assert fitter.classical_chi2 is None

    def test_recording_none_clears_the_metrics(self):
        fitter, results = self._fitter_and_results()
        fitter.record_fit_results(results)

        fitter.record_fit_results(None)

        assert fitter.chi2 is None


class TestMagneticSldData:
    """`Project.magnetic_sld_data_for_model_at_index` (Phase 5a)."""

    @staticmethod
    def _magnetic_project(theta_m: float = 270.0) -> Project:
        global_object.map._clear()
        model = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=theta_m))
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(model)
        return project

    def test_profiles_and_spin_potentials(self):
        project = self._magnetic_project()

        profiles = project.magnetic_sld_data_for_model_at_index(0)

        assert set(profiles) == {'sld', 'rho_m', 'theta_m', 'spin_up', 'spin_down'}
        sld, rho_m = profiles['sld'], profiles['rho_m']
        assert sld.x.size == rho_m.x.size and sld.x.size > 0
        assert np.allclose(profiles['spin_up'].x, sld.x)
        # theta_m = 270 == the guide field, so the projection is the full rho_m:
        # spin-up sees rho + rho_m, spin-down rho - rho_m.
        assert_allclose(profiles['spin_up'].y, sld.y + rho_m.y, atol=1e-10)
        assert_allclose(profiles['spin_down'].y, sld.y - rho_m.y, atol=1e-10)
        # The magnetic layer really is magnetic somewhere along z.
        assert np.abs(rho_m.y).max() > 0

    def test_canted_moment_reduces_the_split(self):
        along_field = self._magnetic_project(theta_m=270.0).magnetic_sld_data_for_model_at_index(0)
        canted = self._magnetic_project(theta_m=270.0 - 60.0).magnetic_sld_data_for_model_at_index(0)

        split_along = np.abs(along_field['spin_up'].y - along_field['spin_down'].y).max()
        split_canted = np.abs(canted['spin_up'].y - canted['spin_down'].y).max()

        # cos(60 deg) = 0.5 of the moment is seen by the spin states.
        assert_allclose(split_canted, split_along * 0.5, rtol=1e-6)

    def test_nuclear_profile_matches_the_ordinary_sld_curve(self):
        project = self._magnetic_project()

        profiles = project.magnetic_sld_data_for_model_at_index(0)
        nuclear = project.sld_data_for_model_at_index(0)

        assert_allclose(profiles['sld'].x, nuclear.x)
        assert_allclose(profiles['sld'].y, nuclear.y)

    def test_non_magnetic_model_raises_and_reports_no_magnetism(self):
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        assert project.model_has_magnetism_at_index(0) is False
        assert project.model_has_magnetism_at_index(7) is False
        with pytest.raises(ValueError, match='no magnetic layer'):
            project.magnetic_sld_data_for_model_at_index(0)


class TestSpinAsymmetry:
    """`Project.spin_asymmetry_for_experiment_at_index` (Phase 5b)."""

    @staticmethod
    def _project_with_channels(channels: dict, model=None) -> Project:
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(model if model is not None else _magnetic_model(None))
        data = _polarized_data(channels, model=project.models[0])
        project._experiments[0] = data
        return project

    def test_asymmetry_and_error_propagation(self):
        global_object.map._clear()
        r_pp = np.full_like(Q, 0.6)
        r_mm = np.full_like(Q, 0.2)
        project = self._project_with_channels({'pp': r_pp, 'mm': r_mm})

        result = project.spin_asymmetry_for_experiment_at_index(0)
        measured = result['measured']

        # (0.6 - 0.2) / 0.8
        assert_allclose(measured.y, 0.5)
        # ye holds variances: sigma_SA = 2 sqrt(R--^2 s++^2 + R++^2 s--^2)/(R+++R--)^2
        var_pp = (0.01 * r_pp) ** 2
        var_mm = (0.01 * r_mm) ** 2
        expected = 4.0 * (r_mm**2 * var_pp + r_pp**2 * var_mm) / (r_pp + r_mm) ** 4
        assert_allclose(measured.ye, expected, rtol=1e-12)
        assert result['masked_points'] == 0

    def test_insignificant_points_are_dropped_and_counted(self):
        global_object.map._clear()
        r_pp = np.full_like(Q, 0.6)
        r_mm = np.full_like(Q, 0.2)
        # Make the last five points pure noise: huge uncertainty, tiny signal.
        r_pp[-5:] = 1e-9
        r_mm[-5:] = 1e-9
        noisy = np.where(np.arange(Q.size) >= Q.size - 5, 1.0, 1e-12)
        datasets = {
            'pp': DataSet1D(name='pp', x=Q, y=r_pp, ye=noisy),
            'mm': DataSet1D(name='mm', x=Q, y=r_mm, ye=noisy),
        }
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(_magnetic_model(None))
        project._experiments[0] = PolarizedDataSet(name='noisy', channels=datasets, model=project.models[0])

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['masked_points'] == 5
        assert result['measured'].x.size == Q.size - 5
        assert np.all(np.isfinite(result['measured'].y))

    def test_channels_on_different_q_grids_are_interpolated(self):
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(_magnetic_model(None))
        q_mm = Q + 0.001
        datasets = {
            'pp': DataSet1D(name='pp', x=Q, y=np.full_like(Q, 0.6), ye=np.full_like(Q, 1e-12)),
            'mm': DataSet1D(name='mm', x=q_mm, y=np.full_like(q_mm, 0.2), ye=np.full_like(q_mm, 1e-12)),
        }
        project._experiments[0] = PolarizedDataSet(name='shifted', channels=datasets, model=project.models[0])

        result = project.spin_asymmetry_for_experiment_at_index(0)
        measured = result['measured']

        # SA lives on the pp grid, restricted to where mm has data; the constant
        # mm channel interpolates to 0.2 there.
        assert_allclose(measured.x, Q[Q >= q_mm.min()])
        assert result['out_of_overlap_points'] == int(np.count_nonzero(Q < q_mm.min()))
        assert_allclose(measured.y, 0.5, atol=1e-9)

    def test_calculated_asymmetry_only_for_a_magnetic_model(self):
        global_object.map._clear()
        magnetic = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        magnetic.interface = _refl1d_interface()
        reference = magnetic.interface.polarized_reflectivity_profiles(Q, magnetic.unique_name)
        project = self._project_with_channels({'pp': reference['pp'], 'mm': reference['mm']}, model=magnetic)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        calculated = result['calculated']
        assert calculated is not None
        # Data and model are the same sample, so the two SA curves agree.
        assert_allclose(calculated.y, result['measured'].y, atol=1e-6)
        assert np.abs(calculated.y).max() > 0.01  # a real magnetic signal

    def test_no_calculated_asymmetry_without_magnetism(self):
        global_object.map._clear()
        project = self._project_with_channels({'pp': np.full_like(Q, 0.6), 'mm': np.full_like(Q, 0.2)})

        assert project.spin_asymmetry_for_experiment_at_index(0)['calculated'] is None

    def test_availability_and_errors(self):
        global_object.map._clear()
        project = self._project_with_channels({'pp': np.full_like(Q, 0.6), 'mm': np.full_like(Q, 0.2)})

        assert project.experiment_supports_spin_asymmetry_at_index(0) is True
        assert project.experiment_supports_spin_asymmetry_at_index(1) is False

        # pp only: no asymmetry to form.
        global_object.map._clear()
        nsf_incomplete = self._project_with_channels({'pp': np.full_like(Q, 0.6)})
        assert nsf_incomplete.experiment_supports_spin_asymmetry_at_index(0) is False
        with pytest.raises(ValueError, match='both non-spin-flip channels'):
            nsf_incomplete.spin_asymmetry_for_experiment_at_index(0)
        with pytest.raises(IndexError):
            project.spin_asymmetry_for_experiment_at_index(3)


class TestMagneticProfileSmoothing:
    """CR1 M1: the profile must interpolate the moment as a vector."""

    @staticmethod
    def _two_layer_model(theta_top: float, theta_bottom: float) -> Model:
        vacuum = Material(sld=0, isld=0, name='Vacuum')
        iron = Material(sld=8.0, isld=0, name='Fe')
        si = Material(sld=2.047, isld=0, name='Si')
        superphase = Layer(material=vacuum, thickness=0, roughness=0, name='Vacuum Superphase')
        top = Layer(
            material=iron,
            thickness=100,
            roughness=5,
            magnetism=LayerMagnetism(rho_m=5.0, theta_m=theta_top),
            name='Fe top',
        )
        bottom = Layer(
            material=iron,
            thickness=100,
            roughness=5,
            magnetism=LayerMagnetism(rho_m=5.0, theta_m=theta_bottom),
            name='Fe bottom',
        )
        subphase = Layer(material=si, thickness=0, roughness=5, name='Si Subphase')
        sample = Sample(Multilayer(superphase), Multilayer(top), Multilayer(bottom), Multilayer(subphase), name='Sample')
        model = Model(sample=sample, scale=1, background=0, name='Two-layer magnetic')
        model.resolution_function = PercentageFwhm(0)
        return model

    def _profiles(self, theta_top: float, theta_bottom: float) -> dict:
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(self._two_layer_model(theta_top, theta_bottom))
        return project.magnetic_sld_data_for_model_at_index(0)

    def test_interface_between_almost_antiparallel_angles_does_not_invent_splitting(self):
        # 359 deg and 1 deg are 2 deg apart, but smoothing the *angle* takes the
        # long way round through 180 deg — and through the guide-field direction
        # at 270 deg, where the full moment would look longitudinal.
        profiles = self._profiles(359.0, 1.0)

        splitting = np.abs(profiles['spin_up'].y - profiles['spin_down'].y)

        # 2 * 5.0 * |cos(89 deg)| ~ 0.18, not 2 * 5.0.
        assert splitting.max() < 0.5

    def test_collinear_layers_keep_the_full_splitting(self):
        # The same geometry with both moments along the guide field must still
        # show the whole moment: the fix must not damp real magnetism.
        profiles = self._profiles(270.0, 270.0)

        splitting = np.abs(profiles['spin_up'].y - profiles['spin_down'].y)

        assert splitting.max() == pytest.approx(2 * 5.0, rel=1e-3)

    def test_magnitude_and_angle_round_trip(self):
        profiles = self._profiles(210.0, 210.0)

        rho_m = profiles['rho_m']
        theta_m = profiles['theta_m']

        assert rho_m.y.max() == pytest.approx(5.0, rel=1e-3)
        # theta_m is reported only where there is a moment.
        assert theta_m.x.size < rho_m.x.size
        assert np.allclose(theta_m.y, 210.0, atol=1e-6)

    def test_angle_is_not_reported_through_non_magnetic_regions(self):
        profiles = self._profiles(270.0, 270.0)

        theta_m = profiles['theta_m']
        rho_m = profiles['rho_m']

        # Every reported angle sits at a depth that carries a moment.
        carried = np.interp(theta_m.x, rho_m.x, rho_m.y)
        assert np.all(np.abs(carried) > 0)
        assert theta_m.x.size > 0


class TestSpinAsymmetryGridPairing:
    """CR1 M2: pair channels only where both were measured."""

    @staticmethod
    def _project_with(pp: DataSet1D, mm: DataSet1D) -> Project:
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(_magnetic_model(None))
        project._experiments[0] = PolarizedDataSet(name='pairing', channels={'pp': pp, 'mm': mm}, model=project.models[0])
        return project

    def test_points_outside_the_mm_range_are_dropped_not_extrapolated(self):
        # pp reaches further in q than mm; np.interp would clamp to the mm edge
        # value and present the result as measured data.
        q_pp = np.linspace(0.01, 0.30, 30)
        q_mm = np.linspace(0.01, 0.20, 20)
        pp = DataSet1D(name='pp', x=q_pp, y=np.full_like(q_pp, 0.6), ye=np.full_like(q_pp, 1e-12))
        mm = DataSet1D(name='mm', x=q_mm, y=np.full_like(q_mm, 0.2), ye=np.full_like(q_mm, 1e-12))
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['out_of_overlap_points'] == int(np.count_nonzero(q_pp > 0.20))
        assert result['measured'].x.max() <= 0.20
        assert np.allclose(result['measured'].y, 0.5)

    def test_disjoint_grids_give_no_asymmetry(self):
        q_pp = np.linspace(0.30, 0.40, 10)
        q_mm = np.linspace(0.01, 0.20, 10)
        pp = DataSet1D(name='pp', x=q_pp, y=np.full_like(q_pp, 0.6), ye=np.full_like(q_pp, 1e-12))
        mm = DataSet1D(name='mm', x=q_mm, y=np.full_like(q_mm, 0.2), ye=np.full_like(q_mm, 1e-12))
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['out_of_overlap_points'] == q_pp.size
        assert result['measured'].x.size == 0
        assert result['calculated'] is None

    def test_a_float_round_trip_difference_still_counts_as_the_same_grid(self):
        q = np.linspace(0.01, 0.2, 20)
        # The kind of difference a text round-trip introduces.
        q_mm = np.array([float(f'{value:.12g}') for value in q * (1 + 1e-12)])
        pp = DataSet1D(name='pp', x=q, y=np.full_like(q, 0.6), ye=np.full_like(q, 1e-12))
        mm = DataSet1D(name='mm', x=q_mm, y=np.full_like(q_mm, 0.2), ye=np.full_like(q_mm, 1e-12))
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['out_of_overlap_points'] == 0
        assert result['measured'].x.size == q.size

    def test_equal_grids_report_no_dropped_points(self):
        q = np.linspace(0.01, 0.2, 20)
        pp = DataSet1D(name='pp', x=q, y=np.full_like(q, 0.6), ye=np.full_like(q, 1e-12))
        mm = DataSet1D(name='mm', x=q, y=np.full_like(q, 0.2), ye=np.full_like(q, 1e-12))
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['out_of_overlap_points'] == 0
        assert result['masked_points'] == 0


class TestSpinAsymmetryDenominatorGuards:
    """CR2: the guard must not depend on the data carrying uncertainties."""

    @staticmethod
    def _project_with(pp: DataSet1D, mm: DataSet1D) -> Project:
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(_magnetic_model(None))
        project._experiments[0] = PolarizedDataSet(name='guards', channels={'pp': pp, 'mm': mm}, model=project.models[0])
        return project

    def test_cancellation_without_uncertainties_is_dropped(self):
        # Background-subtracted tail: the two channels nearly cancel, so SA is
        # a ratio of rounding noise (here it would be 1999).
        q = np.linspace(0.01, 0.2, 5)
        pp = DataSet1D(name='pp', x=q, y=np.full_like(q, 1e-300))
        mm = DataSet1D(name='mm', x=q, y=np.full_like(q, -0.999e-300))
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['measured'].x.size == 0
        assert result['small_denominator_points'] == q.size
        assert result['masked_points'] == q.size

    def test_ordinary_data_without_uncertainties_is_kept(self):
        # A two-column file must still produce its asymmetry.
        q = np.linspace(0.01, 0.2, 5)
        pp = DataSet1D(name='pp', x=q, y=np.full_like(q, 0.6))
        mm = DataSet1D(name='mm', x=q, y=np.full_like(q, 0.2))
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['measured'].x.size == q.size
        assert_allclose(result['measured'].y, 0.5)
        assert result['masked_points'] == 0

    def test_exact_cancellation_is_dropped(self):
        q = np.linspace(0.01, 0.2, 4)
        pp = DataSet1D(name='pp', x=q, y=np.full_like(q, 0.5))
        mm = DataSet1D(name='mm', x=q, y=np.full_like(q, -0.5))
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['measured'].x.size == 0
        assert result['small_denominator_points'] == q.size

    def test_unusable_variances_do_not_silently_become_no_uncertainty(self):
        q = np.linspace(0.01, 0.2, 4)
        variance = np.array([1e-12, -1.0, np.nan, 1e-12])
        pp = DataSet1D(name='pp', x=q, y=np.full_like(q, 0.6), ye=variance)
        mm = DataSet1D(name='mm', x=q, y=np.full_like(q, 0.2), ye=np.full_like(q, 1e-12))
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        # The two malformed points are dropped and counted; the rest survive.
        assert result['invalid_points'] == 2
        assert result['measured'].x.size == 2
        assert np.all(result['measured'].ye >= 0)

    def test_low_significance_and_cancellation_are_counted_separately(self):
        q = np.linspace(0.01, 0.2, 6)
        # First three: small but clean signal drowned in uncertainty.
        # Last three: clean cancellation with negligible uncertainty.
        r_pp = np.array([1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0])
        r_mm = np.array([1e-6, 1e-6, 1e-6, -0.9999, -0.9999, -0.9999])
        variance = np.array([1.0, 1.0, 1.0, 1e-24, 1e-24, 1e-24])
        pp = DataSet1D(name='pp', x=q, y=r_pp, ye=variance)
        mm = DataSet1D(name='mm', x=q, y=r_mm, ye=variance)
        project = self._project_with(pp, mm)

        result = project.spin_asymmetry_for_experiment_at_index(0)

        assert result['low_significance_points'] == 3
        assert result['small_denominator_points'] == 3
        assert result['masked_points'] == 6


class TestSpinAsymmetryChannelValidation:
    """CR2: do not advertise a spin asymmetry that cannot be computed."""

    @staticmethod
    def _project_with(pp: DataSet1D, mm: DataSet1D) -> Project:
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(_magnetic_model(None))
        project._experiments[0] = PolarizedDataSet(name='validation', channels={'pp': pp, 'mm': mm}, model=project.models[0])
        return project

    def test_empty_channel_is_not_advertised(self):
        q = np.linspace(0.01, 0.2, 5)
        project = self._project_with(
            DataSet1D(name='pp', x=np.array([]), y=np.array([])),
            DataSet1D(name='mm', x=q, y=np.full_like(q, 0.2)),
        )

        assert project.experiment_supports_spin_asymmetry_at_index(0) is False

    def test_duplicate_q_is_not_advertised(self):
        q = np.array([0.01, 0.02, 0.02, 0.03])
        project = self._project_with(
            DataSet1D(name='pp', x=q, y=np.full_like(q, 0.6)),
            DataSet1D(name='mm', x=q, y=np.full_like(q, 0.2)),
        )

        assert project.experiment_supports_spin_asymmetry_at_index(0) is False

    def test_non_finite_q_is_not_advertised(self):
        q = np.array([0.01, 0.02, np.nan, 0.03])
        project = self._project_with(
            DataSet1D(name='pp', x=q, y=np.full_like(q, 0.6)),
            DataSet1D(name='mm', x=q, y=np.full_like(q, 0.2)),
        )

        assert project.experiment_supports_spin_asymmetry_at_index(0) is False

    def test_descending_grid_is_sorted_before_pairing(self):
        # np.interp needs an increasing grid and returns nonsense otherwise.
        q = np.linspace(0.01, 0.2, 6)
        descending = q[::-1]
        project = self._project_with(
            DataSet1D(name='pp', x=descending, y=np.full_like(q, 0.6)),
            DataSet1D(name='mm', x=q + 0.0005, y=np.linspace(0.2, 0.3, 6)),
        )

        assert project.experiment_supports_spin_asymmetry_at_index(0) is True
        measured = project.spin_asymmetry_for_experiment_at_index(0)['measured']

        assert np.all(np.diff(measured.x) > 0)
        assert np.all(np.isfinite(measured.y))


class TestSpinAsymmetryInterpolatedVariance:
    """CR2: variances interpolate with squared weights."""

    def test_midpoint_variance_uses_squared_weights(self):
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(_magnetic_model(None))
        # mm sampled at 0.10 and 0.20 with variances 1 and 9; pp asks for 0.15.
        pp = DataSet1D(name='pp', x=np.array([0.15]), y=np.array([1.0]), ye=np.array([0.0]))
        mm = DataSet1D(name='mm', x=np.array([0.10, 0.20]), y=np.array([0.5, 0.5]), ye=np.array([0.01, 0.09]))
        project._experiments[0] = PolarizedDataSet(name='interp', channels={'pp': pp, 'mm': mm}, model=project.models[0])

        result = project.spin_asymmetry_for_experiment_at_index(0)

        # var_mm at the midpoint: 0.25*0.01 + 0.25*0.09 = 0.025 (the linear rule
        # would give 0.05).
        r_pp, r_mm = 1.0, 0.5
        expected = 4.0 * (r_pp**2 * 0.025) / (r_pp + r_mm) ** 4
        assert_allclose(result['measured'].ye, [expected], rtol=1e-12)

    def test_endpoints_take_the_source_variance_unchanged(self):
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(_magnetic_model(None))
        pp = DataSet1D(name='pp', x=np.array([0.10, 0.20]), y=np.array([1.0, 1.0]), ye=np.array([0.0, 0.0]))
        mm = DataSet1D(name='mm', x=np.array([0.10, 0.15, 0.20]), y=np.array([0.5, 0.5, 0.5]), ye=np.array([0.01, 0.04, 0.09]))
        project._experiments[0] = PolarizedDataSet(name='endpoints', channels={'pp': pp, 'mm': mm}, model=project.models[0])

        result = project.spin_asymmetry_for_experiment_at_index(0)

        r_pp, r_mm = 1.0, 0.5
        expected = 4.0 * (r_pp**2 * np.array([0.01, 0.09])) / (r_pp + r_mm) ** 4
        assert_allclose(result['measured'].ye, expected, rtol=1e-12)


class TestMagneticProfileDisplayContinuity:
    """CR2: a periodic angle must not be drawn as a full sweep."""

    def test_angle_is_continuous_across_the_zero_boundary(self):
        profiles = TestMagneticProfileSmoothing()._profiles(359.0, 1.0)

        theta = profiles['theta_m'].y

        # 359 -> 361 rather than 359 -> 1: no 358-degree jump anywhere.
        assert np.abs(np.diff(theta)).max() < 10.0
        assert theta.max() - theta.min() < 10.0

    def test_component_smoothing_failure_is_not_papered_over(self):
        # The wrapper must refuse rather than serve refl1d's angle-smoothed
        # profile, which misreports the splitting at such an interface.
        global_object.map._clear()
        model = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(model)
        calculator = project.models[0].interface()

        with patch.object(
            type(calculator._wrapper), '_smoothed_magnetic_vector', side_effect=NotImplementedError('no microslabs')
        ):
            with pytest.raises(NotImplementedError):
                project.magnetic_sld_data_for_model_at_index(0)


class TestCalculatorCapabilities:
    """Helpers an application needs to offer the engine magnetism requires."""

    def test_reports_which_calculators_support_magnetism(self):
        project = Project()

        supporting = project.calculators_supporting_magnetism

        assert supporting == ['refl1d']
        # Asking must not change the active calculator.
        assert project.calculator == 'refnx'

    def test_models_have_magnetism_follows_the_sample(self):
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        assert project.models_have_magnetism is False

        project.models[0].sample[1].layers[0].magnetism = LayerMagnetism(rho_m=2.0)
        assert project.models_have_magnetism is True

        project.models[0].sample[1].layers[0].magnetism = None
        assert project.models_have_magnetism is False
