# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution functions follow the model and the dataset, not the calculator.

Three things are pinned here:

* one calculator serves every model of a project, and each model keeps its
  own resolution on it (a second model's ``Pointwise`` must not leak into the
  first model's curve);
* a dataset carries the resolution it was measured with, and the fit smears
  each dataset -- each spin channel of a polarized experiment included --
  with that resolution rather than with whatever the model holds;
* several measured curves of one contrast can be merged into one experiment
  whose every point keeps its own width, including overlapping q ranges.
"""

import os
import warnings
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from easyscience import Minimizers
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.data import load_as_dataset
from easyreflectometry.data import merge_datasets
from easyreflectometry.data import resolution_from_dataset
from easyreflectometry.fit_settings import FitSettings
from easyreflectometry.fitting import FitScopeError
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

PATH_STATIC = os.path.join(os.path.dirname(__file__), '_static')

Q = np.linspace(0.01, 0.3, 40)


@pytest.fixture(autouse=True)
def _isolated_global_object():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def _film_model(name: str = 'Film', magnetism: LayerMagnetism | None = None, interface=None) -> Model:
    """Ambient | 100 A film | Si, with the fringes resolution smearing acts on."""
    vacuum = Material(sld=0, isld=0, name=f'Vacuum {name}')
    film = Material(sld=4.0, isld=0, name=f'Film material {name}')
    si = Material(sld=2.07, isld=0, name=f'Si {name}')
    superphase = Layer(material=vacuum, thickness=0, roughness=0, name=f'Superphase {name}')
    layer = Layer(material=film, thickness=100, roughness=3, magnetism=magnetism, name=f'Layer {name}')
    subphase = Layer(material=si, thickness=0, roughness=3, name=f'Subphase {name}')
    sample = Sample(Multilayer(superphase), Multilayer(layer), Multilayer(subphase), name=f'Sample {name}')
    model = Model(sample=sample, scale=1, background=0, name=name, interface=interface)
    return model


def _pointwise(q: np.ndarray, relative_sigma: float) -> Pointwise:
    return Pointwise([q, np.ones_like(q), (relative_sigma * q) ** 2])


def _sharp() -> Pointwise:
    return _pointwise(Q, 0.002)


def _broad() -> Pointwise:
    return _pointwise(Q, 0.08)


def _write_four_column_file(path, q, r, sigma_r, sigma_q):
    np.savetxt(path, np.column_stack([q, r, sigma_r, sigma_q]), header='Qz R sR sQz')


# ----------------------------------------------------------------------------
# Per-model resolution on a shared calculator
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('engine', ['refnx', 'refl1d'])
class TestPerModelResolution:
    def test_wrapper_keeps_one_resolution_per_model(self, engine):
        interface = CalculatorFactory()
        interface.switch(engine)
        first = _film_model('first', interface=interface)
        second = _film_model('second', interface=interface)
        first.resolution_function = _sharp()
        second.resolution_function = _broad()

        wrapper = interface()._wrapper
        assert wrapper.resolution_function_for(first.unique_name) is first.resolution_function
        assert wrapper.resolution_function_for(second.unique_name) is second.resolution_function
        assert isinstance(wrapper.resolution_function_for('no such model'), PercentageFwhm)

    def test_second_model_does_not_change_first_models_curve(self, engine):
        interface = CalculatorFactory()
        interface.switch(engine)
        first = _film_model('first', interface=interface)
        first.resolution_function = _sharp()
        reference = interface.fit_func(Q, first.unique_name)

        second = _film_model('second', interface=interface)
        second.resolution_function = _broad()

        assert_allclose(interface.fit_func(Q, first.unique_name), reference, rtol=1e-12)
        # Sanity: the broad resolution does make a different curve.
        assert not np.allclose(interface.fit_func(Q, second.unique_name), reference, rtol=1e-3)

    def test_explicit_resolution_overrides_the_registered_one(self, engine):
        interface = CalculatorFactory()
        interface.switch(engine)
        model = _film_model('model', interface=interface)
        model.resolution_function = _sharp()
        calculator = interface()

        with_broad = calculator.reflectity_profile(Q, model.unique_name, resolution_function=_broad())
        model.resolution_function = _broad()
        assert_allclose(with_broad, calculator.reflectity_profile(Q, model.unique_name), rtol=1e-12)

    def test_default_resolution_serves_models_without_their_own(self, engine):
        interface = CalculatorFactory()
        interface.switch(engine)
        model = _film_model('model', interface=interface)
        wrapper = interface()._wrapper
        wrapper.set_resolution_function(_broad())
        # The model registered its own (the Model default) on assignment, so
        # the default does not apply to it ...
        assert wrapper.resolution_function_for(model.unique_name) is model.resolution_function
        # ... only to names that never registered one.
        assert wrapper.resolution_function_for(None) is wrapper.resolution_function_for('unregistered')


def test_calculator_switch_keeps_each_models_resolution():
    project = Project()
    project.models = ModelCollection(_film_model('first'), _film_model('second'))
    project.models[0].resolution_function = _sharp()
    project.models[1].resolution_function = _broad()

    project.calculator = 'refl1d'

    wrapper = project._calculator()._wrapper
    assert wrapper.resolution_function_for(project.models[0].unique_name) is project.models[0].resolution_function
    assert wrapper.resolution_function_for(project.models[1].unique_name) is project.models[1].resolution_function


# ----------------------------------------------------------------------------
# Per-dataset resolution in the fit
# ----------------------------------------------------------------------------


class TestDatasetResolutionInFit:
    def test_dataset_without_resolution_uses_the_models(self):
        interface = CalculatorFactory()
        model = _film_model('model', interface=interface)
        model.resolution_function = _broad()
        dataset = DataSet1D(name='plain', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q), model=model, auto_background=False)
        assert dataset.resolution_function is None

        fitter = MultiFitter.for_experiments([dataset])

        assert_allclose(fitter._fit_func[0](Q), interface.fit_func(Q, model.unique_name), rtol=1e-12)

    def test_each_dataset_of_one_model_is_smeared_with_its_own_resolution(self):
        interface = CalculatorFactory()
        model = _film_model('model', interface=interface)
        model.resolution_function = PercentageFwhm(5)
        sharp = DataSet1D(name='sharp', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q), model=model, auto_background=False)
        broad = DataSet1D(name='broad', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q), model=model, auto_background=False)
        sharp.resolution_function = _sharp()
        broad.resolution_function = _broad()

        fitter = MultiFitter.for_experiments([sharp, broad])

        calculator = interface()
        assert_allclose(
            fitter._fit_func[0](Q), calculator.reflectity_profile(Q, model.unique_name, resolution_function=_sharp())
        )
        assert_allclose(
            fitter._fit_func[1](Q), calculator.reflectity_profile(Q, model.unique_name, resolution_function=_broad())
        )
        assert not np.allclose(fitter._fit_func[0](Q), fitter._fit_func[1](Q), rtol=1e-3)
        # The model's own resolution is untouched by the fit setup.
        assert isinstance(model.resolution_function, PercentageFwhm)

    def test_polarized_channels_each_use_their_own_resolution(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        model = _film_model('magnetic', magnetism=LayerMagnetism(rho_m=1.5, theta_m=270.0), interface=interface)
        model.resolution_function = PercentageFwhm(5)
        calculator = interface()
        channels = {
            'pp': DataSet1D(name='pp', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q)),
            'mm': DataSet1D(name='mm', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q)),
        }
        channels['pp'].resolution_function = _sharp()
        channels['mm'].resolution_function = _broad()
        data = PolarizedDataSet(name='polarized', channels=channels, model=model)

        fitter = MultiFitter.for_experiments([data])

        expected_pp = calculator.reflectivity_profile_channel(Q, model.unique_name, 'pp', resolution_function=_sharp())
        expected_mm = calculator.reflectivity_profile_channel(Q, model.unique_name, 'mm', resolution_function=_broad())
        assert fitter.fit_channels == [PolarizationChannel.PP, PolarizationChannel.MM]
        assert_allclose(fitter._fit_func[0](Q), expected_pp, rtol=1e-12)
        assert_allclose(fitter._fit_func[1](Q), expected_mm, rtol=1e-12)
        # Not the same smearing for both channels.
        model_level_mm = calculator.reflectivity_profile_channel(Q, model.unique_name, 'mm', resolution_function=_sharp())
        assert not np.allclose(expected_mm, model_level_mm, rtol=1e-3)

    @pytest.mark.slow
    def test_fit_polarized_uses_the_channel_resolutions(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        truth = _film_model('truth', magnetism=LayerMagnetism(rho_m=2.5, theta_m=270.0), interface=interface)
        calculator = interface()
        reference = {
            'pp': calculator.reflectivity_profile_channel(Q, truth.unique_name, 'pp', resolution_function=_sharp()),
            'mm': calculator.reflectivity_profile_channel(Q, truth.unique_name, 'mm', resolution_function=_broad()),
        }

        model = _film_model('model', magnetism=LayerMagnetism(rho_m=1.0, theta_m=270.0), interface=interface)
        model.resolution_function = PercentageFwhm(5)
        rho_m = model.sample[1].layers[0].magnetism.rho_m
        rho_m.fixed = False
        rho_m.min = 0.0
        rho_m.max = 5.0
        channels = {
            channel: DataSet1D(name=channel, x=Q, y=curve, ye=(0.01 * curve) ** 2) for channel, curve in reference.items()
        }
        channels['pp'].resolution_function = _sharp()
        channels['mm'].resolution_function = _broad()
        data = PolarizedDataSet(name='polarized', channels=channels, model=model)

        results = MultiFitter(model).fit_polarized(data)

        assert all(result.success for result in results.values())
        assert_allclose(rho_m.value, 2.5, atol=0.02)

    @pytest.mark.slow
    def test_fit_single_data_set_1d_uses_the_dataset_resolution(self):
        interface = CalculatorFactory()
        truth = _film_model('truth', interface=interface)
        calculator = interface()
        curve = 0.7 * calculator.reflectity_profile(Q, truth.unique_name, resolution_function=_broad())

        model = _film_model('model', interface=interface)
        model.resolution_function = _sharp()
        model.scale.fixed = False
        model.scale.min = 0.1
        model.scale.max = 2.0
        dataset = DataSet1D(name='broad', x=Q, y=curve, ye=(0.01 * curve) ** 2, model=model, auto_background=False)
        dataset.resolution_function = _broad()

        result = MultiFitter(model).fit_single_data_set_1d(dataset)

        assert result.success
        assert_allclose(model.scale.value, 0.7, atol=0.01)
        # The model is still registered with its own (sharp) resolution.
        assert model.resolution_function.smearing(Q)[-1] == pytest.approx(0.002 * Q[-1])


# ----------------------------------------------------------------------------
# Pointwise at the data points, merging curves
# ----------------------------------------------------------------------------


class TestPointwiseAtDataPoints:
    def test_exact_widths_at_the_stored_points_even_when_q_repeats(self):
        q = np.array([0.01, 0.02, 0.02, 0.03])
        sigma = np.array([1e-4, 2e-4, 5e-4, 3e-4])
        resolution = Pointwise([q, np.ones_like(q), sigma**2])

        assert_allclose(resolution.smearing(q), sigma)
        assert_allclose(resolution.smearing(), sigma)

    def test_interpolation_copes_with_unsorted_points(self):
        q = np.array([0.03, 0.01, 0.02])
        sigma = np.array([3e-4, 1e-4, 2e-4])
        resolution = Pointwise([q, np.ones_like(q), sigma**2])

        assert_allclose(resolution.smearing([0.015, 0.025]), [1.5e-4, 2.5e-4])


class TestMergeDatasets:
    @staticmethod
    def _dataset(name, q, relative_sigma=None):
        """A measured curve as the file loaders return it: resolution set from xe."""
        r = np.exp(-30 * q)
        xe = None if relative_sigma is None else (relative_sigma * q) ** 2
        dataset = DataSet1D(name=name, x=q, y=r, ye=(0.05 * r) ** 2, xe=xe)
        dataset.resolution_function = resolution_from_dataset(dataset)
        return dataset

    def test_points_are_concatenated_sorted_and_keep_their_own_widths(self):
        low = self._dataset('low', np.linspace(0.01, 0.1, 10), 0.01)
        high = self._dataset('high', np.linspace(0.08, 0.3, 12), 0.03)

        merged = merge_datasets([low, high])

        assert merged.name == 'low + high'
        assert len(merged.x) == 22
        assert np.all(np.diff(merged.x) >= 0)
        assert isinstance(merged.resolution_function, Pointwise)
        # Every point keeps the width it came with, overlap included.
        expected = {}
        for dataset in (low, high):
            for q, xe in zip(dataset.x, dataset.xe):
                expected.setdefault(float(q), []).append(np.sqrt(xe))
        widths = merged.resolution_function.smearing(merged.x)
        for q, width in zip(merged.x, widths):
            assert any(np.isclose(width, candidate) for candidate in expected[float(q)])

    def test_overlapping_q_keeps_both_widths(self):
        q = np.array([0.01, 0.02])
        first = DataSet1D(name='a', x=q, y=[1.0, 1.0], ye=[1.0, 1.0], xe=[1e-8, 1e-8])
        second = DataSet1D(name='b', x=q, y=[1.0, 1.0], ye=[1.0, 1.0], xe=[4e-8, 4e-8])
        for dataset in (first, second):
            dataset.resolution_function = resolution_from_dataset(dataset)

        merged = merge_datasets([first, second])

        assert_allclose(merged.x, [0.01, 0.01, 0.02, 0.02])
        assert_allclose(merged.resolution_function.smearing(merged.x), [1e-4, 2e-4, 1e-4, 2e-4])

    def test_explicitly_assigned_resolutions_survive_the_merge(self):
        q = np.array([0.01, 0.02])
        first = DataSet1D(name='a', x=q, y=[1.0, 1.0], ye=[1.0, 1.0], resolution_function=PercentageFwhm(1))
        # Explicit resolution wins over the measured xe column.
        second = DataSet1D(name='b', x=q, y=[1.0, 1.0], ye=[1.0, 1.0], xe=[1e-12, 1e-12])
        second.resolution_function = PercentageFwhm(20)

        merged = merge_datasets([first, second])

        expected = np.array([0.01 * 0.01, 0.20 * 0.01, 0.01 * 0.02, 0.20 * 0.02]) / SIGMA_TO_FWHM
        assert isinstance(merged.resolution_function, Pointwise)
        assert_allclose(merged.resolution_function.smearing(merged.x), expected)
        assert_allclose(merged.xe, expected**2)

    def test_merged_curve_matches_the_unmerged_curves(self):
        interface = CalculatorFactory()
        model = _film_model('model', interface=interface)
        calculator = interface()
        sharp = DataSet1D(name='sharp', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q), resolution_function=_sharp())
        broad = DataSet1D(name='broad', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q), resolution_function=_broad())

        merged = merge_datasets([sharp, broad])
        merged.model = model

        expected = np.empty(2 * len(Q))
        expected[0::2] = calculator.reflectity_profile(Q, model.unique_name, resolution_function=_sharp())
        expected[1::2] = calculator.reflectity_profile(Q, model.unique_name, resolution_function=_broad())
        assert_allclose(MultiFitter.for_experiments([merged])._fit_func[0](merged.x), expected, rtol=1e-10)

    def test_dataset_using_the_models_resolution_is_filled_not_read_from_xe(self):
        measured = self._dataset('measured', np.linspace(0.01, 0.1, 5), 0.01)
        overridden = self._dataset('overridden', np.linspace(0.2, 0.3, 5), 0.01)
        overridden.resolution_function = None

        with pytest.warns(UserWarning, match="\\['overridden'\\]"):
            merged = merge_datasets([measured, overridden], fill_resolution=PercentageFwhm(2.0))

        assert_allclose(merged.resolution_function.smearing(merged.x)[5:], 0.02 * overridden.x / SIGMA_TO_FWHM)

    def test_mixed_resolution_fills_missing_points_and_warns(self):
        with_resolution = self._dataset('measured', np.linspace(0.01, 0.1, 5), 0.01)
        without = self._dataset('bare', np.linspace(0.2, 0.3, 5))

        with pytest.warns(UserWarning, match="\\['bare'\\]"):
            merged = merge_datasets([with_resolution, without], fill_resolution=PercentageFwhm(2.0))

        widths = merged.resolution_function.smearing(merged.x)
        assert_allclose(widths[:5], 0.01 * with_resolution.x)
        assert_allclose(widths[5:], 0.02 * without.x / SIGMA_TO_FWHM)

    def test_mixed_resolution_defaults_to_the_five_percent_fill(self):
        with_resolution = self._dataset('measured', np.linspace(0.01, 0.1, 5), 0.01)
        without = self._dataset('bare', np.linspace(0.2, 0.3, 5))

        with pytest.warns(UserWarning):
            merged = merge_datasets([with_resolution, without])

        assert_allclose(merged.resolution_function.smearing(merged.x)[5:], 0.05 * without.x / SIGMA_TO_FWHM)

    def test_no_resolution_anywhere_leaves_the_model_in_charge(self):
        first = self._dataset('a', np.linspace(0.01, 0.1, 5))
        second = self._dataset('b', np.linspace(0.2, 0.3, 5))

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            merged = merge_datasets([first, second], name='contrast')

        assert merged.name == 'contrast'
        assert merged.resolution_function is None
        assert_allclose(merged.xe, 0.0)

    def test_model_and_header_follow_the_first_dataset(self):
        model = _film_model('model')
        first = self._dataset('a', np.linspace(0.01, 0.1, 5), 0.01)
        first.model = model
        first.orso_header = {'data_source': {}}
        second = self._dataset('b', np.linspace(0.2, 0.3, 5), 0.01)

        merged = merge_datasets([first, second])

        assert merged.model is model
        assert merged.orso_header == {'data_source': {}}

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError, match='At least one dataset'):
            merge_datasets([])


class TestMaskedFitKeepsPointWidths:
    """`legacy_mask` drops points; a merged dataset's repeated q must keep their own widths."""

    @staticmethod
    def _merged_truth(model_name='truth', channel=None, magnetism=None, engine='refnx'):
        interface = CalculatorFactory()
        interface.switch(engine)
        truth = _film_model(model_name, magnetism=magnetism, interface=interface)
        calculator = interface()
        sharp, broad = _sharp(), _broad()
        curves = []
        for resolution in (sharp, broad):
            if channel is None:
                curves.append(calculator.reflectity_profile(Q, truth.unique_name, resolution_function=resolution))
            else:
                curves.append(
                    calculator.reflectivity_profile_channel(Q, truth.unique_name, channel, resolution_function=resolution)
                )
        datasets = [
            DataSet1D(name=str(i), x=Q, y=curve, ye=(0.01 * curve) ** 2, resolution_function=resolution)
            for i, (curve, resolution) in enumerate(zip(curves, (sharp, broad)))
        ]
        merged = merge_datasets(datasets)
        # Dropping any one point leaves the other repeated-q pairs to be fitted
        # with their own widths, not with q-interpolated ones.
        merged.ye[0] = 0.0
        # `truth` is released here; the model fitted next may be given its name
        # (tests/calculators/test_recycled_names.py pins that this is harmless).
        return interface, merged

    def test_resolution_for_fitted_points_keeps_identities(self):
        from easyreflectometry.fitting import _resolution_for_fitted_points

        q = np.array([0.01, 0.02, 0.02, 0.03])
        sigma = np.array([1e-4, 2e-4, 5e-4, 3e-4])
        resolution = Pointwise([q, np.ones_like(q), sigma**2])
        variances = np.array([0.0, 1.0, 1.0, 1.0])

        subset = _resolution_for_fitted_points(resolution, q, np.ones_like(q), variances, 'legacy_mask')

        assert_allclose(subset.smearing(q[1:]), sigma[1:])
        # Nothing dropped, or another objective: the resolution is used as it is.
        assert _resolution_for_fitted_points(resolution, q, np.ones_like(q), variances, 'hybrid') is resolution
        assert _resolution_for_fitted_points(resolution, q, np.ones_like(q), np.ones_like(q), 'legacy_mask') is resolution

    @pytest.mark.slow
    def test_single_dataset_masked_fit_recovers_scale(self):
        interface, merged = self._merged_truth()
        model = _film_model('model', interface=interface)
        model.scale.value = 0.5
        model.scale.fixed = False
        model.scale.min = 0.1
        model.scale.max = 2.0
        merged.model = model

        with pytest.warns(UserWarning, match='Masked 1'):
            result = MultiFitter(model).fit_single_data_set_1d(merged, objective='legacy_mask')

        assert result.success
        assert_allclose(model.scale.value, 1.0, atol=1e-4)

    @pytest.mark.slow
    def test_polarized_masked_fit_recovers_scale(self):
        magnetism = LayerMagnetism(rho_m=1.5, theta_m=270.0)
        interface, merged = self._merged_truth(channel='pp', magnetism=magnetism, engine='refl1d')
        model = _film_model('model', magnetism=LayerMagnetism(rho_m=1.5, theta_m=270.0), interface=interface)
        model.scale.value = 0.5
        model.scale.fixed = False
        model.scale.min = 0.1
        model.scale.max = 2.0
        data = PolarizedDataSet(name='polarized', channels={'pp': merged}, model=model)

        with pytest.warns(UserWarning, match='Masked 1'):
            results = MultiFitter(model).fit_polarized(data, objective='legacy_mask')

        assert results['pp'].success
        assert_allclose(model.scale.value, 1.0, atol=1e-4)


class TestLoadersSetMeasuredResolution:
    def test_load_as_dataset_sets_pointwise(self):
        dataset = load_as_dataset(os.path.join(PATH_STATIC, 'example.ort'))

        assert isinstance(dataset.resolution_function, Pointwise)
        assert_allclose(dataset.resolution_function.smearing(dataset.x), np.sqrt(dataset.xe))

    def test_three_column_file_keeps_none(self, tmp_path):
        path = tmp_path / 'bare.txt'
        np.savetxt(path, np.column_stack([Q, np.ones_like(Q), 0.1 * np.ones_like(Q)]))

        assert load_as_dataset(str(path)).resolution_function is None

    def test_fitter_uses_each_loaded_files_widths(self, tmp_path):
        interface = CalculatorFactory()
        model = _film_model('model', interface=interface)
        model.resolution_function = PercentageFwhm(5)
        calculator = interface()
        datasets = []
        for name, relative in (('sharp', 0.002), ('broad', 0.08)):
            path = tmp_path / f'{name}.txt'
            _write_four_column_file(path, Q, np.exp(-30 * Q), 0.05 * np.exp(-30 * Q), relative * Q)
            dataset = load_as_dataset(str(path))
            dataset.model = model
            datasets.append(dataset)

        fitter = MultiFitter.for_experiments(datasets)

        for index, dataset in enumerate(datasets):
            expected = calculator.reflectity_profile(Q, model.unique_name, resolution_function=resolution_from_dataset(dataset))
            assert_allclose(fitter._fit_func[index](Q), expected, rtol=1e-12)


class TestResolutionFromDataset:
    def test_pointwise_from_positive_variances(self):
        dataset = DataSet1D(x=[0.01, 0.02], y=[1.0, 2.0], ye=[0.1, 0.1], xe=[1e-8, 4e-8])
        resolution = resolution_from_dataset(dataset)
        assert isinstance(resolution, Pointwise)
        assert_allclose(resolution.smearing(), [1e-4, 2e-4])

    @pytest.mark.parametrize('xe', [None, [], [0.0, 0.0], [np.nan, np.nan]])
    def test_none_without_usable_variances(self, xe):
        dataset = DataSet1D(x=[0.01, 0.02], y=[1.0, 2.0], ye=[0.1, 0.1], xe=xe)
        if xe == []:
            dataset.xe = np.array([])
        assert resolution_from_dataset(dataset) is None


# ----------------------------------------------------------------------------
# Project: loading, combining, plotting, persistence
# ----------------------------------------------------------------------------


class TestUnsmearedCurveLeavesTheModelAlone:
    """`sample_data_for_model_at_index` binds a zero width to one calculation; it no longer swaps the model's."""

    def test_sample_data_for_model_at_index_does_not_touch_the_model(self):
        project = Project()
        project.default_model()
        model = project.models[0]
        smeared = project.model_data_for_model_at_index(0, Q)
        before = model.resolution_function
        wrapper = project._calculator()._wrapper

        with patch.object(wrapper, 'set_resolution_function', wraps=wrapper.set_resolution_function) as registered:
            sharp = project.sample_data_for_model_at_index(0, Q)

        assert not registered.called
        assert model.resolution_function is before
        assert wrapper.resolution_function_for(model.unique_name) is before
        assert not np.allclose(sharp.y, smeared.y)
        assert_allclose(project.model_data_for_model_at_index(0, Q).y, smeared.y, rtol=1e-12)


# ----------------------------------------------------------------------------
# Bayesian sampling
# ----------------------------------------------------------------------------


class TestSamplingUsesDatasetResolution:
    """`Project.prepare_sample` + `mcmc_sample()` compare each dataset to the curve the fit does."""

    @staticmethod
    def _two_files(tmp_path, relative_sigmas):
        return TestProjectResolution._two_files(tmp_path, relative_sigmas=relative_sigmas)

    def _project_with_two_datasets(self, tmp_path):
        paths = self._two_files(tmp_path, relative_sigmas=(0.002, 0.08))
        project = Project()
        project.models = ModelCollection(_film_model())
        project.load_new_experiment(paths[0])
        project.load_new_experiment(paths[1])
        project.minimizer = Minimizers.Bumps_simplex
        return project

    @staticmethod
    def _run_sampling(fitter, monkeypatch) -> dict:
        """Run `mcmc_sample()` with the DREAM sampler stubbed out; returns what was prepared."""
        seen = {}
        original = MultiFitter.prepare

        def prepare(self, inputs=None, **kwargs):
            seen['prepared'] = original(self, inputs, **kwargs)
            return seen['prepared']

        monkeypatch.setattr(MultiFitter, 'prepare', prepare)
        sampler = MagicMock()
        sampler.sample.return_value = MagicMock(draws=np.ones((4, 1)), param_names=['p'], state=None, logp=np.zeros(4))
        with patch('easyreflectometry.fitting.Sampler') as sampler_class:
            sampler_class.from_fitter.return_value = sampler
            fitter.mcmc_sample(samples=4, burn=1, thin=1)
        seen['sampler_kwargs'] = dict(zip(('fitter', 'x', 'y', 'weights'), sampler_class.from_fitter.call_args.args))
        return seen

    def test_each_dataset_is_sampled_against_the_curve_the_fit_sees(self, tmp_path, monkeypatch):
        project = self._project_with_two_datasets(tmp_path)
        fitter = project.prepare_sample()
        assert fitter.skipped == []

        seen = self._run_sampling(fitter, monkeypatch)

        prepared = seen['prepared']
        for index in (0, 1):
            x = project.experiments[index].x
            assert_allclose(seen['sampler_kwargs']['x'][index], x)
            assert_allclose(prepared.fit_funcs[index](x), project.model_data_for_experiment_at_index(index).y, rtol=1e-12)
        # The model follows the first dataset, so the model's own curve is not
        # what the broader second dataset is compared to.
        x = project.experiments[1].x
        assert not np.allclose(prepared.fit_funcs[1](x), project.model_data_for_model_at_index(0, q_range=x).y)

    def test_prepare_sample_covers_the_included_experiments_with_a_model(self, tmp_path):
        project = Project()
        project.models = ModelCollection(_film_model())
        with pytest.raises(FitScopeError, match='No experiment to sample'):
            project.prepare_sample()

        paths = self._two_files(tmp_path, relative_sigmas=(0.01, 0.03))
        project.load_new_experiment(paths[0])
        project.load_new_experiment(paths[1])
        project.experiments[1].include_in_fit = False
        fitter = project.prepare_sample()
        assert len(fitter.fit_datasets) == 1
        assert fitter.fit_datasets[0] is project.experiments[0]

        foreign = DataSet1D(name='foreign', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q), model=_film_model('foreign'))
        with pytest.raises(FitScopeError, match="'foreign' cannot be sampled"):
            project.prepare_sample([project.experiments[0], foreign])
        fitter = project.prepare_sample([project.experiments[0], foreign], skip_invalid=True)
        assert [dataset is project.experiments[0] for dataset in fitter.fit_datasets] == [True]
        # Reported like `PreparedFit.skipped`, not dropped silently.
        assert fitter.skipped == ['foreign']

    def test_settings_are_snapshotted_when_sampling_is_prepared(self, tmp_path):
        """An edit between `prepare_sample()` and `mcmc_sample()` waits for the next run, as for `prepare_fit()`."""
        project = self._project_with_two_datasets(tmp_path)
        prepared = project.prepare_fit()
        fitter = project.prepare_sample()
        assert fitter.settings is not project.fit_settings
        assert fitter.settings.minimizer == Minimizers.Bumps_simplex

        project.minimizer = Minimizers.LMFit_leastsq

        assert prepared.settings.minimizer == Minimizers.Bumps_simplex
        assert fitter.settings.minimizer == Minimizers.Bumps_simplex


# ----------------------------------------------------------------------------
# Through the real core Sampler, up to the BUMPS problem
# ----------------------------------------------------------------------------


class TestSamplerBoundary:
    """The likelihood the BUMPS sampler is handed, not just the inputs `prepare` saw.

    The core ``Sampler`` reshapes the datasets and wraps the evaluators before
    BUMPS sees them. Only ``Bumps.mcmc_sample`` is replaced: it builds the BUMPS
    curve exactly as the real one does and reports its chi-square instead of
    running DREAM. Each dataset's data are 1.05 times its correctly smeared
    curve with sigma 0.1 times that curve, so every residual is 0.5 -- but only
    if each dataset is evaluated with its own resolution.
    """

    @staticmethod
    def _sampled_chi2(fitter, data=None, **kwargs) -> tuple[float, int]:
        from easyscience.fitting.samplers.sampler_bumps import DreamSampler
        from easyscience.fitting.samplers.sampler_bumps import build_curve_problem

        seen = {}

        def run(self, x, y, weights, **_):
            problem, _, _ = build_curve_problem(self, np.asarray(x), np.asarray(y), np.asarray(weights))
            residuals = np.asarray(problem.residuals())
            seen['chi2'] = float(np.sum(residuals**2))
            seen['points'] = residuals.size
            return {'draws': np.zeros((1, 1)), 'param_names': ['p'], 'logp': np.zeros(1), 'internal_bumps_object': None}

        with patch.object(DreamSampler, 'run', run):
            fitter.mcmc_sample(data, samples=4, burn=1, thin=1, **kwargs)
        return seen['chi2'], seen['points']

    @staticmethod
    def _bumps(fitter: MultiFitter) -> MultiFitter:
        fitter.settings = FitSettings()
        fitter.switch_minimizer(Minimizers.Bumps)
        return fitter

    @staticmethod
    def _offset_dataset(name, curve, resolution, model=None):
        return DataSet1D(
            name=name,
            x=Q,
            y=1.05 * curve,
            ye=(0.1 * curve) ** 2,
            model=model,
            auto_background=False,
            resolution_function=resolution,
        )

    def _two_unpolarized(self):
        interface = CalculatorFactory()
        model = _film_model('model', interface=interface)
        model.resolution_function = PercentageFwhm(5)
        calculator = interface()
        return [
            self._offset_dataset(
                name, calculator.reflectity_profile(Q, model.unique_name, resolution_function=resolution), resolution, model
            )
            for name, resolution in (('sharp', _sharp()), ('broad', _broad()))
        ]

    def test_each_dataset_reaches_the_sampler_with_its_own_resolution(self):
        datasets = self._two_unpolarized()

        chi2, points = self._sampled_chi2(self._bumps(MultiFitter.for_experiments(datasets)))

        assert points == 2 * len(Q)
        assert_allclose(chi2, points * 0.25, rtol=1e-9)

    def test_a_datagroup_reaches_the_sampler_with_the_datasets_resolutions(self):
        import scipp as sc

        datasets = self._two_unpolarized()
        data = sc.DataGroup({
            'coords': {f'Qz_{i}': sc.array(dims=[f'Qz_{i}'], values=d.x) for i, d in enumerate(datasets)},
            'data': {f'R_{i}': sc.array(dims=[f'Qz_{i}'], values=d.y, variances=d.ye) for i, d in enumerate(datasets)},
        })

        chi2, points = self._sampled_chi2(self._bumps(MultiFitter.for_experiments(datasets)), data)

        assert points == 2 * len(Q)
        assert_allclose(chi2, points * 0.25, rtol=1e-9)

    def test_the_models_resolution_would_not_fit(self):
        """Guards the tests above: with the model's 5 % the residuals are not all 0.5."""
        datasets = self._two_unpolarized()
        for dataset in datasets:
            dataset.resolution_function = None

        chi2, points = self._sampled_chi2(self._bumps(MultiFitter.for_experiments(datasets)))

        assert not np.isclose(chi2, points * 0.25, rtol=1e-3)

    def test_polarized_channels_reach_the_sampler_with_their_own_resolution(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        model = _film_model('magnetic', magnetism=LayerMagnetism(rho_m=1.5, theta_m=270.0), interface=interface)
        model.resolution_function = PercentageFwhm(5)
        calculator = interface()
        channels = {
            channel: self._offset_dataset(
                channel,
                calculator.reflectivity_profile_channel(Q, model.unique_name, channel, resolution_function=resolution),
                resolution,
            )
            for channel, resolution in (('pp', _sharp()), ('mm', _broad()))
        }
        data = PolarizedDataSet(name='polarized', channels=channels, model=model)

        chi2, points = self._sampled_chi2(self._bumps(MultiFitter.for_experiments([data])))

        assert points == 2 * len(Q)
        assert_allclose(chi2, points * 0.25, rtol=1e-9)

    def test_masked_merged_dataset_keeps_each_points_width(self):
        """`legacy_mask` drops a point of a merged dataset whose q repeat with different widths."""
        interface, merged = TestMaskedFitKeepsPointWidths._merged_truth()
        merged.model = _film_model('model', interface=interface)

        with pytest.warns(UserWarning, match='Masked 1'):
            chi2, points = self._sampled_chi2(self._bumps(MultiFitter.for_experiments([merged])), objective='legacy_mask')

        assert points == 2 * len(Q) - 1
        # The data are the exact curves, so any q-interpolated width shows up here.
        assert chi2 < 1e-12


# ----------------------------------------------------------------------------
# MultiFitter.fit with a DataGroup on a for_experiments fitter
# ----------------------------------------------------------------------------


class TestDataGroupFitOnExperiments:
    """Curve *k* of a fitted ``DataGroup`` is written back for dataset *k*'s own model.

    Two datasets share the first model and a third belongs to a second model,
    so the per-curve result index and the de-duplicated model list differ (the
    old code indexed the latter by the former and failed on the second curve).
    """

    def test_shared_model_curves_are_written_for_their_own_models(self):
        import scipp as sc

        interface = CalculatorFactory()
        first = _film_model('first', interface=interface)
        second = _film_model('second', interface=interface)
        second.sample[1].layers[0].thickness.value = 60.0
        calculator = interface()
        datasets = []
        for name, model, resolution in (('a', first, _sharp()), ('b', first, _broad()), ('c', second, _sharp())):
            curve = calculator.reflectity_profile(Q, model.unique_name, resolution_function=resolution)
            datasets.append(
                DataSet1D(
                    name=name,
                    x=Q,
                    y=curve,
                    ye=(0.05 * curve) ** 2,
                    model=model,
                    auto_background=False,
                    resolution_function=resolution,
                )
            )
        data = sc.DataGroup({
            'coords': {f'Qz_{i}': sc.array(dims=[f'Qz_{i}'], values=Q, unit=sc.Unit('1/angstrom')) for i in range(3)},
            'data': {f'R_{i}': sc.array(dims=[f'Qz_{i}'], values=d.y, variances=d.ye) for i, d in enumerate(datasets)},
            'attrs': {},
        })
        fitter = MultiFitter.for_experiments(datasets)

        result = fitter.fit(data)

        for i, dataset in enumerate(datasets):
            assert_allclose(result[f'R_{i}_model'].values, fitter._fit_func[i](Q), rtol=1e-12)
            z, sld = dataset.model.interface.sld_profile(dataset.model.unique_name)
            assert_allclose(result['coords'][f'z_{i}'].values, z)
            assert_allclose(result[f'SLD_{i}'].values, sld * 1e-6)
            assert result['attrs'][f'R_{i}_model']['model'].value['name'] == dataset.model.name
        # The two models really give different SLD profiles here.
        sld_first, sld_second = result['SLD_1'].values, result['SLD_2'].values
        assert sld_first.shape != sld_second.shape or not np.allclose(sld_first, sld_second)

    def test_curves_pair_with_datasets_by_number_not_insertion_order(self):
        import scipp as sc

        interface = CalculatorFactory()
        datasets = [
            DataSet1D(
                name=name,
                x=Q,
                y=np.ones_like(Q),
                ye=np.ones_like(Q),
                model=_film_model(name, interface=interface),
                auto_background=False,
            )
            for name in ('zero', 'one')
        ]
        fitter = MultiFitter.for_experiments(datasets)
        # Inserted as 1, 0: curve 0 is still dataset 0.
        data = sc.DataGroup({
            'coords': {f'Qz_{i}': sc.array(dims=[f'Qz_{i}'], values=Q * (i + 1)) for i in (1, 0)},
            'data': {f'R_{i}': sc.array(dims=[f'Qz_{i}'], values=np.ones_like(Q), variances=np.ones_like(Q)) for i in (1, 0)},
        })

        refl_nums, inputs = fitter._datagroup_inputs(data)

        assert refl_nums == ['0', '1']
        assert [item.model.name for item in inputs] == ['zero', 'one']
        assert_allclose(inputs[1].x, 2 * Q)

    def test_curve_numbers_sort_numerically(self):
        import scipp as sc

        from easyreflectometry.fitting import _curve_ids

        data = sc.DataGroup({'coords': {f'Qz_{i}': sc.array(dims=['q'], values=Q) for i in (10, 2, 0)}, 'data': {}})

        assert _curve_ids(data) == ['0', '2', '10']


class TestProjectResolution:
    @staticmethod
    def _two_files(tmp_path, relative_sigmas=(0.01, 0.03)):
        q_low = np.linspace(0.01, 0.1, 10)
        q_high = np.linspace(0.08, 0.3, 12)
        paths = []
        for name, q, relative in zip(('low', 'high'), (q_low, q_high), relative_sigmas):
            r = np.exp(-30 * q)
            path = tmp_path / f'{name}.txt'
            _write_four_column_file(path, q, r, 0.05 * r, relative * q)
            paths.append(path)
        return paths

    def test_loaded_experiment_carries_its_measured_resolution(self):
        project = Project()
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'example.ort'))

        experiment = project.experiments[0]
        assert isinstance(experiment.resolution_function, Pointwise)
        assert_allclose(experiment.resolution_function.smearing(experiment.x), np.sqrt(experiment.xe))
        assert isinstance(project.models[0].resolution_function, Pointwise)

    def test_two_contrasts_on_two_models_keep_their_own_smearing(self, tmp_path):
        paths = self._two_files(tmp_path, relative_sigmas=(0.002, 0.08))
        project = Project()
        project.models = ModelCollection(_film_model('first'), _film_model('second'))
        project.load_experiment_for_model_at_index(paths[0], 0)
        before = project.model_data_for_experiment_at_index(0).y

        project.load_experiment_for_model_at_index(paths[1], 1)

        after = project.model_data_for_experiment_at_index(0).y
        assert_allclose(after, before, rtol=1e-12)
        # And each model computes with its own resolution on the shared calculator.
        wrapper = project._calculator()._wrapper
        for index in (0, 1):
            assert wrapper.resolution_function_for(project.models[index].unique_name) is (
                project.models[index].resolution_function
            )

    def test_model_data_for_experiment_matches_what_the_fit_sees(self, tmp_path):
        paths = self._two_files(tmp_path, relative_sigmas=(0.002, 0.08))
        project = Project()
        project.default_model()
        project.load_new_experiment(paths[0])
        project.load_new_experiment(paths[1])

        fitter = MultiFitter.for_experiments(list(project.experiments.values()))

        for index in (0, 1):
            curve = project.model_data_for_experiment_at_index(index)
            assert_allclose(curve.x, project.experiments[index].x)
            assert_allclose(curve.y, fitter._fit_func[index](project.experiments[index].x), rtol=1e-12)

    def test_model_data_for_experiment_rejects_wrong_channel_use(self):
        project = Project()
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'example.ort'))
        with pytest.raises(ValueError, match='not polarized'):
            project.model_data_for_experiment_at_index(0, channel='pp')
        with pytest.raises(IndexError):
            project.model_data_for_experiment_at_index(3)

    def test_load_experiment_from_files_combines_curves_into_one_contrast(self, tmp_path):
        paths = self._two_files(tmp_path)
        project = Project()
        project.default_model()

        index = project.load_experiment_from_files(paths)

        assert index == 0
        experiment = project.experiments[0]
        assert experiment.name == 'Experiment 0'
        assert len(experiment.x) == 22
        assert np.all(np.diff(experiment.x) >= 0)
        assert experiment.model is project.models[0]
        assert isinstance(experiment.resolution_function, Pointwise)
        assert isinstance(project.models[0].resolution_function, Pointwise)
        assert_allclose(experiment.resolution_function.smearing(experiment.x), np.sqrt(experiment.xe))

    def test_load_experiment_from_files_honours_name_and_model_index(self, tmp_path):
        paths = self._two_files(tmp_path)
        project = Project()
        project.models = ModelCollection(_film_model('first'), _film_model('second'))

        index = project.load_experiment_from_files(paths, model_index=1, name='Contrast A')

        assert index == 0
        assert project.experiments[0].name == 'Contrast A'
        assert project.experiments[0].model is project.models[1]

    def test_load_experiment_from_files_rejects_empty_and_multi_dataset_files(self):
        project = Project()
        project.default_model()
        with pytest.raises(ValueError, match='At least one file'):
            project.load_experiment_from_files([])
        with pytest.raises(ValueError, match='multiple datasets'):
            project.load_experiment_from_files([os.path.join(PATH_STATIC, 'polarized_2ch.ort')])

    def test_append_to_experiment_extends_the_contrast(self, tmp_path):
        paths = self._two_files(tmp_path)
        project = Project()
        project.default_model()
        project.load_new_experiment(paths[0])
        project.experiments[0].name = 'Contrast A'

        project.append_to_experiment_at_index(0, paths[1])

        experiment = project.experiments[0]
        assert experiment.name == 'Contrast A'
        assert len(experiment.x) == 22
        assert experiment.model is project.models[0]
        assert_allclose(experiment.resolution_function.smearing(experiment.x), np.sqrt(experiment.xe))

    def test_append_to_experiment_rejects_missing_and_polarized(self, tmp_path):
        paths = self._two_files(tmp_path)
        project = Project()
        project.default_model()
        with pytest.raises(IndexError):
            project.append_to_experiment_at_index(0, paths[0])
        project.load_polarized_experiment({'pp': paths[0], 'mm': paths[1]})
        with pytest.raises(ValueError, match='polarized'):
            project.append_to_experiment_at_index(0, paths[0])

    def test_polarized_channels_keep_their_own_resolution(self, tmp_path):
        paths = self._two_files(tmp_path, relative_sigmas=(0.01, 0.03))
        project = Project()
        project.default_model()

        project.load_polarized_experiment({'pp': paths[0], 'mm': paths[1]})

        experiment = project.experiments[0]
        pp, mm = experiment['pp'], experiment['mm']
        assert_allclose(pp.resolution_function.smearing(pp.x), 0.01 * pp.x)
        assert_allclose(mm.resolution_function.smearing(mm.x), 0.03 * mm.x)
        # The model follows the first channel, as before.
        assert project.models[0].resolution_function is pp.resolution_function

    def test_dict_round_trip_restores_dataset_resolutions(self, tmp_path):
        paths = self._two_files(tmp_path)
        project = Project()
        project.default_model()
        project.load_new_experiment(paths[0])
        project.load_polarized_experiment({'pp': paths[0], 'mm': paths[1]})
        project_dict = project.as_dict()

        assert project_dict['experiments'][0][4]['smearing'] == 'Pointwise'
        assert project_dict['experiments'][1]['channels']['mm'][5]['smearing'] == 'Pointwise'

        global_object.registry.clear()
        restored = Project()
        restored.from_dict(project_dict)

        plain = restored.experiments[0]
        assert_allclose(plain.resolution_function.smearing(plain.x), project.experiments[0].resolution_function.smearing())
        mm = restored.experiments[1]['mm']
        assert_allclose(mm.resolution_function.smearing(mm.x), project.experiments[1]['mm'].resolution_function.smearing())

    def test_older_project_files_derive_dataset_resolution_from_xe(self, tmp_path):
        paths = self._two_files(tmp_path)
        project = Project()
        project.default_model()
        project.load_new_experiment(paths[0])
        project.load_polarized_experiment({'pp': paths[0], 'mm': paths[1]})
        project_dict = project.as_dict()
        # Strip the resolution entries, as a file written before they existed.
        project_dict['experiments'][0] = project_dict['experiments'][0][:4]
        for arrays in project_dict['experiments'][1]['channels'].values():
            del arrays[5]

        global_object.registry.clear()
        restored = Project()
        restored.from_dict(project_dict)

        plain = restored.experiments[0]
        assert isinstance(plain.resolution_function, Pointwise)
        assert_allclose(plain.resolution_function.smearing(plain.x), np.sqrt(plain.xe))
        mm = restored.experiments[1]['mm']
        assert_allclose(mm.resolution_function.smearing(mm.x), np.sqrt(mm.xe))

    def test_dataset_without_xe_or_resolution_round_trips_without_either(self):
        project = Project()
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'example.ort'))
        project.experiments[0].xe = None
        project.experiments[0].resolution_function = None
        project_dict = project.as_dict()

        assert len(project_dict['experiments'][0]) == 3

        global_object.registry.clear()
        restored = Project()
        restored.from_dict(project_dict)
        assert restored.experiments[0].resolution_function is None

    @staticmethod
    def _round_trip(project):
        project_dict = project.as_dict()
        global_object.registry.clear()
        restored = Project()
        restored.from_dict(project_dict)
        return project_dict, restored

    def test_explicit_model_resolution_choice_round_trips(self):
        project = Project()
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'example.ort'))
        project.experiments[0].resolution_function = None
        project.models[0].resolution_function = PercentageFwhm(1)
        before = project.model_data_for_experiment_at_index(0).y

        project_dict, restored = self._round_trip(project)

        assert len(project_dict['experiments'][0]) == 5
        assert project_dict['experiments'][0][4] is None
        assert restored.experiments[0].resolution_function is None
        assert_allclose(restored.model_data_for_experiment_at_index(0).y, before, rtol=1e-10)

    def test_explicit_model_resolution_choice_round_trips_for_a_channel(self, tmp_path):
        paths = self._two_files(tmp_path)
        project = Project()
        project.default_model()
        project.load_polarized_experiment({'pp': paths[0], 'mm': paths[1]})
        # 'pp' of a non-magnetic model is the unpolarized calculation.
        project.experiments[0]['pp'].resolution_function = None
        project.models[0].resolution_function = PercentageFwhm(1)
        before = project.model_data_for_experiment_at_index(0, channel='pp').y

        _, restored = self._round_trip(project)

        assert restored.experiments[0]['pp'].resolution_function is None
        assert isinstance(restored.experiments[0]['mm'].resolution_function, Pointwise)
        assert_allclose(restored.model_data_for_experiment_at_index(0, channel='pp').y, before, rtol=1e-10)

    def test_explicit_resolution_without_xe_round_trips(self):
        project = Project()
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'example.ort'))
        project.experiments[0].xe = None
        project.experiments[0].resolution_function = PercentageFwhm(2)
        before = project.model_data_for_experiment_at_index(0).y

        project_dict, restored = self._round_trip(project)

        assert project_dict['experiments'][0][3] is None
        assert isinstance(restored.experiments[0].resolution_function, PercentageFwhm)
        assert_allclose(restored.model_data_for_experiment_at_index(0).y, before, rtol=1e-10)

    def test_append_keeps_the_experiments_explicit_resolution(self, tmp_path):
        paths = self._two_files(tmp_path)
        project = Project()
        project.default_model()
        project.load_new_experiment(paths[0])
        project.experiments[0].resolution_function = PercentageFwhm(20)
        original_x = np.array(project.experiments[0].x)

        project.append_to_experiment_at_index(0, paths[1])

        experiment = project.experiments[0]
        widths = experiment.resolution_function.smearing(experiment.x)
        appended_q = np.linspace(0.08, 0.3, 12)
        for q, width in zip(experiment.x, widths):
            candidates = []
            if np.any(np.isclose(original_x, q)):
                candidates.append(0.20 * q / SIGMA_TO_FWHM)
            if np.any(np.isclose(appended_q, q)):
                candidates.append(0.03 * q)
            assert any(np.isclose(width, candidate) for candidate in candidates)
