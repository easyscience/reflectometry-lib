# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the ``Sampler``-based MCMC workflow in ``MultiFitter.mcmc_sample``.

The actual BUMPS/DREAM sampling is delegated to ``easyscience.fitting.Sampler``;
these tests mock the ``Sampler`` class so they stay fast while still exercising
the wrapper logic: data preparation, zero-variance guards, warning emission,
result-dict construction, and retention of the sampler for chain extension.
"""

from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import scipp as sc
from easyscience import Minimizers
from easyscience import global_object

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.fit_settings import FitSettings
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.fitting import _fit_result_reduced_chi
from easyreflectometry.fitting import _flatten_list
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def _make_fitter() -> MultiFitter:
    model = Model()
    model.interface = CalculatorFactory()
    return MultiFitter(model)


def _make_bumps_fitter() -> MultiFitter:
    """A MultiFitter whose core fitter reports a BUMPS minimizer without running one."""
    fitter = _make_fitter()
    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.minimizer.package = 'bumps'
    return fitter


def _make_data(variances: np.ndarray, n: int = 10) -> sc.DataGroup:
    return sc.DataGroup({
        'coords': {'Qz_0': sc.array(dims=['Qz_0'], values=np.linspace(0.01, 0.3, n))},
        'data': {'R_0': sc.array(dims=['Qz_0'], values=np.ones(n), variances=variances)},
    })


def _fake_sampling_results():
    results = MagicMock()
    results.draws = np.ones((10, 2))
    results.param_names = ['a', 'b']
    results.state = object()
    results.logp = np.zeros(10)
    return results


def _patch_sampler(capture: dict, results=None):
    """Patch ``easyreflectometry.fitting.Sampler`` recording ctor and sample() args."""
    results = results if results is not None else _fake_sampling_results()

    def _ctor(fitter, x, y, weights, **kwargs):
        capture['fitter'] = fitter
        capture['x'] = x
        capture['y'] = y
        capture['weights'] = weights
        instance = MagicMock()

        def _sample(**sample_kwargs):
            capture.update(sample_kwargs)
            return results

        instance.sample = MagicMock(side_effect=_sample)
        capture['instance'] = instance
        return instance

    sampler_cls = MagicMock()
    sampler_cls.from_fitter.side_effect = _ctor
    return patch('easyreflectometry.fitting.Sampler', sampler_cls)


class TestMCMCSampleGuards:
    def test_raises_runtime_error_when_minimizer_is_not_bumps(self):
        fitter = _make_fitter()  # default minimizer is LMFit, not BUMPS
        data = _make_data(np.ones(10) * 0.01)

        with pytest.raises(RuntimeError, match='Bayesian sampling requires a BUMPS minimizer'):
            fitter.mcmc_sample(data)

    def test_engine_is_checked_before_the_data_is_prepared(self, monkeypatch):
        fitter = _make_fitter()
        fitter.settings = FitSettings()  # LMFit, as a project's fitter carries
        monkeypatch.setattr(MultiFitter, 'prepare', MagicMock(side_effect=AssertionError('prepared')))
        with pytest.raises(RuntimeError, match='project.minimizer'):
            fitter.mcmc_sample(_make_data(np.zeros(10)), objective='mighell')

    def test_a_switch_through_the_settings_passes_the_engine_check(self):
        fitter = _make_fitter()
        fitter.settings = FitSettings()
        fitter.switch_minimizer(Minimizers.Bumps)
        capture = {}
        with _patch_sampler(capture):
            fitter.mcmc_sample(_make_data(np.ones(10) * 0.01), samples=10, burn=2, thin=1)
        assert capture['instance'].sample.called

    def test_all_zero_variance_raises_value_error(self):
        """Sampling without any uncertainties has no defined likelihood."""
        fitter = _make_bumps_fitter()
        data = _make_data(np.zeros(10))

        capture = {}
        with _patch_sampler(capture) as sampler_cls:
            with pytest.raises(ValueError, match='all points have zero variance'):
                fitter.mcmc_sample(data)
        sampler_cls.from_fitter.assert_not_called()

    def test_all_zero_variance_allowed_with_mighell_objective(self):
        """objective='mighell' is the explicit opt-in for missing uncertainties."""
        fitter = _make_bumps_fitter()
        data = _make_data(np.zeros(10))

        capture = {}
        with _patch_sampler(capture):
            with pytest.warns(UserWarning, match='Mighell transform to all'):
                result = fitter.mcmc_sample(data, samples=100, burn=10, thin=2, objective='mighell')
        assert set(result) == {'draws', 'param_names', 'state', 'logp'}


class TestMCMCSampleWarnings:
    def test_legacy_mask_warns_about_masked_points(self):
        fitter = _make_bumps_fitter()
        variances = np.ones(10) * 0.01
        variances[3] = 0.0
        data = _make_data(variances)

        capture = {}
        with _patch_sampler(capture):
            with pytest.warns(UserWarning, match='Masked 1 data point'):
                fitter.mcmc_sample(data, samples=100, burn=10, thin=2, objective='legacy_mask')
        # The masked point must not reach the Sampler
        assert len(capture['x'][0]) == 9

    def test_hybrid_warns_about_mighell_substitution(self):
        fitter = _make_bumps_fitter()
        variances = np.ones(10) * 0.01
        variances[3] = 0.0
        data = _make_data(variances)

        capture = {}
        with _patch_sampler(capture):
            with pytest.warns(UserWarning, match='Mighell substitution to 1'):
                fitter.mcmc_sample(data, samples=100, burn=10, thin=2)
        # Hybrid keeps every point
        assert len(capture['x'][0]) == 10


class TestMCMCSampleDispatch:
    def test_returns_dict_built_from_sampling_results(self):
        fitter = _make_bumps_fitter()
        data = _make_data(np.ones(10) * 0.01)
        results = _fake_sampling_results()

        capture = {}
        with _patch_sampler(capture, results=results):
            result = fitter.mcmc_sample(data, samples=100, burn=20, thin=2, population=5)

        assert capture['fitter'] is fitter.easy_science_multi_fitter
        assert result['draws'] is results.draws
        assert result['param_names'] == results.param_names
        assert result['state'] is results.state
        assert result['logp'] is results.logp

    def test_forwards_hyperparameters_to_sampler_sample(self):
        fitter = _make_bumps_fitter()
        data = _make_data(np.ones(10) * 0.01)

        capture = {}
        with _patch_sampler(capture):
            fitter.mcmc_sample(data, samples=500, burn=100, thin=5, population=8)
        assert capture['samples'] == 500
        assert capture['burn'] == 100
        assert capture['thin'] == 5
        assert capture['population'] == 8
        assert capture['sampler_kwargs'] is None

    def test_initializer_forwarded_via_sampler_kwargs(self):
        fitter = _make_bumps_fitter()
        data = _make_data(np.ones(10) * 0.01)

        capture = {}
        with _patch_sampler(capture):
            fitter.mcmc_sample(data, samples=100, burn=20, thin=2, initializer='lhs')
        assert capture['sampler_kwargs'] == {'init': 'lhs'}

    def test_sampler_retained_for_chain_extension(self):
        """The Sampler instance must be kept on ``fitter.sampler`` so the chain
        can be extended with ``fitter.sampler.extend(...)`` without a new burn-in."""
        fitter = _make_bumps_fitter()
        assert fitter.sampler is None

        data = _make_data(np.ones(10) * 0.01)
        capture = {}
        with _patch_sampler(capture):
            fitter.mcmc_sample(data, samples=100, burn=20, thin=2)

        assert fitter.sampler is capture['instance']


class TestMCMCSampleOnExperiments:
    """A `for_experiments` fitter samples each dataset with its own resolution.

    Without a ``DataGroup`` the datasets themselves are sampled; with one,
    curve *k* takes the model, channel and resolution of dataset *k*.
    """

    @staticmethod
    def _fitter_and_datasets():
        model = Model()
        model.interface = CalculatorFactory()
        q = np.linspace(0.01, 0.3, 10)
        datasets = [
            DataSet1D(name=name, x=q, y=np.ones(10), ye=np.ones(10) * 0.01, model=model, resolution_function=resolution)
            for name, resolution in (('sharp', PercentageFwhm(0.5)), ('broad', PercentageFwhm(8.0)))
        ]
        fitter = MultiFitter.for_experiments(datasets)
        fitter.settings = FitSettings()
        fitter.switch_minimizer(Minimizers.Bumps)
        return fitter, datasets

    @staticmethod
    def _spy_on_prepare(monkeypatch) -> dict:
        seen = {}
        original = MultiFitter.prepare

        def prepare(self, inputs=None, **kwargs):
            seen['inputs'] = inputs
            seen['prepared'] = original(self, inputs, **kwargs)
            return seen['prepared']

        monkeypatch.setattr(MultiFitter, 'prepare', prepare)
        return seen

    def test_without_a_datagroup_the_datasets_are_sampled_with_their_own_resolution(self, monkeypatch):
        fitter, datasets = self._fitter_and_datasets()
        seen = self._spy_on_prepare(monkeypatch)

        capture = {}
        with _patch_sampler(capture):
            fitter.mcmc_sample(samples=10, burn=2, thin=1)

        inputs = seen['inputs']
        assert [item.resolution_function for item in inputs] == [dataset.resolution_function for dataset in datasets]
        assert [item.label for item in inputs] == ['sharp', 'broad']
        assert all(item.channel is None for item in inputs)
        assert len(capture['x']) == 2
        np.testing.assert_allclose(capture['x'][1], datasets[1].x)
        assert capture['instance'].sample.called

    def test_a_datagroup_curve_is_sampled_as_its_dataset(self, monkeypatch):
        fitter, datasets = self._fitter_and_datasets()
        seen = self._spy_on_prepare(monkeypatch)
        q = np.linspace(0.02, 0.2, 10)
        data = sc.DataGroup({
            'coords': {f'Qz_{i}': sc.array(dims=[f'Qz_{i}'], values=q) for i in range(2)},
            'data': {
                f'R_{i}': sc.array(dims=[f'Qz_{i}'], values=np.full(10, 2.0 + i), variances=np.ones(10) * 0.01)
                for i in range(2)
            },
        })

        with _patch_sampler({}):
            fitter.mcmc_sample(data, samples=10, burn=2, thin=1)

        inputs = seen['inputs']
        assert [item.resolution_function for item in inputs] == [dataset.resolution_function for dataset in datasets]
        assert all(item.model is datasets[0].model for item in inputs)
        assert [item.label for item in inputs] == ['reflectivity 0', 'reflectivity 1']
        # The arrays are the DataGroup's, not the datasets'.
        np.testing.assert_allclose(inputs[1].x, q)
        np.testing.assert_allclose(inputs[1].y, 3.0)

    def test_a_datagroup_must_hold_one_curve_per_dataset(self):
        fitter, _ = self._fitter_and_datasets()
        with pytest.raises(ValueError, match='one curve per dataset'):
            fitter.mcmc_sample(_make_data(np.ones(10) * 0.01))

    def test_without_a_datagroup_a_fitter_built_from_models_is_refused(self):
        fitter = _make_bumps_fitter()
        with pytest.raises(ValueError, match='for_experiments'):
            fitter.mcmc_sample()


class TestMultiFitterHelpers:
    def test_switch_minimizer_delegates_to_core_fitter(self):
        fitter = _make_fitter()
        fitter.easy_science_multi_fitter = MagicMock()
        fitter.switch_minimizer(Minimizers.Bumps)
        fitter.easy_science_multi_fitter.switch_minimizer.assert_called_once_with(Minimizers.Bumps)

    def test_flatten_list_flattens_nested_lists(self):
        result = _flatten_list([[1, 2], [3], [4, 5]])
        assert isinstance(result, np.ndarray)
        assert list(result) == [1, 2, 3, 4, 5]

    def test_fit_result_reduced_chi_raises_without_any_attribute(self):
        result = MagicMock(spec=[])  # no reduced_chi, no reduced_chi2
        with pytest.raises(AttributeError, match='neither reduced_chi nor reduced_chi2'):
            _fit_result_reduced_chi(result)

    def test_fit_func_computes_reflectivity_through_calculator(self):
        """The factory's fit_func must evaluate the model reflectivity profile."""
        fitter = _make_fitter()
        q = np.linspace(0.01, 0.1, 5)
        reflectivity = fitter._fit_func[0](q)
        assert np.shape(reflectivity) == (5,)
        assert np.all(np.isfinite(reflectivity))
