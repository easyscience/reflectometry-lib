# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""One preparation behind every fit entry point."""

import gc
import warnings

import numpy as np
import pytest
import scipp as sc
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.fitting import FitPreconditionError
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import Pointwise
from easyreflectometry.project import Project

Q = np.linspace(0.01, 0.2, 12)


@pytest.fixture(autouse=True)
def _clean_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


def _model() -> Model:
    model = Model()
    model.interface = CalculatorFactory()
    return model


def _data(variances=None, resolution=None, name='d') -> DataSet1D:
    y = np.exp(-Q * 30)
    ye = np.full(Q.size, 1e-4) if variances is None else np.asarray(variances, dtype=float)
    dataset = DataSet1D(name=name, x=Q, y=y, ye=ye)
    if resolution is not None:
        dataset.resolution_function = resolution
    return dataset


def _datagroup(dataset: DataSet1D) -> sc.DataGroup:
    return sc.DataGroup({
        'coords': {'Qz_0': sc.array(dims=['Qz_0'], values=np.asarray(dataset.x))},
        'data': {'R_0': sc.array(dims=['Qz_0'], values=np.asarray(dataset.y), variances=np.asarray(dataset.ye))},
    })


def _prepared_through(entry: str, fitter: MultiFitter, dataset: DataSet1D):
    """The run an entry point prepares, captured where it would execute."""
    captured = {}

    def capture(self, **kwargs):
        captured['prepared'] = self
        raise RuntimeError('stop')

    from easyreflectometry import fitting

    original = fitting.PreparedFit.execute
    fitting.PreparedFit.execute = capture
    try:
        with pytest.raises(RuntimeError, match='stop'):
            if entry == 'fit':
                fitter.fit(_datagroup(dataset))
            else:
                fitter.fit_single_data_set_1d(dataset)
    finally:
        fitting.PreparedFit.execute = original
    return captured['prepared']


class TestParity:
    @pytest.mark.parametrize('objective', ['hybrid', 'mighell', 'legacy_mask'])
    def test_every_entry_point_prepares_the_same_run(self, objective):
        variances = np.full(Q.size, 1e-4)
        variances[3] = 0.0
        model = _model()
        dataset = _data(variances)
        dataset.model = model
        fitter = MultiFitter(model, objective=objective)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            runs = [
                _prepared_through('fit', fitter, dataset),
                _prepared_through('single', fitter, dataset),
                MultiFitter.for_experiments([dataset], objective=objective).prepare(),
            ]

        reference = runs[0]
        for run in runs[1:]:
            for key in ('x', 'y', 'weights'):
                np.testing.assert_allclose(run.fitted[0][key], reference.fitted[0][key])
            np.testing.assert_allclose(run.fit_funcs[0](run.fitted[0]['x']), reference.fit_funcs[0](reference.fitted[0]['x']))

    def test_polarized_channels_keep_their_order(self):
        model = _model()
        model.interface.switch('refl1d')
        datasets = {
            channel: DataSet1D(name=channel, x=Q, y=np.exp(-Q * k), ye=np.full(Q.size, 1e-4))
            for channel, k in (('pp', 20), ('mm', 40))
        }
        polarized = PolarizedDataSet(name='p', channels=datasets, model=model)

        run = MultiFitter.for_experiments([polarized]).prepare()

        assert [channel.value for channel in run.channels] == ['pp', 'mm']
        np.testing.assert_allclose(run.fitted[1]['y'], np.exp(-Q * 40))


class TestPerPointResolution:
    def test_masked_points_keep_their_own_width(self):
        widths = np.linspace(1e-4, 5e-3, Q.size)
        variances = np.full(Q.size, 1e-4)
        variances[[2, 7]] = 0.0
        model = _model()
        dataset = _data(variances, Pointwise([Q, np.exp(-Q * 30), widths**2]))
        dataset.model = model

        with pytest.warns(UserWarning, match='Masked 2'):
            run = MultiFitter.for_experiments([dataset], objective='legacy_mask').prepare()

        keep = variances > 0
        # The function the engine runs, evaluated on the fitted points, equals
        # the full-resolution curve at those points: every width stayed with its point.
        np.testing.assert_allclose(run.fit_funcs[0](Q[keep]), run.curve_funcs[0](Q)[keep])

    def test_preparing_twice_starts_from_the_measured_data(self):
        variances = np.full(Q.size, 1e-4)
        variances[0] = 0.0
        model = _model()
        dataset = _data(variances)
        dataset.model = model
        fitter = MultiFitter.for_experiments([dataset])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            masked = fitter.prepare(objective='legacy_mask')
            hybrid = fitter.prepare(objective='hybrid')

        assert masked.fitted[0]['x'].size == Q.size - 1
        assert hybrid.fitted[0]['x'].size == Q.size


class TestPreconditions:
    def _project(self) -> tuple[Project, DataSet1D]:
        project = Project()
        project.default_model()
        dataset = _data()
        dataset.model = project.models[0]
        return project, dataset

    def test_differential_evolution_refuses_an_unbounded_parameter(self):
        project, dataset = self._project()
        model = project.models[0]
        model.scale.fixed = False
        model.scale.max = np.inf
        model.background.fixed = False
        model.background.max = 1e-5
        project.minimizer = AvailableMinimizers.LMFit_differential_evolution

        with pytest.raises(FitPreconditionError) as error:
            project.prepare_fit([dataset])

        assert error.value.parameters == [model.scale]

    def test_fixed_parameters_are_not_reported(self):
        project, dataset = self._project()
        project.models[0].scale.max = np.inf  # fixed: irrelevant
        project.minimizer = AvailableMinimizers.LMFit_differential_evolution
        project.prepare_fit([dataset])

    def test_other_minimizers_accept_unbounded_parameters(self):
        project, dataset = self._project()
        project.models[0].scale.fixed = False
        project.models[0].scale.max = np.inf
        project.prepare_fit([dataset])

    def test_the_engine_is_not_reached(self, monkeypatch):
        project, dataset = self._project()
        project.models[0].scale.fixed = False
        project.models[0].scale.max = np.inf
        project.minimizer = AvailableMinimizers.LMFit_differential_evolution
        reached = []
        monkeypatch.setattr('lmfit.Model.fit', lambda *a, **k: reached.append(True))
        with pytest.raises(FitPreconditionError):
            project.fitter.fit_single_data_set_1d(dataset)
        assert reached == []


class TestBoundStarts:
    @staticmethod
    def _project(value: float) -> tuple[Project, DataSet1D]:
        project = Project()
        project.default_model()
        scale = project.models[0].scale
        scale.fixed = False
        scale.min, scale.max = 0.5, 1.5
        scale.value = value
        dataset = _data()
        dataset.model = project.models[0]
        return project, dataset

    def test_leastsq_warns_on_a_parameter_starting_on_a_bound(self):
        project, dataset = self._project(0.5)
        with pytest.warns(UserWarning, match='start on a bound'):
            project.prepare_fit([dataset])

    @pytest.mark.parametrize(
        ('value', 'minimizer'),
        [(1.0, AvailableMinimizers.LMFit_leastsq), (0.5, AvailableMinimizers.LMFit_scipy_least_squares)],
    )
    def test_no_warning_inside_the_range_or_for_other_methods(self, value, minimizer):
        project, dataset = self._project(value)
        project.minimizer = minimizer
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='.*start on a bound')
            project.prepare_fit([dataset])


class TestArrayOwnership:
    @pytest.mark.parametrize('objective', ['hybrid', 'legacy_mask'])
    def test_editing_the_dataset_does_not_reach_a_prepared_run(self, objective):
        variances = np.full(Q.size, 1e-4)
        variances[0] = 0.0
        dataset = _data(variances)
        dataset.model = _model()
        prepared = MultiFitter.for_experiments([dataset]).prepare(objective=objective)
        original = {key: value.copy() for key, value in prepared.original[0].items()}
        fitted = {'x': prepared.x[0].copy(), 'y': prepared.y[0].copy(), 'weights': prepared.weights[0].copy()}

        for array in (dataset.x, dataset.y, dataset.ye):
            np.asarray(array)[:] = 123.0

        for key, value in original.items():
            np.testing.assert_array_equal(prepared.original[0][key], value)
        np.testing.assert_array_equal(prepared.x[0], fitted['x'])
        np.testing.assert_array_equal(prepared.y[0], fitted['y'])
        np.testing.assert_array_equal(prepared.weights[0], fitted['weights'])


class TestFinalize:
    def test_classical_metrics_come_from_the_measured_points(self):
        variances = np.full(Q.size, 1e-4)
        variances[[0, 1]] = 0.0
        model = _model()
        dataset = _data(variances)
        dataset.model = model
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run = MultiFitter.for_experiments([dataset], objective='hybrid').prepare()

        result = type('R', (), {'chi2': 1.0, 'reduced_chi': 0.5, 'n_pars': 1, 'x': run.fitted[0]['x']})()
        metrics = run.finalize([result])[0]

        assert metrics['n_classical_points'] == Q.size - 2
        assert metrics['objective_chi2'] == 1.0

    def test_all_zero_variance_has_no_classical_reduced_chi(self):
        model = _model()
        dataset = _data(np.zeros(Q.size))
        dataset.model = model
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            run = MultiFitter.for_experiments([dataset], objective='hybrid').prepare()
        result = type('R', (), {'chi2': 1.0, 'reduced_chi': 0.5, 'n_pars': 1, 'x': run.fitted[0]['x']})()
        assert run.finalize([result])[0]['classical_reduced_chi'] is None

    def test_recorded_metrics_reach_the_fitter(self):
        project = Project()
        project.default_model()
        dataset = _data()
        dataset.model = project.models[0]
        run = project.prepare_fit([dataset])
        results = run.execute()
        project.fitter.record_fit_results(results, run.finalize(results))
        assert project.fitter.classical_chi2 is not None
        assert project.fitter.classical_reduced_chi is not None


def test_a_prepared_run_keeps_its_constraints_owner_alive():
    project = Project()
    project.default_model()
    dataset = _data()
    dataset.model = project.models[0]
    run = project.prepare_fit([dataset])
    gc.collect()
    # Only the run is held; constraints resolve through the owner at execution.
    assert run.core_fitter._constraints_owner() is not None
