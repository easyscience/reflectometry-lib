# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Does LMFit reach the attainable answer? See ``MD/LMFIT_CONVERGENCE.md``.

The real-data fixtures are representative, not the case the original
complaint was made on; passing them does not close that complaint.
Reference costs and how they were produced are in ``reference.md``.
"""

import os
import warnings

import numpy as np
import pytest
from easyscience import Minimizers
from easyscience import global_object

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.data.measurement import load
from easyreflectometry.data.measurement import load_as_dataset
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

STATIC = os.path.join(os.path.dirname(__file__), '..', '..', '_static')
PATHS = {'example': os.path.join(STATIC, 'example.ort'), 'e361': os.path.join(STATIC, 'e361r.txt')}
#: Classical chi2 references, see reference.md.
REFERENCE = {('example', 'model'): 1262.352, ('example', 'pointwise'): 1008.59, ('e361', 'model'): 380.506}
LEAST_SQUARES = [Minimizers.LMFit_leastsq, Minimizers.LMFit_scipy_least_squares]


@pytest.fixture(autouse=True)
def _clean_map():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def film_model(case: str) -> Model:
    """Si | SiO2 | film | D2O, with the free parameters and bounds of reference.md."""
    example = case == 'example'
    si, sio2 = Material(2.07, 0, 'Si'), Material(3.47, 0, 'SiO2')
    film, d2o = Material(2.0 if example else 1.0, 0, 'Film'), Material(6.36, 0, 'D2O')
    layers = [
        Layer(si, 0, 0, 'Si'),
        Layer(sio2, 30 if example else 10, 3 if example else 4, 'SiO2'),
        Layer(film, 250 if example else 200, 3, 'Film'),
        Layer(d2o, 0, 3, 'D2O'),
    ]
    model = Model(
        Sample(*[Multilayer(layer) for layer in layers]),
        1,
        1e-6 if example else 2e-5,
        PercentageFwhm(0.02 if example else 5.0),
        'Film model',
    )
    _, sio2_layer, film_layer, _ = layers
    if example:
        free = [
            (sio2_layer.thickness, 15, 50),
            (film_layer.thickness, 200, 300),
            (sio2_layer.roughness, 1, 15),
            (film_layer.roughness, 1, 15),
            (film.sld, 0.1, 3),
            (model.background, 1e-7, 1e-5),
            (model.scale, 0.5, 1.5),
        ]
    else:
        free = [
            (model.scale, 0.1, 2),
            (model.background, 0, 5e-5),
            (d2o.sld, 6, 6.36),
            (sio2_layer.thickness, 5, 20),
            (film_layer.thickness, 100, 220),
            (film.sld, 0.2, 1.5),
        ]
    for parameter, low, high in free:
        parameter.fixed = False
        parameter.min = low
        parameter.max = high
    model.interface = CalculatorFactory()
    return model


def classical_cost(model: Model, dataset: DataSet1D, resolution_function=None) -> float:
    x, y, variances = (np.asarray(a) for a in (dataset.x, dataset.y, dataset.ye))
    curve = model.interface.fit_func_for(resolution_function=resolution_function)(x, model.unique_name)
    keep = variances > 0
    return float(np.sum((y[keep] - curve[keep]) ** 2 / variances[keep]))


class TestSyntheticRecovery:
    """An identifiable problem with known truth: the only place parameter assertions are valid."""

    TRUTH = {'thickness': 180.0, 'sld': 2.5, 'roughness': 4.0}

    def _problem(self):
        model = film_model('e361')
        film_layer = model.sample[2].layers[0]
        for parameter in model.get_free_parameters():
            parameter.fixed = True
        truth = [film_layer.thickness, film_layer.material.sld, film_layer.roughness]
        for parameter, value, (low, high) in zip(truth, self.TRUTH.values(), [(100, 250), (1, 5), (1, 10)]):
            parameter.min, parameter.max = low, high  # first: a value outside the old bounds is clamped
            parameter.value = value
        q = np.linspace(0.01, 0.25, 150)
        exact = model.interface.fit_func(q, model.unique_name)
        rng = np.random.default_rng(1)
        sigma = 0.01 * exact
        data = DataSet1D(name='synthetic', x=q, y=exact + rng.normal(0, sigma), ye=sigma**2)
        for parameter in truth:
            parameter.fixed = False
            # 5 % off: within the basin; a 20 % thickness error shifts the fringes
            # by more than a period, into another (correctly found) minimum.
            parameter.value *= 1.05
        return model, data, truth

    @pytest.mark.parametrize('minimizer', LEAST_SQUARES)
    def test_parameters_are_recovered_within_three_sigma(self, minimizer):
        model, data, truth = self._problem()
        fitter = MultiFitter(model)
        fitter.switch_minimizer(minimizer)
        fitter.fit_single_data_set_1d(data)
        for parameter, expected in zip(truth, self.TRUTH.values()):
            assert abs(parameter.value - expected) <= 3 * parameter.error, parameter.name

    def test_reported_cost_matches_an_independent_recomputation(self):
        model, data, _ = self._problem()
        fitter = MultiFitter(model)
        result = fitter.fit_single_data_set_1d(data)
        assert result.chi2 == pytest.approx(classical_cost(model, data), rel=1e-9)

    def test_finite_differences_are_stable(self):
        # What leastsq's Jacobian relies on: forward differences over several
        # steps agree with a central difference.
        model, _, truth = self._problem()
        q = np.linspace(0.01, 0.25, 150)

        def reflectivity():
            return model.interface.fit_func(q, model.unique_name)

        for parameter in truth:
            value = parameter.value
            parameter.value = value + 1e-5 * value
            upper = reflectivity()
            parameter.value = value - 1e-5 * value
            lower = reflectivity()
            central = (upper - lower) / (2e-5 * value)
            parameter.value = value
            base = reflectivity()
            for factor in (1e-7, 1e-6, 1e-4):
                parameter.value = value + factor * value
                forward = (reflectivity() - base) / (factor * value)
                parameter.value = value
                scale = np.max(np.abs(central))
                assert np.max(np.abs(forward - central)) <= 1e-2 * scale, (parameter.name, factor)


@pytest.mark.parametrize('case', ['example', 'e361'])
def test_every_free_parameter_reaches_the_calculator(case):
    model = film_model(case)
    q = np.linspace(0.01, 0.25, 200)
    base = model.interface.fit_func(q, model.unique_name)
    for parameter in model.get_free_parameters():
        value = parameter.value
        step = max(1e-6 * abs(value), 1e-9)
        parameter.value = value + step if value + step <= parameter.max else value - step
        changed = model.interface.fit_func(q, model.unique_name)
        parameter.value = value
        assert np.max(np.abs(changed - base)) > 0, parameter.name


def test_leastsq_warns_when_a_parameter_starts_on_a_bound():
    model = film_model('e361')
    film_thickness = model.sample[2].layers[0].thickness
    film_thickness.value = film_thickness.max
    fitter = MultiFitter(model)
    with pytest.warns(UserWarning, match='start on a bound'):
        fitter.prepare(fitter._datagroup_inputs(load(PATHS['e361']))[1])
    fitter.switch_minimizer(Minimizers.LMFit_scipy_least_squares)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fitter.prepare(fitter._datagroup_inputs(load(PATHS['e361']))[1])


@pytest.mark.slow
class TestRealData:
    @pytest.mark.parametrize('minimizer', LEAST_SQUARES)
    @pytest.mark.parametrize('case', ['example', 'e361'])
    def test_notebook_path_reaches_the_reference(self, case, minimizer):
        model = film_model(case)
        fitter = MultiFitter(model)
        fitter.switch_minimizer(minimizer)
        fitter.fit(load(PATHS[case]))
        cost = classical_cost(model, load_as_dataset(PATHS[case]))
        assert cost <= 1.10 * REFERENCE[(case, 'model')]

    @pytest.mark.parametrize('minimizer', LEAST_SQUARES)
    @pytest.mark.parametrize('case', ['example', 'e361'])
    def test_app_path_reaches_the_reference(self, case, minimizer):
        # As the GUI fits: an experiment with its own resolution, prepared and executed.
        model = film_model(case)
        dataset = load_as_dataset(PATHS[case])
        dataset.model = model
        fitter = MultiFitter.for_experiments([dataset])
        fitter.switch_minimizer(minimizer)
        run = fitter.prepare()
        metrics = run.finalize(run.execute())[0]
        resolution = 'pointwise' if getattr(dataset, 'resolution_function', None) is not None else 'model'
        assert metrics['classical_chi2'] <= 1.10 * REFERENCE[(case, resolution)]

    @pytest.mark.parametrize('case', ['example', 'e361'])
    def test_least_squares_leaves_an_upper_bound_start(self, case):
        # Hypothesis 1: leastsq can stall on a bound (see the warning above);
        # the trust-region least_squares reaches the reference from there.
        model = film_model(case)
        parameter = model.sample[2].layers[0].thickness
        parameter.value = parameter.max
        fitter = MultiFitter(model)
        fitter.switch_minimizer(Minimizers.LMFit_scipy_least_squares)
        fitter.fit(load(PATHS[case]))
        assert classical_cost(model, load_as_dataset(PATHS[case])) <= 1.10 * REFERENCE[(case, 'model')]
