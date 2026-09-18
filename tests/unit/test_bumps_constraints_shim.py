# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the BUMPS inequality-constraints enforcement
"""

import numpy as np
import pytest
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting.minimizers import minimizer_bumps

from easyreflectometry import _bumps_constraints
from easyreflectometry.data import DataSet1D
from easyreflectometry.inequality_constraints import InequalitySpec
from easyreflectometry.model import Model
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


def _two_layer_project():
    air = Material(0.0, 0.0, 'Air')
    film = Material(4.0, 0.0, 'Film')
    substrate = Material(2.047, 0.0, 'Si')
    sample = Sample(
        Multilayer(Layer(air, 0.0, 0.0, 'Superphase')),
        Multilayer(Layer(film, 40.0, 3.0, 'A')),
        Multilayer(Layer(film, 40.0, 3.0, 'B')),
        Multilayer(Layer(substrate, 0.0, 3.0, 'Subphase')),
    )
    model = Model(sample=sample)
    project = Project()
    project.default_model()
    project.models[0] = model
    model.interface = project._calculator
    return project, model


@pytest.fixture
def unpatched_modules():
    """`install()` mutates the core modules and has no undo; keep these tests independent.

    Unwraps a shim left behind by an earlier test so the assertions start from
    BUMPS' own driver, and puts that state back afterwards.
    """
    originals = []
    for module in (minimizer_bumps,):
        driver = module.FitDriver
        while getattr(driver, '_easyreflectometry_shim', False):
            driver = driver.__wrapped__
        originals.append((module, driver))
        module.FitDriver = driver
    yield
    for module, driver in originals:
        module.FitDriver = driver


@pytest.mark.usefixtures('unpatched_modules')
class TestInstall:
    def test_patches_the_consumer_namespace_and_is_idempotent(self, monkeypatch):
        """The consumer binds ``FitDriver`` at import, so its namespace is patched."""
        monkeypatch.setattr(minimizer_bumps, 'FitDriver', minimizer_bumps.FitDriver)

        _bumps_constraints.install()
        patched = minimizer_bumps.FitDriver
        assert getattr(patched, '_easyreflectometry_shim', False)

        _bumps_constraints.install()
        assert minimizer_bumps.FitDriver is patched


@pytest.mark.slow
class TestShimAgainstARealFit:
    def test_the_patched_entry_point_is_the_one_a_fit_calls(self):
        """A signature check would not catch patching the wrong namespace."""
        project, model = _two_layer_project()
        project.minimizer = AvailableMinimizers.Bumps
        layers = [layer for assembly in model.sample for layer in assembly.layers]
        q = np.linspace(0.01, 0.3, 50)
        reflectivity = model.interface.fit_func(q, model.unique_name)
        for layer in layers:
            for parameter in (layer.thickness, layer.roughness, layer.material.sld, layer.material.isld):
                parameter.fixed = True
        layers[1].thickness.fixed = False

        calls = []

        def factory(bumps_parameters):
            calls.append(dict(bumps_parameters))
            return []

        dataset = DataSet1D(name='sim', x=q, y=reflectivity, ye=(0.05 * reflectivity) ** 2)
        project.fitter.fit_single_data_set_1d(dataset, constraints_factory=factory)

        assert len(calls) == 1
        assert any(name.startswith('p') for name in calls[0])

    def test_infeasible_start_point_warns(self):
        """This is what `model_reset()` in the shim buys; without it there is no warning."""
        project, model = _two_layer_project()
        project.minimizer = AvailableMinimizers.Bumps
        layers = [layer for assembly in model.sample for layer in assembly.layers]
        thickness_a, thickness_b = layers[1].thickness, layers[2].thickness
        q = np.linspace(0.01, 0.3, 50)
        reflectivity = model.interface.fit_func(q, model.unique_name)
        for layer in layers:
            for parameter in (layer.thickness, layer.roughness, layer.material.sld, layer.material.isld):
                parameter.fixed = True
        thickness_a.fixed = thickness_b.fixed = False
        model.scale.fixed = model.background.fixed = True
        thickness_a.value, thickness_b.value = 40.0, 60.0  # sum 100, outside the budget

        paths = (project.parameter_path(thickness_a), project.parameter_path(thickness_b))
        project.add_inequality_constraint(InequalitySpec('a + b', '<', '90', {'a': paths[0], 'b': paths[1]}, {}, name='budget'))
        dataset = DataSet1D(name='sim', x=q, y=reflectivity, ye=(0.05 * reflectivity) ** 2)

        with pytest.warns(UserWarning, match=r'Unsatisfied constraints: \[budget fails\]'):
            project.fitter.fit_single_data_set_1d(dataset)

    def test_tolerates_a_model_with_no_free_parameters(self):
        """`curve.pars` is empty then; the shim must not be the thing that breaks."""
        project, model = _two_layer_project()
        project.minimizer = AvailableMinimizers.Bumps
        for parameter in project.parameters:
            if parameter.independent:
                parameter.fixed = True

        seen = []
        q = np.linspace(0.01, 0.3, 50)
        reflectivity = model.interface.fit_func(q, model.unique_name)
        dataset = DataSet1D(name='sim', x=q, y=reflectivity, ye=(0.05 * reflectivity) ** 2)

        with pytest.raises(Exception) as error:
            project.fitter.fit_single_data_set_1d(dataset, constraints_factory=lambda pars: seen.append(dict(pars)) or [])

        assert seen == [{}]
        # Whatever bumps does with an empty problem, it is not an AttributeError
        # from the shim reaching into a Curve that has no parameters.
        assert not isinstance(error.value, AttributeError)


class TestExtendKeepsThePenalty:
    def test_extend_re_enters_the_constraints_context(self):
        """Without the wrapper a continued chain samples an unpenalised posterior."""
        active = []

        class FakeSampler:
            def extend(self, **kwargs):
                active.append(_bumps_constraints._active.get())
                return 'extended'

        sampler = FakeSampler()
        from easyreflectometry.fitting import MultiFitter

        def factory(bumps_parameters):
            return []

        MultiFitter._keep_constraints_on_extend(sampler, factory)
        assert sampler.extend(additional_samples=10) == 'extended'
        assert active == [factory]
        # The context is released again afterwards.
        assert _bumps_constraints._active.get() is None

    def test_no_wrapper_without_constraints(self):
        from easyreflectometry.fitting import MultiFitter

        class FakeSampler:
            def extend(self):
                return None

        sampler = FakeSampler()
        MultiFitter._keep_constraints_on_extend(sampler, None)
        # Nothing shadowed the class's own method.
        assert 'extend' not in sampler.__dict__


class TestEngineRejection:
    def test_non_bumps_engine_is_rejected(self):
        """The message is also printed in the constraints tutorial; keep it verbatim."""
        project, model = _two_layer_project()
        project.minimizer = AvailableMinimizers.LMFit
        q = np.linspace(0.01, 0.3, 50)
        reflectivity = model.interface.fit_func(q, model.unique_name)
        dataset = DataSet1D(name='sim', x=q, y=reflectivity, ye=(0.05 * reflectivity) ** 2)
        path = project.parameter_path([lay for a in model.sample for lay in a.layers][1].thickness)
        project.add_inequality_constraint(InequalitySpec('a', '<', '90', {'a': path}, {}))

        with pytest.raises(ValueError) as error:
            project.fitter.fit_single_data_set_1d(dataset)

        assert str(error.value) == (
            "Inequality constraints (constraints_factory) require the BUMPS engine; the selected minimizer uses 'lmfit'."
        )


class TestTheCoreIsNeverHandedTheKeyword:
    """``constraints_factory`` is this library's own; the core knows nothing of it.

    ``Bumps.fit`` takes ``**kwargs`` and would accept — and drop — an
    unrecognised keyword without a word, so passing one through would fit an
    unconstrained problem silently. These pin the enforcement to this library.
    """

    def test_the_core_does_not_accept_it(self):
        import inspect

        assert 'constraints_factory' not in inspect.signature(minimizer_bumps.Bumps.fit).parameters

    def test_a_constrained_fit_does_not_pass_it_down(self, monkeypatch):
        project, model = _two_layer_project()
        project.minimizer = AvailableMinimizers.Bumps
        layers = [layer for assembly in model.sample for layer in assembly.layers]
        thickness = layers[1].thickness
        q = np.linspace(0.01, 0.3, 50)
        reflectivity = model.interface.fit_func(q, model.unique_name)
        for layer in layers:
            for parameter in (layer.thickness, layer.roughness, layer.material.sld, layer.material.isld):
                parameter.fixed = True
        thickness.fixed = False
        model.scale.fixed = model.background.fixed = True
        dataset = DataSet1D(name='sim', x=q, y=reflectivity, ye=(0.05 * reflectivity) ** 2)
        project.add_inequality_constraint(InequalitySpec('a', '<', '90', {'a': project.parameter_path(thickness)}, {}))

        seen = []
        original = minimizer_bumps.Bumps.fit

        def recording_fit(self, *args, **kwargs):
            seen.append(kwargs)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(minimizer_bumps.Bumps, 'fit', recording_fit)
        project.fitter.fit_single_data_set_1d(dataset)

        assert seen, 'the minimizer was never reached'
        assert all('constraints_factory' not in kwargs for kwargs in seen)


class TestExplicitFactoryOnTheRawFitter:
    """The GUI drives ``easy_science_multi_fitter.fit`` directly (see `for_experiments`).

    An explicit ``constraints_factory`` on that call used to be handed to the
    core, which drops unrecognised keywords through ``**kwargs`` — so the
    penalties were silently lost on exactly the path that asked for them.
    """

    def _fitter_and_arrays(self):
        from easyreflectometry.fitting import MultiFitter

        project, model = _two_layer_project()
        layers = [layer for assembly in model.sample for layer in assembly.layers]
        q = np.linspace(0.01, 0.3, 50)
        reflectivity = model.interface.fit_func(q, model.unique_name)
        for layer in layers:
            for parameter in (layer.thickness, layer.roughness, layer.material.sld, layer.material.isld):
                parameter.fixed = True
        layers[1].thickness.fixed = False
        model.scale.fixed = model.background.fixed = True
        dataset = DataSet1D(name='sim', x=q, y=reflectivity, ye=(0.05 * reflectivity) ** 2, model=model, auto_background=False)
        fitter = MultiFitter.for_experiments([dataset])
        fitter.easy_science_multi_fitter.switch_minimizer(AvailableMinimizers.Bumps)
        weights = 1.0 / np.sqrt(np.asarray(dataset.ye))
        return fitter, ([np.asarray(dataset.x)], [np.asarray(dataset.y)], [weights])

    def test_explicit_factory_is_enforced_not_swallowed(self):
        fitter, (x, y, weights) = self._fitter_and_arrays()
        calls = []

        fitter.easy_science_multi_fitter.fit(
            x, y, weights=weights, constraints_factory=lambda pars: calls.append(dict(pars)) or []
        )

        assert len(calls) == 1
        assert any(name.startswith('p') for name in calls[0])

    def test_explicit_factory_wins_over_the_provider(self):
        fitter, (x, y, weights) = self._fitter_and_arrays()
        provider_calls, explicit_calls = [], []
        fitter.constraints_factory_provider = lambda: lambda pars: provider_calls.append(1) or []

        fitter.easy_science_multi_fitter.fit(
            x, y, weights=weights, constraints_factory=lambda pars: explicit_calls.append(1) or []
        )

        assert explicit_calls and not provider_calls
