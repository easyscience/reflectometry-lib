# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""The project-owned fit configuration: lifetime, serialization and what reaches the engine."""

import contextlib
import os

import lmfit
import numpy as np
import pytest
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting.minimizers.utils import FitError

from easyreflectometry import fit_settings as fit_settings_module
from easyreflectometry.data import DataSet1D
from easyreflectometry.fit_settings import FitSettings
from easyreflectometry.fit_settings import requires_finite_bounds
from easyreflectometry.model import Model
from easyreflectometry.project import Project

PATH_STATIC = os.path.join(os.path.dirname(__file__), '_static')


@pytest.fixture(autouse=True)
def _clean_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


class _Stop(Exception):
    pass


@contextlib.contextmanager
def _stops_at_the_engine():
    """The block ends in a spy's ``_Stop``, as the easyscience adapter wraps it."""
    with pytest.raises(FitError) as error:
        yield
    assert isinstance(error.value.e, _Stop)


@pytest.fixture
def engine_kwargs(monkeypatch):
    """What lmfit's ``Model.fit`` receives; the fit stops there."""
    seen = {}

    def fake_fit(self, *args, **kwargs):
        seen.clear()
        seen['fit_kws'] = dict(kwargs['fit_kws'])
        seen['max_nfev'] = kwargs['max_nfev']
        raise _Stop

    monkeypatch.setattr(lmfit.Model, 'fit', fake_fit)
    return seen


def _dataset() -> DataSet1D:
    q = np.linspace(0.01, 0.2, 20)
    return DataSet1D(name='d', x=q, y=np.exp(-q * 30), ye=np.full(20, 1e-4))


class TestValidation:
    @pytest.mark.parametrize('value', [0, -1e-3, float('inf'), float('nan'), True, '1e-3'])
    def test_tolerance_must_be_finite_and_positive(self, value):
        with pytest.raises(ValueError):
            FitSettings(tolerance=value).validate()

    def test_dfo_tolerance_is_capped(self):
        with pytest.raises(ValueError, match='0.1'):
            FitSettings(minimizer=AvailableMinimizers.DFO_leastsq, tolerance=0.5).validate()

    @pytest.mark.parametrize('value', [0, -5, 2.5, True])
    def test_budget_must_be_a_positive_integer(self, value):
        with pytest.raises(ValueError):
            FitSettings(max_evaluations=value).validate()

    def test_sampling_needs_bumps(self):
        with pytest.raises(ValueError, match='BUMPS'):
            FitSettings(mode='sample').validate()
        FitSettings(minimizer=AvailableMinimizers.Bumps_simplex, mode='sample').validate()

    def test_auto_objective_is_valid(self):
        FitSettings(objective='auto').validate()
        with pytest.raises(ValueError, match='Unknown objective'):
            FitSettings(objective='nonsense').validate()

    @pytest.mark.parametrize(
        ('name', 'value'),
        [('tolerance', -1), ('max_evaluations', 0), ('mode', 'invalid'), ('mode', 'sample'), ('objective', 'nonsense')],
    )
    def test_in_place_edits_are_refused_when_a_fit_is_prepared(self, name, value):
        project = Project()
        project.default_model()
        dataset = _dataset()
        dataset.model = project.models[0]
        setattr(project.fit_settings, name, value)
        with pytest.raises(ValueError):
            project.prepare_fit([dataset])

    def test_an_engine_switch_can_invalidate_the_settings(self, engine_kwargs):
        project = Project()
        project.default_model()
        project.fit_settings.tolerance = 0.5  # fine for LMFit
        project.minimizer = AvailableMinimizers.DFO_leastsq
        with pytest.raises(ValueError, match='0.1'):
            project.fitter.fit_single_data_set_1d(_dataset())

    def test_minimizer_must_be_a_member(self):
        project = Project()
        with pytest.raises(ValueError, match='AvailableMinimizers'):
            project.minimizer = 'LMFit_leastsq'
        assert project.minimizer is fit_settings_module.DEFAULT_MINIMIZER

    def test_finite_bounds_only_for_differential_evolution(self):
        assert requires_finite_bounds(AvailableMinimizers.LMFit_differential_evolution)
        assert not requires_finite_bounds(AvailableMinimizers.LMFit_leastsq)
        assert not requires_finite_bounds(AvailableMinimizers.Bumps_simplex)


class TestConfigure:
    def test_writes_and_clears_the_generic_settings(self):
        project = Project()
        project.default_model()
        project.minimizer = AvailableMinimizers.Bumps_simplex
        core = project.fitter.easy_science_multi_fitter
        project.fit_settings.tolerance = 1e-4
        project.fit_settings.max_evaluations = 77
        project.fit_settings.configure(core)
        assert (core.tolerance, core.max_evaluations) == (1e-4, 77)

        project.fit_settings.tolerance = None
        project.fit_settings.max_evaluations = None
        project.fit_settings.configure(core)
        assert (core.tolerance, core.max_evaluations) == (None, None)

    def test_lmfit_tolerance_goes_per_call_under_the_method_keyword(self):
        # easyscience 2.5 would send a core-side tolerance to Powell/COBYLA as `ftol`.
        settings = FitSettings(tolerance=1e-4)
        project = Project()
        project.default_model()
        core = project.fitter.easy_science_multi_fitter
        settings.configure(core)
        assert core.tolerance is None
        assert settings.engine_kwargs() == {'minimizer_kwargs': {'ftol': 1e-4}}
        settings.minimizer = AvailableMinimizers.LMFit_powell
        assert settings.engine_kwargs() == {'minimizer_kwargs': {'tol': 1e-4}}
        settings.minimizer = AvailableMinimizers.LMFit_differential_evolution
        settings.set_option('seed', 1)
        assert settings.engine_kwargs() == {'minimizer_kwargs': {'seed': 1, 'tol': 1e-4}}

    def test_does_not_rebuild_an_unchanged_minimizer(self):
        project = Project()
        project.default_model()
        core = project.fitter.easy_science_multi_fitter
        minimizer = core.minimizer
        project.fit_settings.configure(core)
        assert core.minimizer is minimizer


class TestSerialization:
    def test_round_trip(self):
        settings = FitSettings(
            minimizer=AvailableMinimizers.Bumps_simplex,
            mode='sample',
            tolerance=1e-5,
            max_evaluations=300,
            objective='legacy_mask',
            engine_options={'LMFit_differential_evolution': {'seed': 3}, 'DFO_leastsq': {'rhobeg': 0.5}},
        )
        restored, report = FitSettings.from_dict(settings.to_dict())
        assert restored == settings
        assert report == []

    @pytest.mark.parametrize(
        ('alias', 'member'), [('LMFit', 'LMFit_leastsq'), ('Bumps', 'Bumps_simplex'), ('DFO', 'DFO_leastsq')]
    )
    def test_aliases_resolve_without_warning(self, alias, member):
        settings, report = FitSettings.from_dict({'minimizer': alias})
        assert settings.minimizer is AvailableMinimizers[member]
        assert report == []
        assert FitSettings(minimizer=AvailableMinimizers[alias]).minimizer is AvailableMinimizers[member]

    @pytest.mark.parametrize(('alias', 'option', 'value'), [('LMFit', 'epsfcn', 1e-5), ('DFO', 'rhobeg', 0.5)])
    def test_options_set_on_an_alias_survive_a_round_trip(self, alias, option, value):
        settings = FitSettings(minimizer=AvailableMinimizers[alias])
        settings.set_option(option, value)
        restored, report = FitSettings.from_dict(settings.to_dict())
        assert restored.engine_kwargs() == settings.engine_kwargs() != {}
        assert report == []

    def test_option_keys_saved_under_an_alias_are_normalised(self):
        settings, report = FitSettings.from_dict({
            'engine_options': {
                'LMFit_leastsq': {'epsfcn': 1e-6},
                'LMFit': {'epsfcn': 1e-5},
                'DFO': {'rhobeg': 0.5},
            },
        })
        # The member's own key wins over its alias's.
        assert settings.engine_options == {'LMFit_leastsq': {'epsfcn': 1e-6}, 'DFO_leastsq': {'rhobeg': 0.5}}
        assert report == []

    @pytest.mark.parametrize(
        'data',
        [
            None,
            [],
            {'minimizer': []},
            {'objective': []},
            {'mode': {}},
            {'engine_options': []},
            {'engine_options': {'LMFit_leastsq': []}},
        ],
    )
    def test_malformed_content_is_reported_not_raised(self, data):
        settings, report = FitSettings.from_dict(data)
        assert settings == FitSettings()
        assert len(report) == 1

    def test_a_trust_region_conflict_drops_rhobeg(self):
        settings, report = FitSettings.from_dict({
            'minimizer': 'DFO_leastsq',
            'tolerance': 1e-3,
            'engine_options': {'DFO_leastsq': {'rhobeg': 1e-3}},
        })
        assert settings.tolerance == 1e-3
        assert settings.engine_options == {}
        assert 'rhobeg' in report[0]

    def test_auto_objective_is_hybrid(self):
        settings, report = FitSettings.from_dict({'objective': 'auto'})
        assert settings.objective == 'hybrid'
        assert report == []

    def test_unknown_minimizer_falls_back_with_a_warning(self):
        settings, report = FitSettings.from_dict({'minimizer': 'NoSuchEngine'})
        assert settings.minimizer is fit_settings_module.DEFAULT_MINIMIZER
        assert 'NoSuchEngine' in report[0]

    def test_sampling_without_a_bumps_minimizer_falls_back(self):
        settings, report = FitSettings.from_dict({'minimizer': 'LMFit_leastsq', 'mode': 'sample'})
        assert settings.mode == 'minimize'
        assert report

    def test_invalid_values_become_engine_defaults(self):
        settings, report = FitSettings.from_dict({'tolerance': -1, 'max_evaluations': 'many'})
        assert settings.tolerance is None and settings.max_evaluations is None
        assert len(report) == 2

    def test_unknown_or_invalid_options_are_dropped_and_reported(self):
        settings, report = FitSettings.from_dict({
            'engine_options': {
                'Bumps_simplex': {'xtol': 1e-7},
                'LMFit_differential_evolution': {'popsize': 0},
                'LMFit_leastsq': {'epsfcn': 1e-6},
                'NoSuchEngine': {'a': 1},
            },
        })
        assert settings.engine_options == {'LMFit_leastsq': {'epsfcn': 1e-6}}
        assert len(report) == 3


class TestOptions:
    @pytest.mark.parametrize(
        'options',
        [
            {'popsize': 0},
            {'popsize': 2.5},
            {'popsize': True},
            {'mutation': 3.0},
            {'mutation': 2.0},
            {'mutation': float('inf')},
            {'strategy': 'nonsense'},
            {'seed': -1},
            {'seed': 2**32},
            {'epsfcn': 1e-6},
        ],
    )
    def test_invalid_options_are_rejected(self, options):
        settings = FitSettings(minimizer=AvailableMinimizers.LMFit_differential_evolution)
        with pytest.raises(ValueError):
            for name, value in options.items():
                settings.set_option(name, value)
        assert settings.engine_options == {}

    @pytest.mark.parametrize(
        ('minimizer', 'name', 'value'),
        [
            ('LMFit_differential_evolution', 'mutation', 0.0),
            ('LMFit_differential_evolution', 'mutation', 1.999),
            ('LMFit_differential_evolution', 'seed', 2**32 - 1),
            ('DFO_leastsq', 'rhobeg', 1e-9),
        ],
    )
    def test_values_next_to_an_excluded_bound_are_accepted(self, minimizer, name, value):
        FitSettings(minimizer=AvailableMinimizers[minimizer]).set_option(name, value)

    def test_rhobeg_must_be_positive(self):
        with pytest.raises(ValueError, match=r'\(0.0, None\]'):
            FitSettings(minimizer=AvailableMinimizers.DFO_leastsq).set_option('rhobeg', 0.0)

    def test_rhobeg_must_exceed_the_dfo_tolerance(self):
        settings = FitSettings(minimizer=AvailableMinimizers.DFO_leastsq, tolerance=1e-3)
        settings.set_option('rhobeg', 1e-3)
        with pytest.raises(ValueError, match='rhobeg'):
            settings.validate()
        settings.set_option('rhobeg', 2e-3)
        settings.validate()

    def test_bumps_has_no_options(self):
        assert fit_settings_module.option_schema(AvailableMinimizers.Bumps_simplex) == []

    def test_options_are_kept_per_minimizer(self):
        settings = FitSettings(minimizer=AvailableMinimizers.LMFit_cobyla)
        with pytest.raises(ValueError):
            settings.set_option('epsfcn', 1e-6)
        settings.minimizer = AvailableMinimizers.LMFit_leastsq
        settings.set_option('epsfcn', 1e-6)
        settings.minimizer = AvailableMinimizers.DFO_leastsq
        settings.set_option('rhobeg', 0.5)
        assert settings.engine_kwargs() == {'rhobeg': 0.5}
        settings.minimizer = AvailableMinimizers.LMFit_leastsq
        assert settings.engine_kwargs() == {'minimizer_kwargs': {'epsfcn': 1e-6}}
        settings.set_option('epsfcn', None)
        assert settings.engine_options == {'DFO_leastsq': {'rhobeg': 0.5}}

    def test_an_option_reaches_the_engine(self, engine_kwargs):
        project = Project()
        project.default_model()
        project.fit_settings.set_option('epsfcn', 1e-6)
        with _stops_at_the_engine():
            project.fitter.fit_single_data_set_1d(_dataset())
        assert engine_kwargs['fit_kws'] == {'epsfcn': 1e-6}


class TestProjectLifetime:
    def test_settings_reach_the_engine_through_every_rebuild(self, engine_kwargs):
        project = Project()
        project.default_model()
        # Edited before any fitter exists
        project.fit_settings.tolerance = 1e-4
        project.fit_settings.max_evaluations = 99
        with _stops_at_the_engine():
            project.fitter.fit_single_data_set_1d(_dataset())
        assert engine_kwargs == {'fit_kws': {'ftol': 1e-4}, 'max_nfev': 99}

        # A second model becomes current: a new fitter, same settings
        second = Model()
        second.interface = project.models[0].interface
        project.models.append(second)
        project.current_model_index = 1
        with _stops_at_the_engine():
            project.fitter.fit_single_data_set_1d(_dataset())
        assert engine_kwargs == {'fit_kws': {'ftol': 1e-4}, 'max_nfev': 99}

        # The calculator is switched: the fitter is dropped, the settings are not
        project.calculator = 'refl1d'
        with _stops_at_the_engine():
            project.fitter.fit_single_data_set_1d(_dataset())
        assert engine_kwargs == {'fit_kws': {'ftol': 1e-4}, 'max_nfev': 99}

        # Reset to the engine default: nothing is sent
        project.fit_settings.tolerance = None
        project.fit_settings.max_evaluations = None
        with _stops_at_the_engine():
            project.fitter.fit_single_data_set_1d(_dataset())
        assert engine_kwargs == {'fit_kws': {}, 'max_nfev': None}

    def test_loading_replaces_the_settings_and_drops_the_fitter(self):
        project = Project()
        project.default_model()
        project.fit_settings.tolerance = 1e-4
        project.fit_settings.objective = 'mighell'
        core = project.fitter.easy_science_multi_fitter
        project_dict = project.as_dict()
        project_dict['fit_settings'] = FitSettings(minimizer=AvailableMinimizers.Bumps_simplex).to_dict()
        global_object.map._clear()

        project.from_dict(project_dict)

        assert project.fit_settings == FitSettings(minimizer=AvailableMinimizers.Bumps_simplex)
        assert project.fitter.easy_science_multi_fitter is not core
        assert project.fitter.easy_science_multi_fitter.tolerance is None

    def test_switch_minimizer_on_the_project_fitter_switches_the_settings(self):
        project = Project()
        project.default_model()
        dataset = _dataset()
        dataset.model = project.models[0]
        fitter = project.fitter
        fitter.switch_minimizer(AvailableMinimizers.Bumps)
        assert project.minimizer is AvailableMinimizers.Bumps_simplex
        assert fitter.easy_science_multi_fitter.minimizer.enum is AvailableMinimizers.Bumps_simplex
        # A new hand-out of the fitter keeps it, and so do the runs.
        assert project.fitter.easy_science_multi_fitter.minimizer.enum is AvailableMinimizers.Bumps_simplex
        assert project.prepare_fit([dataset]).core_fitter.minimizer.enum is AvailableMinimizers.Bumps_simplex

    def test_malformed_settings_in_a_project_file_fall_back(self):
        project = Project()
        project.default_model()
        project_dict = project.as_dict()
        project_dict['fit_settings'] = {'minimizer': [], 'objective': [], 'engine_options': []}
        global_object.map._clear()
        project.from_dict(project_dict)
        assert project.fit_settings == FitSettings()
        assert len(project.load_report) == 3
        assert len(project.models) == 1

    def test_load_report_is_cleared_by_each_load(self):
        project = Project()
        project.default_model()
        project_dict = project.as_dict()
        project_dict['fit_settings'] = {'minimizer': 'NoSuchEngine'}
        global_object.map._clear()
        project.from_dict(project_dict)
        assert project.load_report

        project_dict['fit_settings'] = FitSettings().to_dict()
        global_object.map._clear()
        project.from_dict(project_dict)
        assert project.load_report == []

    def test_legacy_file_without_fit_settings(self):
        project = Project()
        project.default_model()
        project_dict = project.as_dict()
        del project_dict['fit_settings']
        project_dict['fitter_minimizer'] = 'Bumps'
        global_object.map._clear()
        project.from_dict(project_dict)
        assert project.minimizer is AvailableMinimizers.Bumps_simplex
        assert project.fit_settings.mode == 'minimize'

    def test_prepared_fit_uses_a_snapshot(self):
        project = Project()
        project.default_model()
        dataset = _dataset()
        dataset.model = project.models[0]
        project.fit_settings.tolerance = 1e-4

        prepared = project.prepare_fit([dataset])
        project.fit_settings.tolerance = 1e-2
        project.fit_settings.objective = 'legacy_mask'

        assert prepared.settings.tolerance == 1e-4
        assert prepared.call_kwargs() == {'minimizer_kwargs': {'ftol': 1e-4}}
        assert prepared.objective == 'hybrid'
        assert project.prepare_fit([dataset]).call_kwargs() == {'minimizer_kwargs': {'ftol': 1e-2}}

    def test_per_call_objective_wins_over_the_settings(self):
        project = Project()
        project.default_model()
        dataset = _dataset()
        dataset.model = project.models[0]
        project.fit_settings.objective = 'legacy_mask'
        assert project.prepare_fit([dataset]).objective == 'legacy_mask'
        assert project.prepare_fit([dataset], objective='mighell').objective == 'mighell'

    def test_prepare_fit_defaults_to_the_loaded_experiments(self):
        project = Project()
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'example.ort'))
        prepared = project.prepare_fit()
        assert len(prepared.fitted) == 1
        assert isinstance(prepared.channels, list)


class TestNativeBoundary:
    """What the engines receive, captured where they are called (easyscience 2.5 adapters)."""

    @staticmethod
    def _project(minimizer):
        project = Project()
        project.default_model()
        model = project.models[0]
        for parameter, low, high in ((model.scale, 0.5, 1.5), (model.background, 0.0, 1e-5)):
            parameter.fixed = False
            parameter.min, parameter.max = low, high
        project.minimizer = minimizer
        return project

    @pytest.mark.parametrize(
        'minimizer',
        [AvailableMinimizers.LMFit_powell, pytest.param(AvailableMinimizers.LMFit_cobyla, marks=pytest.mark.slow)],
    )
    def test_lmfit_scalar_methods_run_with_a_tolerance(self, minimizer):
        # Regression: on easyscience 2.5 a core-side tolerance crashes these.
        project = self._project(minimizer)
        project.fit_settings.tolerance = 1e-6
        result = project.fitter.fit_single_data_set_1d(_dataset())
        assert result.n_evaluations > 0

    def test_a_switch_on_the_project_fitter_is_the_engine_that_runs(self):
        from easyscience.fitting.minimizers.minimizer_bumps import Bumps

        project = self._project(AvailableMinimizers.LMFit_leastsq)
        project.fit_settings.max_evaluations = 5
        fitter = project.fitter
        fitter.switch_minimizer(AvailableMinimizers.Bumps_simplex)
        result = fitter.fit_single_data_set_1d(_dataset())
        assert result.minimizer_engine is Bumps

    def test_dfo_trust_region_reaches_dfols(self, monkeypatch):
        from easyscience.fitting.minimizers import minimizer_dfo

        seen = {}

        def fake_solve(*args, **kwargs):
            seen.update(kwargs)
            raise _Stop

        monkeypatch.setattr(minimizer_dfo.dfols, 'solve', fake_solve)
        project = self._project(AvailableMinimizers.DFO_leastsq)
        project.fit_settings.tolerance = 1e-3
        project.fit_settings.set_option('rhobeg', 0.25)
        with _stops_at_the_engine():
            project.fitter.fit_single_data_set_1d(_dataset())
        assert (seen['rhobeg'], seen['rhoend']) == (0.25, 1e-3)

    def test_differential_evolution_seed_is_reproducible(self):
        values = []
        for _ in range(2):
            global_object.map._clear()
            project = self._project(AvailableMinimizers.LMFit_differential_evolution)
            project.fit_settings.max_evaluations = 200
            project.fit_settings.set_option('seed', 7)
            project.fit_settings.set_option('popsize', 5)
            project.fitter.fit_single_data_set_1d(_dataset())
            values.append((project.models[0].scale.value, project.models[0].background.value))
        assert values[0] == values[1]
