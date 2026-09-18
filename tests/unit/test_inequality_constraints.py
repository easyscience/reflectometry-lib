# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cross-parameter inequality constraints (specs, translation, project integration)."""

import json

import numpy as np
import pytest
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers
from easyscience.variable import Parameter

from easyreflectometry import Project
from easyreflectometry.constraints import constrain
from easyreflectometry.constraints import derived_parameter
from easyreflectometry.data import DataSet1D
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.inequality_constraints import InequalitySpec
from easyreflectometry.inequality_constraints import UnitError
from easyreflectometry.inequality_constraints import build_constraints_factory
from easyreflectometry.inequality_constraints import check_units
from easyreflectometry.inequality_constraints import evaluate_spec
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


class _FakeBumpsParameter:
    def __init__(self, value):
        self.value = value


def _resolver(mapping):
    return lambda path: mapping[path]


# --------------------------------------------------------------------------- spec


class TestInequalitySpec:
    def test_normalizes_relation_aliases(self):
        assert InequalitySpec('a', '≤', 'b', {'a': 'x'}, {'b': 'y'}).op == '<='
        assert InequalitySpec('a', '≥', 'b', {'a': 'x'}, {'b': 'y'}).op == '>='

    def test_rejects_unknown_relation(self):
        with pytest.raises(ValueError, match='Unsupported relation'):
            InequalitySpec('a', '==', 'b', {'a': 'x'}, {'b': 'y'})

    def test_rejects_unmapped_identifiers(self):
        with pytest.raises(ValueError, match='unmapped names: c'):
            InequalitySpec('a + c', '<', 'b', {'a': 'x'}, {'b': 'y'})

    def test_rejects_empty_side_and_bad_syntax(self):
        with pytest.raises(ValueError, match='cannot be empty'):
            InequalitySpec('', '<', 'b', {}, {'b': 'y'})
        with pytest.raises(SyntaxError):
            InequalitySpec('a +', '<', 'b', {'a': 'x'}, {'b': 'y'})

    def test_numeric_rhs_and_math_symbols_need_no_mapping(self):
        spec = InequalitySpec('sqrt(a) * pi', '<', '90', {'a': 'x'}, {})
        assert spec.rhs_paths == {}

    def test_round_trip_dict(self):
        spec = InequalitySpec('a + b', '<=', 'c', {'a': 'p/a', 'b': 'p/b'}, {'c': 'p/c'}, name='n', enabled=False)
        restored = InequalitySpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert restored == spec
        assert str(restored) == 'a + b <= c'

    def test_rejects_alias_mapped_to_different_paths_on_the_two_sides(self):
        # `paths` merges both sides; without the check the right side would silently win.
        with pytest.raises(ValueError, match='different parameters'):
            InequalitySpec('a', '<', 'a + b', {'a': 'p/x'}, {'a': 'p/y', 'b': 'p/z'})
        # The same alias for the same parameter on both sides is fine.
        spec = InequalitySpec('a', '<', 'a + b', {'a': 'p/x'}, {'a': 'p/x', 'b': 'p/z'})
        assert spec.paths == {'a': 'p/x', 'b': 'p/z'}


# --------------------------------------------------------------------------- translation


class TestTranslation:
    def _params(self):
        a = Parameter('a', 10.0, unit='angstrom', min=0, max=100)
        b = Parameter('b', 20.0, unit='angstrom', min=0, max=100)
        c = Parameter('c', 5.0, unit='angstrom', min=0, max=100, fixed=True)
        return a, b, c

    def test_operands_read_bumps_values_not_easyscience_values(self):
        a, b, c = self._params()
        spec = InequalitySpec('a + b', '<', '25', {'a': 'pa', 'b': 'pb'}, {})
        factory = build_constraints_factory([spec], _resolver({'pa': a, 'pb': b}))
        bumps_a, bumps_b = _FakeBumpsParameter(10.0), _FakeBumpsParameter(20.0)
        (constraint,) = factory({'p' + a.unique_name: bumps_a, 'p' + b.unique_name: bumps_b})

        assert float(constraint) == pytest.approx(5.0)  # 30 - 25, linear violation
        bumps_a.value = 1.0  # optimizer trial point: EasyScience `a` is still 10
        assert a.value == 10.0
        assert float(constraint) == 0.0
        assert str(constraint) == 'a + b < 25'

    def test_fixed_parameters_are_constants(self):
        a, b, c = self._params()
        spec = InequalitySpec('a', '>', 'c', {'a': 'pa'}, {'c': 'pc'})
        factory = build_constraints_factory([spec], _resolver({'pa': a, 'pc': c}))
        (constraint,) = factory({'p' + a.unique_name: _FakeBumpsParameter(3.0)})  # c not in the problem
        assert float(constraint) == pytest.approx(2.0)  # 5 - 3

    def test_dependent_parameters_are_expanded_to_free_leaves(self):
        a, b, c = self._params()
        total = derived_parameter('total', 'x + y + z', x=a, y=b, z=c)
        half = Parameter('half', 0.0, unit='angstrom', min=-1e6, max=1e6)
        constrain(half, 't / 2', t=total)  # dependent on a dependent
        spec = InequalitySpec('a', '<', 'h', {'a': 'pa'}, {'h': 'ph'})
        factory = build_constraints_factory([spec], _resolver({'pa': a, 'ph': half}))
        bumps_a, bumps_b = _FakeBumpsParameter(10.0), _FakeBumpsParameter(20.0)
        (constraint,) = factory({'p' + a.unique_name: bumps_a, 'p' + b.unique_name: bumps_b})

        # half = (10 + 20 + 5) / 2 = 17.5 > a = 10 -> satisfied
        assert float(constraint) == 0.0
        bumps_b.value = 0.0  # half = 7.5 < a = 10 -> violated by 2.5, read from the trial vector
        assert float(constraint) == pytest.approx(2.5)

    def test_disabled_specs_give_no_factory(self):
        a, b, c = self._params()
        spec = InequalitySpec('a', '<', 'b', {'a': 'pa'}, {'b': 'pb'}, enabled=False)
        assert build_constraints_factory([spec], _resolver({'pa': a, 'pb': b})) is None
        assert build_constraints_factory([], _resolver({})) is None

    @pytest.mark.parametrize(
        'op, lhs, rhs, violation',
        [('<', 3.0, 5.0, 0.0), ('<', 5.0, 5.0, 0.0), ('<', 7.0, 5.0, 2.0), ('>', 3.0, 5.0, 2.0), ('>=', 6.0, 5.0, 0.0)],
    )
    def test_evaluate_spec_violation(self, op, lhs, rhs, violation):
        a = Parameter('a', lhs, min=-100, max=100)
        b = Parameter('b', rhs, min=-100, max=100)
        spec = InequalitySpec('a', op, 'b', {'a': 'pa'}, {'b': 'pb'})
        result = evaluate_spec(spec, _resolver({'pa': a, 'pb': b}))
        assert result.lhs == lhs and result.rhs == rhs
        assert result.violation == pytest.approx(violation)
        assert result.satisfied is (violation == 0.0)

    def test_check_units(self):
        a, b, c = self._params()
        sld = Parameter('sld', 2.0, unit='1/angstrom**2', min=-10, max=10)
        check_units(
            InequalitySpec('a + b', '<', 'c', {'a': 'pa', 'b': 'pb'}, {'c': 'pc'}),
            _resolver({'pa': a, 'pb': b, 'pc': c}),
        )
        check_units(InequalitySpec('a', '<', '90', {'a': 'pa'}, {}), _resolver({'pa': a}))
        with pytest.raises(ValueError, match='Incompatible units'):
            check_units(InequalitySpec('a', '<', 's', {'a': 'pa'}, {'s': 'ps'}), _resolver({'pa': a, 'ps': sld}))

    def test_check_units_raises_typed_unit_error(self):
        # A typed exception so callers do not have to match message substrings;
        # it subclasses ValueError, so the assertions above stay valid too.
        a, _, _ = self._params()
        sld = Parameter('sld_t', 2.0, unit='1/angstrom**2', min=-10, max=10)
        with pytest.raises(UnitError):
            check_units(InequalitySpec('a', '<', 's', {'a': 'pa'}, {'s': 'ps'}), _resolver({'pa': a, 'ps': sld}))

    def test_check_units_mixed_literals_fall_back_to_numeric(self):
        # '90 - b' cannot be evaluated with units (a literal has none); it is
        # checked numerically and its literals read in the other side's unit.
        a, b, c = self._params()
        check_units(InequalitySpec('a', '<', '90 - b', {'a': 'pa'}, {'b': 'pb'}), _resolver({'pa': a, 'pb': b}))
        # broken syntax is still rejected (at spec construction)
        with pytest.raises(SyntaxError):
            InequalitySpec('a', '<', '90 - b +', {'a': 'pa'}, {'b': 'pb'})


# --------------------------------------------------------------------------- project integration


def _two_layer_project():
    project = Project()
    project.default_model()
    model = project.models[0]
    film_a = Multilayer(Layer(Material(3.0, 0.0, 'A'), thickness=40.0, roughness=3.0, name='A'), name='A')
    film_b = Multilayer(Layer(Material(5.0, 0.0, 'B'), thickness=60.0, roughness=3.0, name='B'), name='B')
    substrate = model.sample[-1]
    model.remove_assembly(len(model.sample) - 1)
    model.remove_assembly(len(model.sample) - 1)
    model.add_assemblies(film_a, film_b, substrate)
    return project, model


class TestProjectPaths:
    def test_parameter_path_round_trip(self):
        project, model = _two_layer_project()
        t_a = model.sample[1].layers[0].thickness
        path = project.parameter_path(t_a)
        assert path == 'models/0/sample/1/layers/0/thickness'
        assert project.resolve_parameter_path(path) is t_a
        assert project.parameter_path(model.scale) == 'models/0/scale'
        assert project.parameter_path(model.total_thickness) == 'models/0/total_thickness'
        sld_path = project.parameter_path(model.sample[1].layers[0].material.sld)
        assert sld_path == 'models/0/sample/1/layers/0/material/sld'
        assert project.resolve_parameter_path(sld_path) is model.sample[1].layers[0].material.sld

    def test_unreachable_parameter_and_bad_paths(self):
        project, model = _two_layer_project()
        assert project.parameter_path(Parameter('loose', 1.0)) is None
        with pytest.raises(KeyError):
            project.resolve_parameter_path('models/0/sample/99/layers/0/thickness')
        with pytest.raises(KeyError):
            project.resolve_parameter_path('models/0/nope')
        with pytest.raises(KeyError):
            project.resolve_parameter_path('models/0/sample')  # not a parameter
        with pytest.raises(KeyError):
            project.resolve_parameter_path('models/0/_sample/0')  # private attributes are off limits


class TestProjectInequalities:
    def test_registry_validation_and_persistence(self):
        project, model = _two_layer_project()
        t_a = model.sample[1].layers[0].thickness
        t_b = model.sample[2].layers[0].thickness
        pa, pb = project.parameter_path(t_a), project.parameter_path(t_b)
        project.add_inequality_constraint(InequalitySpec('a', '<', 'b', {'a': pa}, {'b': pb}, name='order'))
        project.add_inequality_constraint(InequalitySpec('a + b', '<', '90', {'a': pa, 'b': pb}, {}, name='budget'))
        with pytest.raises(ValueError, match='Incompatible units'):
            project.add_inequality_constraint(
                InequalitySpec('a', '<', 's', {'a': pa}, {'s': project.parameter_path(model.scale)})
            )
        assert [s.name for s in project.violated_inequality_constraints()] == ['budget']

        project_dict = json.loads(json.dumps(project.as_dict()))
        assert len(project_dict['inequality_constraints']) == 2
        global_object.map._clear()
        reloaded = Project()
        reloaded.from_dict(project_dict)
        assert [str(s) for s in reloaded.inequality_constraints] == ['a < b', 'a + b < 90']
        evaluations = reloaded.evaluate_inequality_constraints()
        assert [e.satisfied for e in evaluations] == [True, False]

        reloaded.remove_inequality_constraint('budget')
        assert [s.name for s in reloaded.inequality_constraints] == ['order']
        reloaded.remove_inequality_constraint(0)
        assert reloaded.inequality_constraints == []
        assert 'inequality_constraints' not in reloaded.as_dict()

    def test_old_project_files_without_inequalities_load(self):
        project, _ = _two_layer_project()
        project_dict = project.as_dict()
        project_dict.pop('inequality_constraints', None)
        global_object.map._clear()
        reloaded = Project()
        reloaded.from_dict(project_dict)
        assert reloaded.inequality_constraints == []


@pytest.mark.slow
class TestInequalityFit:
    def test_bumps_fit_respects_inequality_and_lmfit_is_rejected(self):
        project, model = _two_layer_project()
        project.minimizer = AvailableMinimizers.Bumps
        layers = [layer for assembly in model.sample for layer in assembly.layers]
        t_a, t_b = layers[1].thickness, layers[2].thickness
        q = np.linspace(0.01, 0.3, 150)
        t_a.value, t_b.value = 45.0, 55.0  # truth sums to 100
        r_true = model.interface.fit_func(q, model.unique_name)
        t_a.value, t_b.value = 30.0, 50.0  # feasible start
        for layer in layers:
            for par in (layer.thickness, layer.roughness, layer.material.sld, layer.material.isld):
                par.fixed = True
        t_a.fixed = False
        t_b.fixed = False
        model.scale.fixed = True
        model.background.fixed = True
        pa, pb = project.parameter_path(t_a), project.parameter_path(t_b)
        project.add_inequality_constraint(InequalitySpec('a + b', '<', '90', {'a': pa, 'b': pb}, {}, name='budget'))
        project.add_inequality_constraint(InequalitySpec('a', '<', 'b', {'a': pa}, {'b': pb}, name='order'))
        dataset = DataSet1D(name='sim', x=q, y=r_true, ye=(0.05 * r_true) ** 2)

        result = project.fitter.fit_single_data_set_1d(dataset)

        assert result.success
        assert t_a.value + t_b.value <= 90.0 + 1e-3
        assert t_a.value + t_b.value == pytest.approx(90.0, abs=0.05)  # lands on the boundary
        assert t_a.value <= t_b.value + 1e-3

        project.minimizer = AvailableMinimizers.LMFit
        with pytest.raises(ValueError, match='require the BUMPS engine'):
            project.fitter.fit_single_data_set_1d(dataset)

    def test_for_experiments_raw_fit_applies_project_inequalities(self):
        """The documented GUI path — `for_experiments` then driving the raw
        `easy_science_multi_fitter.fit(...)` — must apply the project's
        inequality constraints (and refuse non-BUMPS engines) instead of
        silently fitting an unconstrained problem."""
        project, model = _two_layer_project()
        layers = [layer for assembly in model.sample for layer in assembly.layers]
        t_a, t_b = layers[1].thickness, layers[2].thickness
        q = np.linspace(0.01, 0.3, 150)
        t_a.value, t_b.value = 45.0, 55.0  # truth sums to 100
        r_true = model.interface.fit_func(q, model.unique_name)
        t_a.value, t_b.value = 30.0, 50.0  # feasible start
        for layer in layers:
            for par in (layer.thickness, layer.roughness, layer.material.sld, layer.material.isld):
                par.fixed = True
        t_a.fixed = False
        t_b.fixed = False
        model.scale.fixed = True
        model.background.fixed = True
        pa, pb = project.parameter_path(t_a), project.parameter_path(t_b)
        project.add_inequality_constraint(InequalitySpec('a + b', '<', '90', {'a': pa, 'b': pb}, {}, name='budget'))
        dataset = DataSet1D(name='sim', x=q, y=r_true, ye=(0.05 * r_true) ** 2, model=model, auto_background=False)

        fitter = MultiFitter.for_experiments([dataset], constraints_factory_provider=project.build_constraints_factory)
        fitter.easy_science_multi_fitter.switch_minimizer(AvailableMinimizers.Bumps)
        weights = 1.0 / np.sqrt(np.asarray(dataset.ye))
        results = fitter.easy_science_multi_fitter.fit([np.asarray(dataset.x)], [np.asarray(dataset.y)], weights=[weights])

        assert results[0].success
        assert t_a.value + t_b.value <= 90.0 + 1e-3  # unconstrained optimum (100) is refused

        fitter.easy_science_multi_fitter.switch_minimizer(AvailableMinimizers.LMFit)
        with pytest.raises(ValueError, match='require the BUMPS engine'):
            fitter.easy_science_multi_fitter.fit([np.asarray(dataset.x)], [np.asarray(dataset.y)], weights=[weights])
