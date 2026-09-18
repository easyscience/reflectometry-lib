# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the user-facing constraint helpers
"""

import json

import pytest
from easyscience import global_object
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easyreflectometry import clamp_sum_partners
from easyreflectometry import constrain
from easyreflectometry import constrain_equal
from easyreflectometry import constrain_to_sum
from easyreflectometry import derived_parameter
from easyreflectometry import is_constrained_to_sum
from easyreflectometry import restore_sum_partners
from easyreflectometry import unconstrain
from easyreflectometry.project import Project
from easyreflectometry.sample import Material


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


class TestConstrainEqual:
    def test_ties_value_and_follows(self):
        leader = Parameter('leader', 5.0, unit='angstrom', min=0.0, max=10.0)
        follower = Parameter('follower', 1.0, unit='angstrom', min=0.0, max=2.0)

        constrain_equal(follower, to=leader)

        assert follower.independent is False
        assert follower.value == 5.0
        leader.value = 7.0
        assert follower.value == 7.0

    def test_overwrites_bounds_and_clears_fixed(self):
        leader = Parameter('leader', 5.0, unit='angstrom', min=0.0, max=10.0)
        follower = Parameter('follower', 1.0, unit='angstrom', min=0.5, max=2.0, fixed=True)

        constrain_equal(follower, to=leader)

        assert follower.min == leader.min
        assert follower.max == leader.max
        assert follower.fixed is False

    def test_dependent_setters_are_locked(self):
        leader = Parameter('leader', 5.0)
        follower = Parameter('follower', 1.0)
        constrain_equal(follower, to=leader)

        with pytest.raises(AttributeError):
            follower.value = 3.0
        with pytest.raises(AttributeError):
            follower.fixed = True

    def test_descriptor_dependency_gives_degenerate_bounds(self):
        # A DescriptorNumber dependency evaluates to a non-Parameter, so
        # EasyScience sets min == max == value on the dependent. Pinned
        # here because it is why the docs say "reset bounds after
        # unconstrain".
        leader = DescriptorNumber('leader', 3.0)
        follower = Parameter('follower', 1.0, min=0.0, max=2.0)

        constrain_equal(follower, to=leader)
        assert follower.min == 3.0
        assert follower.max == 3.0

        unconstrain(follower)
        assert follower.min == 3.0
        assert follower.max == 3.0


class TestConstrain:
    def test_scale_expression_follows(self):
        leader = Parameter('leader', 5.0, unit='angstrom', min=0.0, max=10.0)
        follower = Parameter('follower', 1.0, unit='angstrom', min=0.0, max=2.0)

        constrain(follower, '2 * t', t=leader)

        assert follower.value == 10.0
        leader.value = 6.0
        assert follower.value == 12.0

    def test_multi_parameter_expression(self):
        fraction = Parameter('fraction', 0.25, min=0.0, max=1.0)
        sld_a = Parameter('sld_a', 2.0)
        sld_b = Parameter('sld_b', 6.0)
        mixed = Parameter('mixed', 0.0)

        constrain(mixed, 'frac * a + (1 - frac) * b', frac=fraction, a=sld_a, b=sld_b)

        assert mixed.value == pytest.approx(0.25 * 2.0 + 0.75 * 6.0)
        fraction.value = 0.5
        assert mixed.value == pytest.approx(0.5 * 2.0 + 0.5 * 6.0)

    def test_reconstrain_replaces_dependency(self):
        first = Parameter('first', 1.0)
        second = Parameter('second', 2.0)
        follower = Parameter('follower', 0.0)

        constrain_equal(follower, to=first)
        constrain_equal(follower, to=second)
        assert follower.value == 2.0

        # No stale updates from the previous target
        first.value = 100.0
        assert follower.value == 2.0
        second.value = 3.0
        assert follower.value == 3.0

    def test_unknown_name_raises_and_reverts(self):
        leader = Parameter('leader', 5.0)
        follower = Parameter('follower', 1.0)

        with pytest.raises(NameError):
            constrain(follower, 'a + b', a=leader)

        assert follower.independent is True
        assert follower.value == 1.0


class TestUnconstrain:
    def test_removes_constraint_and_keeps_last_value(self):
        leader = Parameter('leader', 5.0, min=0.0, max=10.0)
        follower = Parameter('follower', 1.0, min=0.0, max=2.0)
        constrain_equal(follower, to=leader)
        leader.value = 7.0

        unconstrain(follower)

        assert follower.independent is True
        assert follower.value == 7.0
        # Bounds and fixed state are not restored
        assert follower.min == 0.0
        assert follower.max == 10.0
        assert follower.fixed is False
        # Fittable again
        follower.value = 1.5
        assert follower.value == 1.5
        assert leader.value == 7.0

    def test_idempotent_on_independent_parameter(self):
        parameter = Parameter('parameter', 1.0)
        unconstrain(parameter)
        unconstrain(parameter)
        assert parameter.independent is True


class TestIsConstrainedToSum:
    def _sum(self):
        a = Parameter('a', 30.0, unit='angstrom', min=0.0, max=100.0)
        b = Parameter('b', 30.0, unit='angstrom', min=0.0, max=100.0)
        constrain_to_sum(b, [a, b])
        return a, b

    def test_detects_the_sum_dependency(self):
        a, b = self._sum()
        assert is_constrained_to_sum(b) is True
        assert is_constrained_to_sum(b, [a, b]) is True
        assert is_constrained_to_sum(b, [a]) is True  # `parameter` itself may be omitted

    def test_independent_parameter_is_not(self):
        parameter = Parameter('p', 1.0)
        assert is_constrained_to_sum(parameter) is False

    def test_other_dependencies_are_not(self):
        leader = Parameter('leader', 5.0)
        follower = Parameter('follower', 1.0)
        constrain_equal(follower, to=leader)
        assert is_constrained_to_sum(follower) is False

    def test_foreign_partner_requirement_fails(self):
        a, b = self._sum()
        stranger = Parameter('stranger', 1.0)
        assert is_constrained_to_sum(b, [a, stranger]) is False

    def test_survives_project_reload(self):
        source = Project()
        source.default_model()
        sample = source.models[0].sample
        first = sample[1].layers[0].thickness
        second = sample[2].layers[0].thickness
        constrain_to_sum(second, [first, second])
        assert is_constrained_to_sum(second, [first, second])

        project_dict = json.loads(json.dumps(source.as_dict()))
        global_object.map._clear()
        project = Project()
        project.from_dict(project_dict)
        sample = project.models[0].sample
        assert is_constrained_to_sum(sample[2].layers[0].thickness, [sample[1].layers[0].thickness])


class TestSumPartnerClamp:
    def test_clamp_narrows_and_restore_widens(self):
        a = Parameter('a', 30.0, unit='angstrom', min=0.0, max=1000.0)
        b = Parameter('b', 30.0, unit='angstrom', min=0.0, max=1000.0)
        constrain_to_sum(b, [a, b])

        clamp_sum_partners([a], float(b.value))
        # a's headroom is the whole budget: value * (1 + remainder/occupied)
        assert float(a.max) == pytest.approx(60.0)
        a.value = 1e6
        assert float(a.value) == pytest.approx(60.0)
        assert float(b.value) == pytest.approx(0.0)

        unconstrain(b)
        restore_sum_partners([a])
        assert float(a.max) == pytest.approx(1000.0)

    def test_restore_is_idempotent_and_never_narrows(self):
        a = Parameter('a', 30.0, unit='angstrom', min=0.0, max=1000.0)
        b = Parameter('b', 30.0, unit='angstrom', min=0.0, max=1000.0)
        constrain_to_sum(b, [a, b])
        clamp_sum_partners([a], float(b.value))
        unconstrain(b)
        a.max = 2000.0  # user widened beyond the original in the meantime
        restore_sum_partners([a])
        restore_sum_partners([a])
        assert float(a.max) == pytest.approx(2000.0)

    def test_already_tight_bound_is_left_alone(self):
        a = Parameter('a', 30.0, unit='angstrom', min=0.0, max=40.0)
        b = Parameter('b', 30.0, unit='angstrom', min=0.0, max=1000.0)
        constrain_to_sum(b, [a, b])
        clamp_sum_partners([a], float(b.value))
        assert float(a.max) == pytest.approx(40.0)
        restore_sum_partners([a])
        assert float(a.max) == pytest.approx(40.0)

    def test_backups_survive_project_reload(self):
        source = Project()
        source.default_model()
        sample = source.models[0].sample
        first = sample[1].layers[0].thickness
        second = sample[2].layers[0].thickness
        original_max = float(first.max)
        constrain_to_sum(second, [first, second])
        clamp_sum_partners([first], float(second.value))
        clamped_max = float(first.max)
        assert clamped_max < original_max

        project_dict = json.loads(json.dumps(source.as_dict()))
        assert 'sum_partner_max_backups' in project_dict
        global_object.map._clear()
        project = Project()
        project.from_dict(project_dict)
        sample = project.models[0].sample
        first = sample[1].layers[0].thickness
        second = sample[2].layers[0].thickness

        # The clamp survives the round trip, and removing the constraint
        # afterwards still hands back the original bound.
        assert float(first.max) == pytest.approx(clamped_max)
        unconstrain(second)
        restore_sum_partners([first])
        assert float(first.max) == pytest.approx(original_max)

    def test_no_backups_key_when_nothing_is_clamped(self):
        project = Project()
        project.default_model()
        assert 'sum_partner_max_backups' not in project.as_dict()


class TestProjectRoundTrip:
    def test_constraint_survives_as_dict_from_dict(self):
        # Requires the easyscience serializer to route nested Parameters through
        # ``Parameter.as_dict`` and park the dependency as pending on rebuild.
        src_project = Project()
        src_project._info['name'] = 'Test'
        src_project.default_model()
        src_project._with_experiments = False
        sample = src_project.models[0].sample
        leader = sample[1].layers[0].thickness
        follower = sample[2].layers[0].thickness
        constrain(follower, '2 * t', t=leader)
        assert follower.value == 2 * leader.value

        project_dict = src_project.as_dict()
        global_object.map._clear()

        project = Project()
        project.from_dict(project_dict)
        sample = project.models[0].sample
        leader = sample[1].layers[0].thickness
        follower = sample[2].layers[0].thickness

        assert follower.independent is False
        leader.value = 60.0  # within the default [50, 200] thickness limits
        assert follower.value == 120.0

    def test_constrain_to_sum_with_numeric_total_survives(self):
        """The object-less constant built for an explicit total is embedded by value."""
        source = Project()
        source.default_model()
        sample = source.models[0].sample
        constrain_to_sum(
            sample[2].layers[0].roughness,
            [sample[1].layers[0].roughness, sample[2].layers[0].roughness],
            total=10.0,
        )
        project_dict = json.loads(json.dumps(source.as_dict()))
        global_object.map._clear()

        project = Project()
        project.from_dict(project_dict)
        sample = project.models[0].sample
        project.models[0].sample[1].layers[0].roughness.value = 4.0
        assert sample[2].layers[0].roughness.value == 6.0

    def test_constraint_against_a_derived_parameter_survives(self):
        """`Model.total_thickness` is reachable by path, so it can be a dependency."""
        source = Project()
        source.default_model()
        model = source.models[0]
        constrain(model.sample[1].layers[0].roughness, 'total / 100', total=model.total_thickness)
        project_dict = json.loads(json.dumps(source.as_dict()))
        global_object.map._clear()

        project = Project()
        project.from_dict(project_dict)
        model = project.models[0]
        roughness = model.sample[1].layers[0].roughness
        assert roughness.independent is False
        assert roughness.value == model.total_thickness.value / 100

    def test_unconstrain_does_not_resurrect_on_reload(self):
        source = Project()
        source.default_model()
        sample = source.models[0].sample
        follower = sample[2].layers[0].thickness
        constrain(follower, '2 * t', t=sample[1].layers[0].thickness)
        unconstrain(follower)

        project_dict = json.loads(json.dumps(source.as_dict()))
        assert 'parameter_constraints' not in project_dict
        global_object.map._clear()

        project = Project()
        project.from_dict(project_dict)
        assert project.models[0].sample[2].layers[0].thickness.independent is True

    def test_raw_make_independent_does_not_resurrect_either(self):
        """The marker alone is not enough: the parameter must still be dependent."""
        source = Project()
        source.default_model()
        sample = source.models[0].sample
        follower = sample[2].layers[0].thickness
        constrain(follower, '2 * t', t=sample[1].layers[0].thickness)
        follower.make_independent()  # bypasses `unconstrain`, so the marker survives

        assert 'parameter_constraints' not in source.as_dict()

    def test_unreachable_dependency_raises_rather_than_freezing(self):
        """Embedding a live parameter by value would silently kill the dependency."""
        project = Project()
        project.default_model()
        detached = Parameter('detached', 5.0, unit='angstrom')
        constrain(project.models[0].sample[1].layers[0].roughness, 'a', a=detached)

        with pytest.raises(ValueError, match='not reachable from'):
            project.as_dict()

    def test_unreachable_target_raises_rather_than_being_dropped(self):
        """A constraint *on* an off-model parameter is refused, not silently lost.

        The material is saved through `materials_not_in_model`, so without this
        the file would come back looking complete while the constraint had
        quietly disappeared.
        """
        project = Project()
        project.default_model()
        off_model = Material(sld=1.0, isld=0.0, name='OffModel')
        project.add_material(off_model)
        constrain(off_model.sld, '2 * s', s=project.models[0].sample[0].layers[0].material.sld)

        assert project.parameter_path(off_model.sld) is None
        with pytest.raises(ValueError, match='not reachable from'):
            project.as_dict(include_materials_not_in_model=True)

    def test_standalone_derived_parameter_is_session_only(self):
        """A `derived_parameter` belongs to no model: it has no path, and a
        constraint depending on it cannot be saved (documented limitation)."""
        project = Project()
        project.default_model()
        sample = project.models[0].sample
        total = derived_parameter('total', 'a + b', a=sample[1].layers[0].thickness, b=sample[2].layers[0].thickness)

        assert project.parameter_path(total) is None
        constrain(sample[2].layers[0].roughness, 'T / 10', T=total)
        with pytest.raises(ValueError, match='not reachable from'):
            project.as_dict()

    def test_warns_when_dependencies_are_embedded_in_the_parameters(self):
        """A file from a core that serializes dependencies in-place cannot be restored."""
        project = Project()
        project.default_model()
        project_dict = project.as_dict()
        # Mimic the shape such a core writes for a dependent nested parameter.
        project_dict['models']['data'][0]['scale']['_dependency_string'] = 'a'
        global_object.map._clear()

        with pytest.warns(UserWarning, match='must be re-applied'):
            Project().from_dict(project_dict)

    def test_chained_constraints_survive_and_still_follow(self):
        """Records are restored in tree order, which need not be dependency order."""
        source = Project()
        source.default_model()
        sample = source.models[0].sample
        root = sample[1].layers[0].roughness
        middle = sample[2].layers[0].roughness
        leaf = sample[2].layers[0].thickness
        # leaf <- middle <- root, i.e. the chain runs against the tree order.
        constrain(middle, '2 * r', r=root)
        constrain(leaf, '10 * m', m=middle)

        project_dict = json.loads(json.dumps(source.as_dict()))
        global_object.map._clear()

        project = Project()
        project.from_dict(project_dict)
        sample = project.models[0].sample
        root = sample[1].layers[0].roughness
        middle = sample[2].layers[0].roughness
        leaf = sample[2].layers[0].thickness

        assert middle.independent is False
        assert leaf.independent is False
        root.value = 3.0
        assert middle.value == 6.0
        assert leaf.value == 60.0
