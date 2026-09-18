# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for derived read-only parameters: helpers, ``Model.total_thickness`` and assembly toggles."""

import json

import pytest
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry import Project
from easyreflectometry.constraints import constrain_to_sum
from easyreflectometry.constraints import derived_parameter
from easyreflectometry.model import Model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


def _thickness(value, name):
    return Parameter(name, value, unit='angstrom', min=0.0, max=1000.0)


class TestDerivedParameter:
    def test_follows_expression_and_is_read_only(self):
        a, b = _thickness(10.0, 'a'), _thickness(20.0, 'b')
        total = derived_parameter('total', 'a + b', a=a, b=b)

        assert total.value == 30.0
        assert str(total.unit) == 'Å'
        assert total.independent is False
        a.value = 15.0
        assert total.value == 35.0
        with pytest.raises(AttributeError):
            total.value = 1.0

    def test_requires_at_least_one_dependency(self):
        with pytest.raises(ValueError):
            derived_parameter('total', '1 + 1')

    def test_can_drive_other_dependencies(self):
        a, b, c = _thickness(10.0, 'a'), _thickness(20.0, 'b'), _thickness(0.0, 'c')
        total = derived_parameter('total', 'a + b', a=a, b=b)
        c.make_dependent_on('t * 2', {'t': total})
        b.value = 30.0
        assert c.value == 80.0


class TestConstrainToSum:
    def test_absorbs_remainder_against_numeric_total(self):
        a, b = _thickness(40.0, 'a'), _thickness(60.0, 'b')
        constrain_to_sum(b, [a, b], total=120.0)
        assert b.value == 80.0
        a.value = 50.0
        assert b.value == 70.0
        assert b.independent is False

    def test_default_total_is_current_sum(self):
        a, b, c = _thickness(10.0, 'a'), _thickness(20.0, 'b'), _thickness(30.0, 'c')
        constrain_to_sum(c, [a, b, c])
        assert c.value == 30.0
        a.value = 25.0
        assert c.value == 15.0

    def test_total_can_be_a_parameter(self):
        a, b, total = _thickness(10.0, 'a'), _thickness(20.0, 'b'), _thickness(100.0, 'T')
        constrain_to_sum(b, [a], total=total)
        assert b.value == 90.0
        total.value = 50.0
        assert b.value == 40.0

    def test_rejects_nothing_to_constrain_against(self):
        a = _thickness(10.0, 'a')
        with pytest.raises(ValueError):
            constrain_to_sum(a, [a])
        with pytest.raises(TypeError):
            constrain_to_sum(a, [_thickness(1.0, 'b')], total='12')


def _model_with_film(*thicknesses):
    sample = Sample(populate_if_none=False)
    sample.add_assembly(Multilayer(Layer(Material(0.0, 0.0, 'air'), thickness=0.0, roughness=0.0, name='air')))
    for index, thickness in enumerate(thicknesses):
        sample.add_assembly(
            Multilayer(Layer(Material(2.0, 0.0, f'm{index}'), thickness=thickness, roughness=2.0, name=f'L{index}'))
        )
    sample.add_assembly(Multilayer(Layer(Material(2.07, 0.0, 'Si'), thickness=0.0, roughness=2.0, name='Si')))
    return Model(sample=sample)


class TestModelTotalThickness:
    def test_sums_film_layers_only(self):
        model = _model_with_film(40.0, 60.0)
        total = model.total_thickness
        assert total.value == 100.0
        assert total.independent is False
        assert total not in model.get_fit_parameters()
        with pytest.raises(AttributeError):
            total.value = 5.0

    def test_tracks_edits_and_structure_changes(self):
        model = _model_with_film(40.0, 60.0)
        model.sample[1].layers[0].thickness.value = 45.0
        assert model.total_thickness.value == 105.0

        # layer appended inside an assembly: no notification path exists, the
        # property re-derives on access
        model.sample[2].layers.append(Layer(Material(1.0, 0.0, 'x'), thickness=10.0, roughness=1.0, name='X'))
        assert model.total_thickness.value == 115.0

        model.remove_assembly(1)
        assert model.total_thickness.value == 70.0

    def test_no_film_gives_zero_and_independent(self):
        model = _model_with_film()
        assert model.total_thickness.value == 0.0
        assert model.total_thickness.independent is True
        assert model.total_thickness.fixed is True
        model.add_assemblies(Multilayer(Layer(Material(1.0, 0.0, 'x'), thickness=10.0, roughness=1.0, name='X')))
        # the new assembly became the last layer (subphase); the former Si (0 A) is now film
        assert model.total_thickness.value == 0.0
        assert model.total_thickness.independent is False

    def test_emptied_film_does_not_leak_a_free_fit_parameter(self):
        """Becoming dependent clears `fixed`; losing the film must put it back.

        Otherwise the parameter returns to the fit as a free variable sitting at
        zero, and a fitter would happily vary it.
        """
        model = _model_with_film(40.0)
        assert model.total_thickness.independent is False

        model.remove_assembly(1)
        total = model.total_thickness
        assert total.value == 0.0
        assert total.independent is True
        assert total.fixed is True
        assert 'total_thickness' not in [parameter.name for parameter in model.get_fit_parameters()]

    def test_not_serialized_but_rebuilt_on_load(self):
        project = Project()
        project.default_model()
        model = project.models[0]
        before = model.total_thickness.value
        project_dict = json.loads(json.dumps(project.as_dict()))
        assert 'total_thickness' not in json.dumps(project_dict)

        global_object.map._clear()
        reloaded = Project()
        reloaded.from_dict(project_dict)
        assert reloaded.models[0].total_thickness.value == before
        assert reloaded.models[0].total_thickness.independent is False


class TestAssemblyConformalToggles:
    def _assembly(self):
        layers = [
            Layer(Material(1.0, 0.0, f'm{i}'), thickness=10.0 * (i + 1), roughness=float(i + 1), name=f'L{i}') for i in range(3)
        ]
        return Multilayer(layers)

    def test_conformal_thickness(self):
        assembly = self._assembly()
        assert assembly.conformal_thickness is False
        assembly.conformal_thickness = True
        assert assembly.conformal_thickness is True
        assert [layer.thickness.value for layer in assembly.layers] == [10.0, 10.0, 10.0]
        assembly.layers[0].thickness.value = 25.0
        assert assembly.layers[2].thickness.value == 25.0
        assembly.conformal_thickness = False
        assert assembly.conformal_thickness is False
        assert assembly.layers[1].thickness.independent is True

    def test_conformal_roughness(self):
        assembly = self._assembly()
        assert assembly.conformal_roughness is False
        assembly.conformal_roughness = True
        assert assembly.conformal_roughness is True
        assembly.layers[0].roughness.value = 7.0
        assert [layer.roughness.value for layer in assembly.layers] == [7.0, 7.0, 7.0]
        assembly.conformal_roughness = False
        assert assembly.layers[2].roughness.independent is True

    def test_single_layer_assembly_is_never_conformal(self):
        assembly = Multilayer(Layer(Material(1.0, 0.0, 'm'), thickness=1.0, roughness=1.0, name='L'))
        assert assembly.conformal_thickness is False
        assembly.conformal_thickness = True
        assert assembly.conformal_thickness is False
