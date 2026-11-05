# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

from easyscience import global_object
from numpy.testing import assert_almost_equal

from easyreflectometry.sample.elements.materials.material import Material
from easyreflectometry.sample.elements.materials.material_mixture import MaterialMixture


class TestMaterialMixture:
    def test_default(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        assert str(material_mixture._sld.unit) == '1/Å^2'
        assert str(material_mixture._isld.unit) == '1/Å^2'

    def test_default_constraint(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        material_mixture.material_a.sld.value = 0
        material_mixture.material_b.isld.value = -1
        assert_almost_equal(material_mixture.sld, 2.093)
        assert_almost_equal(material_mixture.isld, -0.5)
        assert str(material_mixture._sld.unit) == '1/Å^2'
        assert str(material_mixture._isld.unit) == '1/Å^2'

    def test_fraction_constraint(self):
        p = Material()
        q = Material(6.908, -0.278, 'Boron')
        material_mixture = MaterialMixture(p, q, 0.2)
        assert material_mixture.fraction.value == 0.2
        assert_almost_equal(material_mixture.sld, 4.7304)
        assert_almost_equal(material_mixture.isld, -0.0556)
        material_mixture._fraction.value = 0.5
        assert material_mixture.fraction.value == 0.5
        assert_almost_equal(material_mixture.sld, 5.54700)
        assert_almost_equal(material_mixture.isld, -0.1390)

    def test_material_a_change(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        q = Material(6.908, -0.278, 'Boron')
        material_mixture.material_a = q
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 5.54700)
        assert_almost_equal(material_mixture.isld, -0.1390)

    def test_material_b_change(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        q = Material(6.908, -0.278, 'Boron')
        material_mixture.material_b = q
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 5.54700)
        assert_almost_equal(material_mixture.isld, -0.1390)

    def test_material_b_change_double(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        q = Material(6.908, -0.278, 'Boron')
        material_mixture.material_b = q
        assert material_mixture.name == 'EasyMaterial/Boron'
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 5.54700)
        assert_almost_equal(material_mixture.isld, -0.1390)
        r = Material(0.00, 0.00, 'ACMW')
        material_mixture.material_b = r
        assert material_mixture.name == 'EasyMaterial/ACMW'
        assert material_mixture.fraction.value == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 2.0930)
        assert_almost_equal(material_mixture.isld, 0.0000)

    def test_from_pars(self):
        p = Material()
        q = Material(6.908, -0.278, 'Boron')
        material_mixture = MaterialMixture(p, q, 0.2)
        assert material_mixture.fraction.value == 0.2
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.7304)
        assert_almost_equal(material_mixture.isld, -0.0556)
        assert str(material_mixture._sld.unit) == '1/Å^2'
        assert str(material_mixture._isld.unit) == '1/Å^2'

    def test_dict_repr(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture._dict_repr == {
            'EasyMaterial/EasyMaterial': {
                'fraction': '0.500 dimensionless',
                'sld': '4.186e-6 1/Å^2',
                'isld': '0.000e-6 1/Å^2',
                'material_a': {'EasyMaterial': {'sld': '4.186e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
                'material_b': {'EasyMaterial': {'sld': '4.186e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
            }
        }

    def test_dict_round_trip(self) -> None:
        # When
        p = MaterialMixture()
        p_dict = p.as_dict()
        global_object.registry.clear()

        # Then
        q = MaterialMixture.from_dict(p_dict)

        # Expect
        assert sorted(p.as_dict()) == sorted(q.as_dict())

    def test_update_name(self) -> None:
        # When
        material_mixture = MaterialMixture()
        mock_material_a = MagicMock()
        mock_material_a.name = 'name_a'
        material_mixture._material_a = mock_material_a
        mock_material_b = MagicMock()
        mock_material_b.name = 'name_b'
        material_mixture._material_b = mock_material_b

        # Then
        material_mixture._update_name()

        # Expect
        assert material_mixture.name == 'name_a/name_b'

    def test_calculator_binding_uses_mixed_sld(self) -> None:
        """Regression: the calculator wrapper must bind to the mixture's own
        ``_sld``/``_isld`` (the weighted average), not to either child material's
        sld/isld parameter. Without an explicit ``_get_linkable_attributes``
        override the inherited dir-walk picks up the first matching child
        parameter and the wrapper silently gets the wrong SLD.
        """
        from easyreflectometry.calculators import CalculatorFactory

        interface = CalculatorFactory()
        material_a = Material(sld=2.0, isld=0.0)
        material_b = Material(sld=6.0, isld=0.0)
        mixture = MaterialMixture(material_a, material_b, fraction=0.25, interface=interface)

        # 2 * 0.75 + 6 * 0.25 = 1.5 + 1.5 = 3.0
        assert_almost_equal(mixture.sld, 3.0)
        wrapper_material = interface()._wrapper.storage['material'][mixture.unique_name]
        assert_almost_equal(wrapper_material.real.value, 3.0)
        assert_almost_equal(wrapper_material.imag.value, 0.0)

    def test_mutation_propagates_after_round_trip(self) -> None:
        """Regression: after ``from_dict`` swaps in the saved ``_fraction``
        Parameter, the dependency graph for ``_sld``/``_isld`` must point at
        the live ``_fraction`` (not the temp Parameter created from the
        float kwarg in ``__init__``)."""
        p = MaterialMixture(Material(sld=2.0), Material(sld=6.0), fraction=0.25)
        p_dict = p.as_dict()
        global_object.registry.clear()

        q = MaterialMixture.from_dict(p_dict)
        assert_almost_equal(q.sld, 3.0)

        q.fraction = 0.8
        # 2 * 0.2 + 6 * 0.8 = 0.4 + 4.8 = 5.2
        assert_almost_equal(q.sld, 5.2)
