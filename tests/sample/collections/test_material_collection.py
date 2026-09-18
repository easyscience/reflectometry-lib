# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for LayerCollection class.
"""

import pytest
from easyscience import global_object

from easyreflectometry.sample.collections.material_collection import MaterialCollection
from easyreflectometry.sample.elements.materials.material import Material
from easyreflectometry.sample.elements.materials.material_density import MaterialDensity


class TestMaterialCollection:
    def test_default(self):
        p = MaterialCollection()
        assert p.name == 'EasyMaterials'
        assert p.interface is None
        assert len(p) == 3
        assert p[0].name == 'Air'
        assert p[1].name == 'D2O'
        assert p[2].name == 'Si'

    def test_dont_populate(self):
        p = MaterialCollection(populate_if_none=False)
        assert p.name == 'EasyMaterials'
        assert p.interface is None
        assert len(p) == 0

    def test_from_pars(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = MaterialCollection(m, k, name='thinBoron')
        assert p.name == 'thinBoron'
        assert p.interface is None
        assert len(p) == 2
        assert p[0].name == 'Boron'
        assert p[1].name == 'Potassium'

    def test_empty_list(self):
        p = MaterialCollection([])
        assert p.name == 'EasyMaterials'
        assert p.interface is None
        assert len(p) == 0

    def test_dict_repr(self):
        p = MaterialCollection()
        assert p._dict_repr == {
            'EasyMaterials': [
                {'Air': {'isld': '0.000e-6 1/Å^2', 'sld': '0.000e-6 1/Å^2'}},
                {'D2O': {'isld': '0.000e-6 1/Å^2', 'sld': '6.335e-6 1/Å^2'}},
                {'Si': {'isld': '0.000e-6 1/Å^2', 'sld': '2.074e-6 1/Å^2'}},
            ]
        }

    def test_repr(self):
        p = MaterialCollection()
        p.__repr__()
        assert (
            p.__repr__()
            == 'EasyMaterials:\n- Air:\n    sld: 0.000e-6 1/Å^2\n    isld: 0.000e-6 1/Å^2\n- D2O:\n    sld: 6.335e-6 1/Å^2\n    isld: 0.000e-6 1/Å^2\n- Si:\n    sld: 2.074e-6 1/Å^2\n    isld: 0.000e-6 1/Å^2\n'  # noqa: E501
        )

    def test_dict_round_trip(self):
        # When
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = MaterialCollection()
        p.insert(0, m)
        p.add_material(k)
        p_dict = p.as_dict()
        global_object.map._clear()

        # Then
        q = MaterialCollection.from_dict(p_dict)

        # Expect
        assert sorted(p.as_dict()) == sorted(q.as_dict())

    def test_add_material(self):
        # When
        p = MaterialCollection()
        m = Material(6.908, -0.278, 'Boron')

        # Then
        p.add_material()
        p.add_material(m)

        # Expect
        assert p[4] == m

    def test_duplicate_material(self):
        # When
        p = MaterialCollection()
        m = Material(6.908, -0.278, 'Boron')
        p.add_material(m)

        # Then
        p.duplicate_material(3)

        # Expect
        assert p[4].name == 'Boron duplicate'

    def test_duplicate_material_of_a_subclass(self):
        # When
        p = MaterialCollection()
        m = MaterialDensity('Si', 2.33, 'Silicon')
        p.add_material(m)

        # Then
        p.duplicate_material(3)

        # Expect
        # The copy has to come back as its own class: rebuilding it as `Material` raised,
        # so duplicating a density material was not a lossy copy but an outright failure.
        assert isinstance(p[4], MaterialDensity)
        assert p[4].name == 'Silicon duplicate'
        assert p[4].chemical_structure == 'Si'
        assert p[4].density.value == 2.33
        assert p[4].sld.value == pytest.approx(p[3].sld.value)
