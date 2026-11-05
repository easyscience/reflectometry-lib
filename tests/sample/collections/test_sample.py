# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Sample class.
"""

import pytest
from easyscience import global_object
from numpy.testing import assert_equal

from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerCollection
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import SurfactantLayer


class TestSample:
    def test_default(self):
        # When Then
        p = Sample()

        # Expect
        assert_equal(p.name, 'EasySample')
        assert_equal(p.interface, None)
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, 'EasyMultilayer')

    def test_dont_populate(self):
        p = Sample(populate_if_none=False)
        assert p.name == 'EasySample'
        assert p.interface is None
        assert len(p) == 0

    def test_add_assembly(self):
        # When
        p = Sample()
        surfactant = SurfactantLayer()

        # Then
        p.add_assembly()
        p.add_assembly(surfactant)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, 'EasyMultilayer')
        assert_equal(p[2].name, 'EasyMultilayer added')
        assert_equal(p[3].name, 'EasySurfactantLayer')

    # Problems with parameterized tests START
    def test_duplicate_assembly_multilayer(self):
        # When
        assembly_to_duplicate = Multilayer()
        p = Sample()
        p.add_assembly(assembly_to_duplicate)

        # Then
        p.duplicate_assembly(2)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, 'EasyMultilayer')
        assert_equal(p[2].name, assembly_to_duplicate.name)
        assert_equal(p[3].name, assembly_to_duplicate.name + ' duplicate')

    def test_duplicate_assembly_repeating_multilayer(self):
        # When
        assembly_to_duplicate = RepeatingMultilayer()
        p = Sample()
        p.add_assembly(assembly_to_duplicate)

        # Then
        p.duplicate_assembly(2)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, 'EasyMultilayer')
        assert_equal(p[2].name, assembly_to_duplicate.name)
        assert_equal(p[3].name, assembly_to_duplicate.name + ' duplicate')

    def test_duplicate_assembly_surfactant(self):
        # When
        assembly_to_duplicate = SurfactantLayer()
        p = Sample()
        p.add_assembly(assembly_to_duplicate)

        # Then
        p.duplicate_assembly(2)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, 'EasyMultilayer')
        assert_equal(p[2].name, assembly_to_duplicate.name)
        assert_equal(p[3].name, assembly_to_duplicate.name + ' duplicate')

    # Problems with parameterized tests END

    def test_move_assembly_up(self):
        # When
        p = Sample()
        surfactant = SurfactantLayer()
        p.add_assembly(surfactant)

        # Then
        p.move_up(2)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, surfactant.name)
        assert_equal(p[2].name, 'EasyMultilayer')

    def test_move_assembly_up_index_0(self):
        # When
        p = Sample()
        surfactant = SurfactantLayer()
        p.add_assembly(surfactant)

        # Then
        p.move_up(0)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, 'EasyMultilayer')
        assert_equal(p[2].name, surfactant.name)

    def test_move_assembly_down(self):
        # When
        p = Sample()
        surfactant = SurfactantLayer()
        p.add_assembly(surfactant)

        # Then
        p.move_down(1)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, surfactant.name)
        assert_equal(p[2].name, 'EasyMultilayer')

    def test_move_assembly_down_index_2(self):
        # When
        p = Sample()
        surfactant = SurfactantLayer()
        p.add_assembly(surfactant)

        # Then
        p.move_down(2)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, 'EasyMultilayer')
        assert_equal(p[2].name, surfactant.name)

    def test_remove_assembly(self):
        # When
        p = Sample()
        surfactant = SurfactantLayer()
        p.add_assembly(surfactant)

        # Then
        p.remove_assembly(1)

        # Expect
        assert_equal(p[0].name, 'EasyMultilayer')
        assert_equal(p[1].name, surfactant.name)

    def test_subphase(self):
        # When
        p = Sample()
        layer = Multilayer(Layer(name='new layer'))
        p.add_assembly(layer)

        # Then
        layer = p.subphase

        # Expect
        assert_equal(layer.name, 'new layer')

    def test_superphase(self):
        # When
        p = Sample()
        layer = Multilayer(Layer(name='new layer'))
        p.add_assembly(layer)
        p.move_up(2)
        p.move_up(1)

        # Then
        layer = p.superphase

        # Expect
        assert_equal(layer.name, 'new layer')

    def test_from_pars(self):
        # When
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')

        # Then
        d = Sample(o1, o2, name='myModel')

        # Expect
        assert_equal(d.name, 'myModel')
        assert_equal(d.interface, None)
        assert_equal(d[0].name, 'twoLayerItem1')
        assert_equal(d[1].name, 'oneLayerItem2')

    def test_from_pars_error(self):
        m1 = Material(6.908, -0.278, 'Boron')

        with pytest.raises(ValueError):
            _ = Sample(m1, name='myModel')

    def test_repr(self):
        p = Sample()
        assert (
            p.__repr__()
            == 'EasySample:\n- EasyMultilayer:\n    EasyLayerCollection:\n    - EasyLayer:\n        material:\n          EasyMaterial:\n            sld: 4.186e-6 1/Å^2\n            isld: 0.000e-6 1/Å^2\n        thickness: 10.000 Å\n        roughness: 3.300 Å\n- EasyMultilayer:\n    EasyLayerCollection:\n    - EasyLayer:\n        material:\n          EasyMaterial:\n            sld: 4.186e-6 1/Å^2\n            isld: 0.000e-6 1/Å^2\n        thickness: 10.000 Å\n        roughness: 3.300 Å\n'  # noqa: E501
        )

    def test_dict_round_trip(self):
        # When
        p = Sample()
        surfactant = SurfactantLayer()
        p.append(surfactant)
        multilayer = Multilayer()
        p.append(multilayer)
        repeating = RepeatingMultilayer()
        p.append(repeating)
        p_dict = p.as_dict()
        global_object.registry.clear()

        # Then
        q = Sample.from_dict(p_dict)

        # Expect
        assert sorted(p.as_dict()) == sorted(q.as_dict())
