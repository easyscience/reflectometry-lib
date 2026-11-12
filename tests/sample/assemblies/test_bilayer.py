# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for Bilayer class module
"""

__author__ = 'github.com/easyscience'


from easyscience import global_object

from easyreflectometry.constraints import USER_CONSTRAINT_FLAG
from easyreflectometry.sample.assemblies.bilayer import Bilayer
from easyreflectometry.sample.elements.layers.layer import Layer
from easyreflectometry.sample.elements.layers.layer_area_per_molecule import LayerAreaPerMolecule
from easyreflectometry.sample.elements.materials.material import Material


class TestBilayer:
    def setup_method(self):
        from easyscience import global_object

        # Clear the global object registry to prevent name collisions
        global_object.registry.clear()

    def test_default(self):
        """Test default bilayer creation with expected structure."""
        p = Bilayer()
        assert p.name == 'EasyBilayer'
        assert p._type == 'Bilayer'

        # Check layer count
        assert len(p.layers) == 4

        # Check layer order: front_head, front_tail, back_tail, back_head
        assert p.layers[0].name == 'DPPC Head Front'
        assert p.front_head_layer.name == 'DPPC Head Front'

        assert p.layers[1].name == 'DPPC Tail'
        assert p.front_tail_layer.name == 'DPPC Tail'

        assert p.layers[2].name == 'DPPC Tail Back'
        assert p.back_tail_layer.name == 'DPPC Tail Back'

        assert p.layers[3].name == 'DPPC Head Back'
        assert p.back_head_layer.name == 'DPPC Head Back'

    def test_default_constraints_enabled(self):
        """Test that default bilayer has constraints enabled."""
        p = Bilayer()

        # Default should have conformal roughness and head constraints
        assert p.conformal_roughness is True
        assert p.constrain_heads is True

    def test_layer_structure(self):
        """Verify 4 layers in correct order."""
        p = Bilayer()

        assert p.front_head_layer is p.layers[0]
        assert p.front_tail_layer is p.layers[1]
        assert p.back_tail_layer is p.layers[2]
        assert p.back_head_layer is p.layers[3]

    def test_custom_layers(self):
        """Test creation with custom head/tail layers."""
        d2o = Material(sld=6.36, isld=0, name='D2O')
        air_matched_water = Material(sld=0, isld=0, name='Air Matched Water')

        front_head = LayerAreaPerMolecule(
            molecular_formula='C10H18NO8P',
            thickness=12.0,
            solvent=d2o,
            solvent_fraction=0.3,
            area_per_molecule=50.0,
            roughness=2.0,
            name='Custom Front Head',
        )
        tail = LayerAreaPerMolecule(
            molecular_formula='C32D64',
            thickness=18.0,
            solvent=air_matched_water,
            solvent_fraction=0.0,
            area_per_molecule=50.0,
            roughness=2.0,
            name='Custom Tail',
        )
        back_head = LayerAreaPerMolecule(
            molecular_formula='C10H18NO8P',
            thickness=12.0,
            solvent=d2o,
            solvent_fraction=0.4,  # Different hydration
            area_per_molecule=50.0,
            roughness=2.0,
            name='Custom Back Head',
        )

        p = Bilayer(
            front_head_layer=front_head,
            front_tail_layer=tail,
            back_head_layer=back_head,
            name='Custom Bilayer',
        )

        assert p.name == 'Custom Bilayer'
        assert p.front_head_layer.name == 'Custom Front Head'
        assert p.front_tail_layer.name == 'Custom Tail'
        assert p.back_head_layer.name == 'Custom Back Head'
        assert p.front_head_layer.thickness.value == 12.0
        assert p.front_tail_layer.thickness.value == 18.0

    def test_tail_layers_linked(self):
        """Test that both tail layers share parameters."""
        p = Bilayer()

        # Initial values should match
        assert p.front_tail_layer.thickness.value == p.back_tail_layer.thickness.value
        assert p.front_tail_layer.area_per_molecule.value == p.back_tail_layer.area_per_molecule.value

        # Change front tail thickness - back tail should follow
        p.front_tail_layer.thickness.value = 20.0
        assert p.front_tail_layer.thickness.value == 20.0
        assert p.back_tail_layer.thickness.value == 20.0

        # Change front tail area per molecule - back tail should follow
        p.front_tail_layer.area_per_molecule = 55.0
        assert p.front_tail_layer.area_per_molecule.value == 55.0
        assert p.back_tail_layer.area_per_molecule.value == 55.0

    def test_constrain_heads_enabled(self):
        """Test head thickness/area constraint when enabled."""
        p = Bilayer(constrain_heads=True)

        # Change front head thickness - back head should follow
        p.front_head_layer.thickness.value = 15.0
        assert p.front_head_layer.thickness.value == 15.0
        assert p.back_head_layer.thickness.value == 15.0

        # Change front head area per molecule - back head should follow
        p.front_head_layer.area_per_molecule = 60.0
        assert p.front_head_layer.area_per_molecule.value == 60.0
        assert p.back_head_layer.area_per_molecule.value == 60.0

    def test_constrain_heads_disabled(self):
        """Test heads are independent when constraint disabled."""
        p = Bilayer(constrain_heads=False)

        # Set different values for front and back heads
        p.front_head_layer.thickness.value = 15.0
        p.back_head_layer.thickness.value = 12.0

        assert p.front_head_layer.thickness.value == 15.0
        assert p.back_head_layer.thickness.value == 12.0

    def test_constrain_heads_toggle(self):
        """Test enabling/disabling head constraints at runtime."""
        p = Bilayer(constrain_heads=False)

        # Set different values
        p.front_head_layer.thickness.value = 15.0
        p.back_head_layer.thickness.value = 12.0

        # Enable constraint - back head should match front head
        p.constrain_heads = True
        assert p.constrain_heads is True

        # Change front head - back should follow
        p.front_head_layer.thickness.value = 20.0
        assert p.back_head_layer.thickness.value == 20.0

        # Disable constraint
        p.constrain_heads = False
        assert p.constrain_heads is False

        # Now they can be independent
        p.back_head_layer.thickness.value = 18.0
        assert p.front_head_layer.thickness.value == 20.0
        assert p.back_head_layer.thickness.value == 18.0

    def test_head_hydration_independent(self):
        """Test that head hydrations remain independent even with constraints."""
        p = Bilayer(constrain_heads=True)

        # Set different solvent fractions
        p.front_head_layer.solvent_fraction = 0.3
        p.back_head_layer.solvent_fraction = 0.5

        # They should remain independent
        assert p.front_head_layer.solvent_fraction.value == 0.3
        assert p.back_head_layer.solvent_fraction.value == 0.5

    def test_conformal_roughness_enabled(self):
        """Test all roughnesses are linked when conformal roughness enabled."""
        p = Bilayer(conformal_roughness=True)

        # Change front head roughness - all should follow
        p.front_head_layer.roughness.value = 5.0
        assert p.front_head_layer.roughness.value == 5.0
        assert p.front_tail_layer.roughness.value == 5.0
        assert p.back_tail_layer.roughness.value == 5.0
        assert p.back_head_layer.roughness.value == 5.0

    def test_conformal_roughness_disabled(self):
        """Test roughnesses are independent when conformal roughness disabled."""
        p = Bilayer(conformal_roughness=False)

        # Set different roughnesses
        p.front_head_layer.roughness.value = 2.0
        p.front_tail_layer.roughness.value = 3.0
        p.back_tail_layer.roughness.value = 4.0
        p.back_head_layer.roughness.value = 5.0

        assert p.front_head_layer.roughness.value == 2.0
        assert p.front_tail_layer.roughness.value == 3.0
        assert p.back_tail_layer.roughness.value == 4.0
        assert p.back_head_layer.roughness.value == 5.0

    def test_conformal_roughness_toggle(self):
        """Test enabling/disabling conformal roughness at runtime."""
        p = Bilayer(conformal_roughness=False)

        # Set different values
        p.front_head_layer.roughness.value = 2.0
        p.back_head_layer.roughness.value = 5.0

        # Enable conformal roughness
        p.conformal_roughness = True
        assert p.conformal_roughness is True

        # Change front head - all should follow
        p.front_head_layer.roughness.value = 4.0
        assert p.front_tail_layer.roughness.value == 4.0
        assert p.back_tail_layer.roughness.value == 4.0
        assert p.back_head_layer.roughness.value == 4.0

        # Disable conformal roughness
        p.conformal_roughness = False
        assert p.conformal_roughness is False

    def test_get_set_front_head_layer(self):
        """Test getting and setting front head layer."""
        p = Bilayer()
        new_layer = LayerAreaPerMolecule(
            molecular_formula='C8H16NO6P',
            thickness=8.0,
            name='New Front Head',
        )

        p.front_head_layer = new_layer

        assert p.front_head_layer == new_layer
        assert p.layers[0] == new_layer

    def test_get_set_back_head_layer(self):
        """Test getting and setting back head layer."""
        p = Bilayer()
        new_layer = LayerAreaPerMolecule(
            molecular_formula='C8H16NO6P',
            thickness=8.0,
            name='New Back Head',
        )

        p.back_head_layer = new_layer

        assert p.back_head_layer == new_layer
        assert p.layers[3] == new_layer

    def test_dict_repr(self):
        """Test dictionary representation."""
        p = Bilayer()

        dict_repr = p._dict_repr
        assert 'EasyBilayer' in dict_repr
        assert 'front_head_layer' in dict_repr['EasyBilayer']
        assert 'front_tail_layer' in dict_repr['EasyBilayer']
        assert 'back_tail_layer' in dict_repr['EasyBilayer']
        assert 'back_head_layer' in dict_repr['EasyBilayer']
        assert 'constrain_heads' in dict_repr['EasyBilayer']
        assert 'conformal_roughness' in dict_repr['EasyBilayer']


def test_dict_round_trip():
    """Test serialization/deserialization round trip."""
    # When
    d2o = Material(sld=6.36, isld=0, name='D2O')
    air_matched_water = Material(sld=0, isld=0, name='Air Matched Water')

    front_head = LayerAreaPerMolecule(
        molecular_formula='C10H18NO8P',
        thickness=12.0,
        solvent=d2o,
        solvent_fraction=0.3,
        area_per_molecule=50.0,
        roughness=2.0,
        name='Custom Front Head',
    )
    tail = LayerAreaPerMolecule(
        molecular_formula='C32D64',
        thickness=18.0,
        solvent=air_matched_water,
        solvent_fraction=0.0,
        area_per_molecule=50.0,
        roughness=2.0,
        name='Custom Tail',
    )
    back_head = LayerAreaPerMolecule(
        molecular_formula='C10H18NO8P',
        thickness=12.0,
        solvent=d2o,
        solvent_fraction=0.4,
        area_per_molecule=50.0,
        roughness=2.0,
        name='Custom Back Head',
    )

    p = Bilayer(
        front_head_layer=front_head,
        front_tail_layer=tail,
        back_head_layer=back_head,
        constrain_heads=False,
        conformal_roughness=False,
    )
    p_dict = p.as_dict()
    global_object.registry.clear()

    # Then
    q = Bilayer.from_dict(p_dict)

    # Expect
    assert sorted(p.as_dict()) == sorted(q.as_dict())


def test_dict_round_trip_constraints_enabled():
    """Test round trip with constraints enabled."""
    # When
    p = Bilayer(constrain_heads=True, conformal_roughness=True)
    p_dict = p.as_dict()
    global_object.registry.clear()

    # Then
    q = Bilayer.from_dict(p_dict)

    # Expect
    assert q.constrain_heads is True
    assert q.conformal_roughness is True
    assert sorted(p.as_dict()) == sorted(q.as_dict())


def test_dict_round_trip_constraints_disabled():
    """Test round trip with constraints disabled."""
    # When
    p = Bilayer(constrain_heads=False, conformal_roughness=False)
    p_dict = p.as_dict()
    global_object.registry.clear()

    # Then
    q = Bilayer.from_dict(p_dict)

    # Expect
    assert q.constrain_heads is False
    assert q.conformal_roughness is False
    assert sorted(p.as_dict()) == sorted(q.as_dict())


def test_constrain_multiple_contrast():
    """Test multi-contrast constraints between bilayers."""
    # When
    p1 = Bilayer(name='Bilayer 1', constrain_heads=False)
    p2 = Bilayer(name='Bilayer 2', constrain_heads=False)

    # Set initial values
    p1.front_head_layer.thickness.value = 10.0
    p1.front_tail_layer.thickness.value = 16.0

    # Constrain p2 to p1
    p2.constrain_multiple_contrast(
        p1,
        front_head_thickness=True,
        tail_thickness=True,
    )

    # Then - p2 values should match p1
    assert p2.front_head_layer.thickness.value == 10.0
    assert p2.front_tail_layer.thickness.value == 16.0

    # Change p1 - p2 should follow
    p1.front_head_layer.thickness.value = 12.0
    assert p2.front_head_layer.thickness.value == 12.0


def test_constrain_solvent_roughness():
    """Test constraining solvent roughness to bilayer roughness."""
    # When
    p = Bilayer(conformal_roughness=True)
    layer = Layer()

    p.front_head_layer.roughness.value = 4.0

    # Then
    p.constrain_solvent_roughness(layer.roughness)

    # Expect
    assert layer.roughness.value == 4.0

    # Change bilayer roughness - solvent should follow
    p.front_head_layer.roughness.value = 5.0
    assert layer.roughness.value == 5.0


def test_a_constrained_solvent_roughness_can_follow_another_bilayer():
    # One solvent layer shared by two contrasts: the second call rewires it.
    first, second = Bilayer(conformal_roughness=True), Bilayer(conformal_roughness=True)
    first.front_head_layer.roughness.value = 2.0
    second.front_head_layer.roughness.value = 5.0
    solvent = Layer()
    first.constrain_solvent_roughness(solvent.roughness)

    second.constrain_solvent_roughness(solvent.roughness)

    assert solvent.roughness.value == 5.0
    second.front_head_layer.roughness.value = 6.0
    assert solvent.roughness.value == 6.0


def test_constrain_solvent_roughness_is_a_user_constraint():
    # The flag is what Project.save records, so the tie survives a reload.
    bilayer = Bilayer(conformal_roughness=True)
    solvent = Layer()

    bilayer.constrain_solvent_roughness(solvent.roughness)

    assert getattr(solvent.roughness, USER_CONSTRAINT_FLAG, False)
    assert not solvent.roughness.independent


def test_constrain_solvent_roughness_error():
    """Test error when constraining solvent roughness without conformal roughness."""
    import pytest

    p = Bilayer(conformal_roughness=False)
    layer = Layer()

    with pytest.raises(ValueError, match='Roughness must be conformal'):
        p.constrain_solvent_roughness(layer.roughness)
