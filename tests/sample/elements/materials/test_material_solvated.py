# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

import pytest
from easyscience import global_object

from easyreflectometry.sample.elements.materials.material import Material
from easyreflectometry.sample.elements.materials.material_solvated import MaterialSolvated


class TestMaterialSolvated:
    @pytest.fixture
    def material_solvated(self, monkeypatch) -> MaterialSolvated:
        self.material = Material(sld=1.0, isld=0, name='material')
        self.solvent = Material(sld=2.0, isld=0, name='solvent')
        # self.mock_solvent_fraction = MagicMock(spec=Parameter)
        # self.mock_solvent_fraction.value = 0.1
        self.mock_interface = MagicMock()
        self.mock_Parameter = MagicMock()
        # monkeypatch.setattr(easyreflectometry.sample.elements.materials.material_mixture, 'Parameter', self.mock_Parameter)
        return MaterialSolvated(
            material=self.material,
            solvent=self.solvent,
            solvent_fraction=0.1,
            name='name',
            interface=self.mock_interface,
        )

    def test_init(self, material_solvated: MaterialSolvated) -> None:
        # When Then Expect
        assert material_solvated.material_a == self.material
        assert material_solvated.material_b == self.solvent
        assert material_solvated.fraction.value == 0.1
        assert material_solvated.name == 'name'
        assert material_solvated.interface == self.mock_interface
        self.mock_interface.generate_bindings.call_count == 2

    def test_material(self, material_solvated: MaterialSolvated) -> None:
        # When Then Expect
        assert material_solvated.material == self.material

    def test_set_material(self, material_solvated: MaterialSolvated) -> None:
        # When
        new_material = Material(sld=1.0, isld=0, name='new_material')

        # Then
        material_solvated.material = new_material

        # Expect
        assert material_solvated.material == new_material
        assert material_solvated.name == 'new_material in solvent'

    def test_solvent(self, material_solvated: MaterialSolvated) -> None:
        # When Then Expect
        assert material_solvated.solvent == self.solvent

    def test_set_solvent(self, material_solvated: MaterialSolvated) -> None:
        # When
        new_solvent = Material(sld=2.0, isld=0, name='new_solvent')

        # Then
        material_solvated.solvent = new_solvent

        # Expect
        assert material_solvated.solvent == new_solvent
        assert material_solvated.name == 'material in new_solvent'

    def test_solvent_fraction(self, material_solvated: MaterialSolvated) -> None:
        # When Then Expect
        assert material_solvated.solvent_fraction.value == 0.1

    def test_set_solvent_fraction(self, material_solvated: MaterialSolvated) -> None:
        # When Then
        material_solvated.solvent_fraction = 1.0

        # Expect
        assert material_solvated.solvent_fraction.value == 1.0

    def test_set_solvent_fraction_exception(self, material_solvated: MaterialSolvated) -> None:
        # When Then Expect
        with pytest.raises(ValueError):
            material_solvated.solvent_fraction = 'not float'

        with pytest.raises(ValueError):
            material_solvated.solvent_fraction = 1.1

        with pytest.raises(ValueError):
            material_solvated.solvent_fraction = -0.1

        # When Then Expect (no exception)
        material_solvated.solvent_fraction = 0.0

        # When Then Expect (no exception)
        material_solvated.solvent_fraction = 1.0

    def test_dict_repr(self) -> None:
        # When Then
        p = MaterialSolvated()

        # Expect
        assert p._dict_repr == {
            'D2O in H2O': {
                'solvent_fraction': '0.200 dimensionless',
                'sld': '4.976e-6 1/Å^2',
                'isld': '0.000e-6 1/Å^2',
                'material': {'D2O': {'sld': '6.360e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
                'solvent': {'H2O': {'sld': '-0.561e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
            }
        }

    def test_dict_round_trip(self):
        p = MaterialSolvated()
        p_dict = p.as_dict()
        global_object.registry.clear()

        q = MaterialSolvated.from_dict(p_dict)

        assert sorted(p.as_dict()) == sorted(q.as_dict())

    def test_update_name(self, material_solvated: MaterialSolvated) -> None:
        # When
        mock_material_a = MagicMock()
        mock_material_a.name = 'name_a'
        material_solvated._material_a = mock_material_a
        mock_material_b = MagicMock()
        mock_material_b.name = 'name_b'
        material_solvated._material_b = mock_material_b

        # Then
        material_solvated._update_name()

        # Expect
        assert material_solvated.name == 'name_a in name_b'

    def test_solvent_fraction_metadata_survives_round_trip(self) -> None:
        """Regression: ``solvent_fraction`` is a constructor argument, but its
        backing storage is ``_fraction`` (inherited from MaterialMixture).
        ``ModelBase.from_dict`` would write the saved Parameter to
        ``_solvent_fraction`` (an orphan), silently resetting the active
        parameter to constructor defaults. We re-route to ``_fraction`` in
        ``MaterialSolvated.from_dict``.
        """
        material = Material(sld=6.36, isld=0, name='D2O')
        solvent = Material(sld=-0.561, isld=0, name='H2O')
        p = MaterialSolvated(material=material, solvent=solvent, solvent_fraction=0.3)
        # Tweak fit metadata that the default would not have.
        p.solvent_fraction.fixed = False
        p.solvent_fraction.min = 0.12

        p_dict = p.as_dict()
        global_object.registry.clear()

        q = MaterialSolvated.from_dict(p_dict)

        assert q.solvent_fraction.value == 0.3
        assert q.solvent_fraction.fixed is False
        assert q.solvent_fraction.min == 0.12
        # No orphan field.
        assert '_solvent_fraction' not in q.__dict__
