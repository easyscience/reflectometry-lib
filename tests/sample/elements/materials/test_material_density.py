# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import numpy as np
from easyscience import global_object
from numpy.testing import assert_almost_equal

from easyreflectometry.sample.elements.materials.material_density import MaterialDensity


class TestMaterialDensity(unittest.TestCase):
    def test_default(self):
        p = MaterialDensity()
        assert p.name == 'EasyMaterialDensity'
        assert p.interface is None
        assert p.density.display_name == 'density'
        assert str(p.density.unit) == 'kg/L'
        assert p.density.value == 2.33
        assert p.density.min == 0
        assert p.density.max == np.inf
        assert p.density.fixed is True

    def test_default_constraint(self):
        p = MaterialDensity()
        assert p.density.value == 2.33
        assert_almost_equal(p.sld.value, 2.0737423003838087)
        p.density.value = 2
        assert_almost_equal(p.sld.value, 1.7800363093423253)

    def test_from_pars(self):
        p = MaterialDensity('Co', 8.9, 'Cobalt')
        assert p.density.value == 8.9
        assert_almost_equal(p.sld.value, 2.264541463379026)
        assert p.chemical_structure == 'Co'

    def test_chemical_structure_change(self):
        p = MaterialDensity('Co', 8.9, 'Cobalt')
        assert p.density.value == 8.9
        assert_almost_equal(p.sld.value, 2.264541463379026)
        assert_almost_equal(p.isld.value, 0.0)
        assert p.chemical_structure == 'Co'
        p.chemical_structure = 'B'
        assert p.density.value == 8.9
        # The setter updates the molecular weight along with the scattering
        # lengths; the derived sld reflects boron's mw, not cobalt's.
        assert_almost_equal(p.molecular_weight.value, 10.81)
        assert_almost_equal(p.sld.value, 26.277925961998147)
        assert_almost_equal(p.isld.value, -1.0412008400037)
        assert p.chemical_structure == 'B'

    def test_dict_repr(self):
        p = MaterialDensity()
        print(p._dict_repr)
        assert p._dict_repr == {
            'EasyMaterialDensity': {'sld': '2.074e-6 kmol/m^5', 'isld': '0.000e-6 kmol/m^5'},
            'chemical_structure': 'Si',
            'density': '2.33e+00 kg/L',
        }

    def test_dict_round_trip(self):
        p = MaterialDensity()
        p_dict = p.as_dict()
        global_object.map._clear()

        q = MaterialDensity.from_dict(p_dict)

        assert sorted(p.as_dict()) == sorted(q.as_dict())

    def test_chemical_structure_invalid_formula_leaves_material_unchanged(self):
        p = MaterialDensity('Co', 8.9, 'Cobalt')
        mw = p.molecular_weight.value
        sld = p.sld.value
        with self.assertRaises(ValueError):
            p.chemical_structure = '###'
        assert p.chemical_structure == 'Co'
        assert_almost_equal(p.molecular_weight.value, mw)
        assert_almost_equal(p.sld.value, sld)

    def test_sld_coupled_default_true(self):
        p = MaterialDensity()
        assert p.sld_coupled is True
        assert p.sld.independent is False
        assert p.isld.independent is False

    def test_decouple_keeps_values_and_detaches_density(self):
        p = MaterialDensity(chemical_structure='Si', density=2.33)
        coupled_sld = p.sld.value
        p.sld_coupled = False
        assert p.sld_coupled is False
        assert p.sld.independent is True
        assert p.isld.independent is True
        assert_almost_equal(p.sld.value, coupled_sld)
        # Density edits no longer propagate; sld is directly settable.
        p.density.value = 9.99
        assert_almost_equal(p.sld.value, coupled_sld)
        p.sld.value = 5.5
        assert p.sld.value == 5.5

    def test_recouple_recomputes_and_discards_manual_sld(self):
        p = MaterialDensity(chemical_structure='Si', density=2.33)
        coupled_sld = p.sld.value
        p.sld_coupled = False
        p.sld.value = 5.5
        p.sld_coupled = True
        assert p.sld_coupled is True
        assert_almost_equal(p.sld.value, coupled_sld)
        # And propagation is restored.
        p.density.value = 4.66
        assert_almost_equal(p.sld.value, 2 * coupled_sld)

    def test_sld_coupled_setter_is_idempotent(self):
        p = MaterialDensity()
        p.sld_coupled = True  # no-op, must not raise or rewire
        p.sld_coupled = False
        p.sld_coupled = False  # no-op on the decoupled side too
        assert p.sld_coupled is False

    def test_decoupled_sld_can_be_freed_for_fitting(self):
        p = MaterialDensity()
        p.sld_coupled = False
        p.sld.fixed = False
        free = p.get_fit_parameters()
        assert any(parameter is p.sld for parameter in free)

    def test_dict_round_trip_decoupled_preserves_manual_sld(self):
        p = MaterialDensity(chemical_structure='Si', density=2.33)
        p.sld_coupled = False
        p.sld.value = 7.25
        p.isld.value = -0.5
        p_dict = p.as_dict()
        assert p_dict['sld_coupled'] is False
        assert p_dict['sld'] == 7.25
        assert p_dict['isld'] == -0.5
        global_object.map._clear()

        q = MaterialDensity.from_dict(p_dict)
        assert q.sld_coupled is False
        assert_almost_equal(q.sld.value, 7.25)
        assert_almost_equal(q.isld.value, -0.5)
        # Still decoupled: density edits must not clobber the restored values.
        q.density.value = 1.0
        assert_almost_equal(q.sld.value, 7.25)

    def test_dict_round_trip_coupled_carries_flag_but_no_sld(self):
        p = MaterialDensity()
        p_dict = p.as_dict()
        assert p_dict['sld_coupled'] is True
        assert 'sld' not in p_dict
        assert 'isld' not in p_dict

    def test_from_dict_without_flag_restores_coupled(self):
        """Project files predating the feature restore with current behavior."""
        p = MaterialDensity(chemical_structure='Si', density=2.33)
        p_dict = {k: v for k, v in p.as_dict().items() if k != 'sld_coupled'}
        global_object.map._clear()

        q = MaterialDensity.from_dict(p_dict)
        assert q.sld_coupled is True
        q.density.value = 4.66
        assert_almost_equal(q.sld.value, 2 * p.sld.value)

    def test_density_mutation_propagates_after_round_trip(self):
        """Regression: after ``from_dict`` reattaches the saved ``_density``
        Parameter, mutating it must propagate to ``sld`` / ``isld`` (which
        are constrained off it). The ``__init__``-time constraint references
        the temporary constructor Parameter; ``from_dict`` rebuilds the
        graph so subsequent mutations propagate correctly.
        """
        p = MaterialDensity(chemical_structure='Si', density=2.33)
        original_sld = p.sld.value
        p_dict = p.as_dict()
        global_object.map._clear()

        q = MaterialDensity.from_dict(p_dict)
        assert_almost_equal(q.sld.value, original_sld)

        q.density = 4.66
        # SLD scales linearly with density (constraint: d * sl / mw, etc.)
        assert_almost_equal(q.sld.value, 2 * original_sld)
