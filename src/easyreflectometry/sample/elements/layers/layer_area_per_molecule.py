# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.special.calculations import neutron_scattering_length
from easyreflectometry.utils import get_as_parameter

from ..materials.material import Material
from ..materials.material_solvated import DEFAULTS as MATERIAL_SOLVATED_DEFAULTS
from ..materials.material_solvated import MaterialSolvated
from .layer import DEFAULTS as LAYER_DEFAULTS
from .layer import Layer

DEFAULTS = {
    'molecular_formula': 'C10H18NO8P',
    'area_per_molecule': {
        'description': 'Surface coverage',
        'value': 48.2,
        'unit': 'angstrom^2',
        'min': 0,
        'max': np.inf,
        'fixed': True,
    },
    'sl': {
        'description': 'The real scattering length for a molecule formula in angstrom.',
        'url': 'https://www.ncnr.nist.gov/resources/activation/',
        'value': 4.186,
        'unit': 'angstrom',
        'min': -np.inf,
        'max': np.inf,
        'fixed': True,
    },
    'isl': {
        'description': 'The real scattering length for a molecule formula in angstrom.',
        'url': 'https://www.ncnr.nist.gov/resources/activation/',
        'value': 0.0,
        'unit': 'angstrom',
        'min': -np.inf,
        'max': np.inf,
        'fixed': True,
    },
}
DEFAULTS.update(MATERIAL_SOLVATED_DEFAULTS)
DEFAULTS.update(LAYER_DEFAULTS)


class LayerAreaPerMolecule(Layer):
    """The `LayerAreaPerMolecule` class allows a layer to be defined in terms of some
    molecular formula an area per molecule, and a solvent.
    """

    def __init__(
        self,
        molecular_formula: Union[str, None] = None,
        thickness: Union[Parameter, float, None] = None,
        solvent: Union[Material, None] = None,
        solvent_fraction: Union[Parameter, float, None] = None,
        area_per_molecule: Union[Parameter, float, None] = None,
        roughness: Union[Parameter, float, None] = None,
        name: str = 'EasyLayerAreaPerMolecule',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        molecular_formula : Union[str, None], optional
            Formula for the molecule in the layer. By default, None.
        thickness : Union[Parameter, float, None], optional
            Layer thickness in Angstrom. By default, None.
        solvent : Union[Material, None], optional
            Solvent containing the molecule. By default, None.
        solvent_fraction : Union[Parameter, float, None], optional
            Fraction of solvent in layer. Fx solvation or surface coverage. By default, None.
        area_per_molecule : Union[Parameter, float, None], optional
            Area per molecule in the layer. By default, None.
        roughness : Union[Parameter, float, None], optional
            Upper roughness on the layer in Angstrom. By default, None.
        name : str, optional
            Name of the layer. By default, 'EasyLayerAreaPerMolecule'.
        interface :
            Interface object. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if solvent is None:
            solvent = Material(
                sld=6.36,
                isld=0,
                name='D2O',
                unique_name=unique_name + '_MaterialSolvent',
                interface=interface,
            )

        if molecular_formula is None:
            molecular_formula = DEFAULTS['molecular_formula']
        molecule_material = Material(
            sld=0.0,
            isld=0.0,
            name=molecular_formula,
            unique_name=unique_name + '_MaterialMolecule',
            interface=interface,
        )

        thickness = get_as_parameter(
            name='thickness',
            value=thickness,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Thickness',
        )
        area_per_molecule_param = get_as_parameter(
            name='area_per_molecule',
            value=area_per_molecule,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_AreaPerMolecule',
        )
        scattering_length_real = get_as_parameter(
            name='scattering_length_real',
            value=0.0,
            default_dict=DEFAULTS['sl'],
            unique_name_prefix=f'{unique_name}_Sl',
        )
        scattering_length_imag = get_as_parameter(
            name='scattering_length_imag',
            value=0.0,
            default_dict=DEFAULTS['isl'],
            unique_name_prefix=f'{unique_name}_Isl',
        )

        # Constrain molecule.sld / .isld to scattering_length / (thickness * area_per_molecule).
        # `_setup_sld_constraints` rebuilds the same expression after from_dict, so keep the
        # variable names (`a`, `b`, `p`) consistent with that path.
        dependency_expression = 'a / (b*p) * 1e6'
        molecule_material.sld.make_dependent_on(
            dependency_expression=dependency_expression,
            dependency_map={'a': scattering_length_real, 'b': thickness, 'p': area_per_molecule_param},
        )
        molecule_material.isld.make_dependent_on(
            dependency_expression=dependency_expression,
            dependency_map={'a': scattering_length_imag, 'b': thickness, 'p': area_per_molecule_param},
        )

        solvated_molecule_material = MaterialSolvated(
            material=molecule_material,
            solvent=solvent,
            solvent_fraction=solvent_fraction,
            unique_name=unique_name + '_MaterialSolvated',
            interface=interface,
        )
        super().__init__(
            material=solvated_molecule_material,
            thickness=thickness,
            roughness=roughness,
            name=name,
            unique_name=unique_name,
            interface=None,
        )
        self._area_per_molecule = area_per_molecule_param
        self._scattering_length_real = scattering_length_real
        self._scattering_length_imag = scattering_length_imag

        scattering_length = neutron_scattering_length(molecular_formula)
        self._scattering_length_real.value = scattering_length.real
        self._scattering_length_imag.value = scattering_length.imag
        self._molecular_formula = molecular_formula

        if interface is not None:
            self.interface = interface

    # ----- constraint plumbing -----

    def _setup_sld_constraints(self) -> None:
        """Wire the inner molecule material's ``sld`` / ``isld`` to depend on
        the current scattering-length, thickness, and area-per-molecule
        parameters.

        Idempotent — called once from ``__init__`` and again from
        ``from_dict`` after the saved Parameter objects replace the
        constructor-time temporaries.
        """
        molecule_material = self.material.material
        for derived in (molecule_material.sld, molecule_material.isld):
            if not derived.independent:
                derived.make_independent()

        dependency_expression = 'a / (b*p) * 1e6'
        molecule_material.sld.make_dependent_on(
            dependency_expression=dependency_expression,
            dependency_map={
                'a': self._scattering_length_real,
                'b': self._thickness,
                'p': self._area_per_molecule,
            },
        )
        molecule_material.isld.make_dependent_on(
            dependency_expression=dependency_expression,
            dependency_map={
                'a': self._scattering_length_imag,
                'b': self._thickness,
                'p': self._area_per_molecule,
            },
        )

    # ----- deserialization -----

    @classmethod
    def from_dict(cls, obj_dict: dict) -> 'LayerAreaPerMolecule':
        """Re-route the saved ``solvent_fraction`` Parameter and rebuild the
        molecule-SLD constraint chain after :class:`ModelBase.from_dict`
        swaps in the persisted Parameter objects.

        `ModelBase.from_dict` writes the deserialized ``solvent_fraction``
        Parameter to ``self._solvent_fraction`` (orphan — the live property
        delegates to ``self.material.solvent_fraction``, which is
        ``self.material._fraction``). It also reassigns ``self._thickness``
        and ``self._area_per_molecule``, but the constraint graph built in
        ``__init__`` still references the temporary Parameters created from
        the float kwargs. We fix both here.
        """
        instance = super().from_dict(obj_dict)

        saved_solvent_fraction = instance.__dict__.pop('_solvent_fraction', None)
        if saved_solvent_fraction is not None:
            mixture = instance.material
            old = mixture._fraction
            mixture._fraction = saved_solvent_fraction
            try:
                instance._global_object.registry.discard(old.unique_name)
            except (AttributeError, KeyError):
                pass
            mixture._materials_constraints()

        instance._setup_sld_constraints()
        return instance

    @property
    def area_per_molecule_parameter(self) -> Parameter:
        """Get the parameter for area per molecule."""
        return self._area_per_molecule

    @property
    def area_per_molecule(self) -> Parameter:
        """The Parameter that controls area per molecule."""
        return self._area_per_molecule

    @area_per_molecule.setter
    def area_per_molecule(self, value: float) -> None:
        if value < 0:
            raise ValueError('area_per_molecule must be greater than 0.0.')
        self._area_per_molecule.value = value

    @property
    def molecule(self) -> Material:
        """Get the molecule material."""
        return self.material.material

    @property
    def solvent(self) -> Material:
        """Get the solvent material."""
        return self.material.solvent

    @solvent.setter
    def solvent(self, new_solvent: Material) -> None:
        self.material.solvent = new_solvent

    @property
    def solvent_fraction_parameter(self) -> Parameter:
        """Get parameter for the fraction of the layer occupied by the solvent."""
        return self.material.solvent_fraction_parameter

    @property
    def solvent_fraction(self) -> Parameter:
        """The Parameter for the fraction of the layer occupied by the solvent."""
        return self.material.solvent_fraction

    @solvent_fraction.setter
    def solvent_fraction(self, value: float) -> None:
        self.material.solvent_fraction = value

    @property
    def molecular_formula(self) -> str:
        """Get the formula of molecule the layer."""
        return self._molecular_formula

    @molecular_formula.setter
    def molecular_formula(self, formula_string: str) -> None:
        self._molecular_formula = formula_string
        scattering_length = neutron_scattering_length(formula_string)
        self._scattering_length_real.value = scattering_length.real
        self._scattering_length_imag.value = scattering_length.imag

        self.molecule.name = formula_string
        self.material._update_name()

    @property
    def _dict_repr(self) -> dict[str, str]:
        """Dictionary representation of the `area_per_molecule` object."""
        dict_repr = super()._dict_repr
        dict_repr['molecular_formula'] = self._molecular_formula
        dict_repr['area_per_molecule'] = f'{self._area_per_molecule.value:.2f} {self._area_per_molecule.unit}'
        return dict_repr
