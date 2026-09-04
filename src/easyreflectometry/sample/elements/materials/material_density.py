# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easyreflectometry.special.calculations import density_to_sld
from easyreflectometry.special.calculations import molecular_weight
from easyreflectometry.special.calculations import neutron_scattering_length
from easyreflectometry.utils import get_as_parameter

from .material import DEFAULTS as MATERIAL_DEFAULTS
from .material import Material

DEFAULTS = {
    'chemical_structure': 'Si',
    'density': {
        'description': 'The mass density of the material.',
        'url': 'https://en.wikipedia.org/wiki/Density',
        'value': 2.33,
        'unit': 'gram / centimeter ** 3',
        'min': 0,
        'max': np.inf,
        'fixed': True,
    },
    # A DescriptorNumber, not a Parameter: the molecular weight is a constant
    # of the chemical formula (recomputed whenever the formula changes) and
    # must never enter a fit — it is fully degenerate with density in the
    # derived SLD (only the ratio density/molecular_weight is observable).
    'molecular_weight': {
        'description': 'The molecular weight of a material.',
        'url': 'https://en.wikipedia.org/wiki/Molecular_mass',
        'value': 28.02,
        'unit': 'g / mole',
    },
}
DEFAULTS.update(MATERIAL_DEFAULTS)


class MaterialDensity(Material):
    """A material defined by chemical formula and mass density.

    The scattering length density is derived rather than set: from the
    formula, the coherent neutron scattering length ``b`` (real and
    imaginary parts, tabulated per isotope) and the molecular weight ``M``
    are computed, and ``sld``/``isld`` are wired as *dependent* parameters

        sld = N_A * density * b / M

    so ``density`` is the natural fit parameter and edits to the density or
    the formula propagate to the SLD automatically.

    The coupling can be switched off per material via :attr:`sld_coupled`:
    when ``False``, ``sld``/``isld`` are independent parameters that can be
    set and fitted directly, while ``density``, ``molecular_weight`` and the
    scattering lengths stop affecting anything until the coupling is
    restored. Restoring it (``sld_coupled = True``) recomputes the SLDs from
    the current formula and density, discarding manually set values. The
    state round-trips through ``as_dict``/``from_dict``, including the
    manual SLD values of a decoupled material; dictionaries from before
    this feature deserialize as coupled.

    Assigning :attr:`chemical_structure` updates the scattering lengths and
    the molecular weight together; an invalid formula raises ``ValueError``
    and leaves the material unchanged.

    :attr:`molecular_weight` is a read-only ``DescriptorNumber``, never a fit
    parameter: it is fully determined by the formula, and freeing it alongside
    density would make the fit degenerate (only ``density / molecular_weight``
    enters the derived SLD).
    """

    def __init__(
        self,
        chemical_structure: Union[str, None] = None,
        density: Union[Parameter, float, None] = None,
        name: str = 'EasyMaterialDensity',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        chemical_structure : Union[str, None], optional
            Chemical formula for the material. By default, None.
        density : Union[Parameter, float, None], optional
            Mass density for the material. By default, None.
        name : str, optional
            Identifier. By default, 'EasyMaterialDensity'.
        interface :
            Interface object. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if chemical_structure is None:
            chemical_structure = DEFAULTS['chemical_structure']

        density = get_as_parameter(
            name='density',
            value=density,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Density',
        )

        scattering_length = neutron_scattering_length(chemical_structure)

        mw = DescriptorNumber(
            name='molecular_weight',
            value=molecular_weight(chemical_structure),
            unit=DEFAULTS['molecular_weight']['unit'],
            description=DEFAULTS['molecular_weight']['description'],
            url=DEFAULTS['molecular_weight']['url'],
            unique_name=global_object.generate_unique_name(f'{unique_name}_Mw'),
        )
        scattering_length_real = get_as_parameter(
            name='scattering_length_real',
            value=scattering_length.real,
            default_dict=DEFAULTS['sld'],
            unique_name_prefix=f'{unique_name}_ScatteringLengthReal',
        )
        scattering_length_imag = get_as_parameter(
            name='scattering_length_imag',
            value=scattering_length.imag,
            default_dict=DEFAULTS['isld'],
            unique_name_prefix=f'{unique_name}_ScatteringLengthImag',
        )
        sld = get_as_parameter(
            name='sld',
            value=density_to_sld(scattering_length_real.value, mw.value, density.value),
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Sld',
        )
        isld = get_as_parameter(
            name='isld',
            value=density_to_sld(scattering_length_imag.value, mw.value, density.value),
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Isld',
        )

        dependency_expression = '1e-23*(0.602214076e6 * d * sl) / mw'
        dependency_map = {'d': density, 'sl': scattering_length_real, 'mw': mw}
        sld.make_dependent_on(dependency_expression=dependency_expression, dependency_map=dependency_map)

        dependency_map = {'d': density, 'sl': scattering_length_imag, 'mw': mw}
        isld.make_dependent_on(dependency_expression=dependency_expression, dependency_map=dependency_map)

        super().__init__(sld=sld, isld=isld, name=name, unique_name=unique_name, interface=None)

        self._scattering_length_real = scattering_length_real
        self._scattering_length_imag = scattering_length_imag
        self._molecular_weight = mw
        self._density = density
        self._chemical_structure = chemical_structure

        if interface is not None:
            self.interface = interface

    def _setup_sld_constraints(self) -> None:
        """Wire the derived `sld` / `isld` to depend on the current density and
        scattering-length Parameters.

        Idempotent — invoked once from `__init__` and again from `from_dict`
        after :class:`ModelBase` has swapped in the saved Parameter objects.
        """
        for derived in (self._sld, self._isld):
            if not derived.independent:
                derived.make_independent()

        dependency_expression = '1e-23*(0.602214076e6 * d * sl) / mw'
        self._sld.make_dependent_on(
            dependency_expression=dependency_expression,
            dependency_map={
                'd': self._density,
                'sl': self._scattering_length_real,
                'mw': self._molecular_weight,
            },
        )
        self._isld.make_dependent_on(
            dependency_expression=dependency_expression,
            dependency_map={
                'd': self._density,
                'sl': self._scattering_length_imag,
                'mw': self._molecular_weight,
            },
        )

    @property
    def sld_coupled(self) -> bool:
        """Whether ``sld``/``isld`` are derived from formula & density (True,
        the default) or independent, directly editable/fittable parameters
        (False). The dependency state itself is the source of truth."""
        return not self._sld.independent

    @sld_coupled.setter
    def sld_coupled(self, couple: bool) -> None:
        if couple == self.sld_coupled:
            return
        if couple:
            # Recomputes sld/isld from the current density/scattering
            # length/molecular weight — manually set values are discarded.
            self._setup_sld_constraints()
        else:
            # make_independent raises on an already-independent parameter,
            # so guard each individually. Values are kept.
            for parameter in (self._sld, self._isld):
                if not parameter.independent:
                    parameter.make_independent()

    def _convert_to_dict(self, d: dict, serializer, skip: Optional[list] = None, **kwargs) -> dict:
        """Serializer hook (see ``SerializerBase._convert_to_dict``).

        ``sld``/``isld`` are not constructor arguments, so the argspec-driven
        encoder never persists them; in the decoupled state their manually
        entered or fitted values would be lost on save/load without this.
        """
        d['sld_coupled'] = self.sld_coupled
        if not self.sld_coupled:
            d['sld'] = self._sld.value
            d['isld'] = self._isld.value
        return d

    @classmethod
    def from_dict(cls, obj_dict: dict) -> 'MaterialDensity':
        """Re-attach sld/isld dependencies after deserialization.

        :class:`ModelBase.from_dict` re-points `self._density` at the
        deserialized Parameter (because `density` is a constructor argument);
        the constraint graph built in `__init__` still references the
        temporary Parameter created from the float kwarg. Rebuild here so
        `q.density = X` propagates to the derived SLDs.

        The keys written by ``_convert_to_dict`` are not constructor
        arguments and must be removed before the parent's ``cls(**data)``
        call; they are then used to restore the coupling state. A dict
        without them (pre-feature project files) restores as coupled.
        """
        obj_dict = dict(obj_dict)
        sld_coupled = obj_dict.pop('sld_coupled', True)
        manual_sld = obj_dict.pop('sld', None)
        manual_isld = obj_dict.pop('isld', None)

        instance = super().from_dict(obj_dict)
        if sld_coupled:
            instance._setup_sld_constraints()
        else:
            # __init__ wired the dependencies; undo them and restore the
            # saved manual values.
            instance.sld_coupled = False
            if manual_sld is not None:
                instance._sld.value = manual_sld
            if manual_isld is not None:
                instance._isld.value = manual_isld
        return instance

    @property
    def chemical_structure(self) -> str:
        """Get the chemical structure string."""
        return self._chemical_structure

    @chemical_structure.setter
    def chemical_structure(self, structure_string: str) -> None:
        """Set the chemical structure string.

        Parameters
        ----------
        structure_string : str
            String that defines the chemical structure.
        """
        # Derive everything before mutating any state: an invalid formula
        # must leave the material fully unchanged. periodictable parses
        # garbage to an *empty* formula (b=0, mw=0) instead of raising, and
        # mw=0 would put a division by zero into the sld dependency.
        scattering_length = neutron_scattering_length(structure_string)
        # The molecular weight enters the sld dependency alongside the
        # scattering length; leaving it at the old formula's value would make
        # the derived sld a mix of two formulas.
        mw = molecular_weight(structure_string)
        if not mw:
            raise ValueError(f'Invalid chemical formula: {structure_string!r}')
        self._chemical_structure = structure_string
        self._scattering_length_real.value = scattering_length.real
        self._scattering_length_imag.value = scattering_length.imag
        self._molecular_weight.value = mw

    @property
    def density(self) -> Parameter:
        return self._density

    @density.setter
    def density(self, value: float) -> None:
        self._density.value = value

    @property
    def molecular_weight(self) -> DescriptorNumber:
        """The molecular weight of the formula. A read-only descriptor, not a
        fittable parameter: it is a constant of the chemical formula and is
        recomputed whenever :attr:`chemical_structure` is assigned."""
        return self._molecular_weight

    @property
    def scattering_length_real(self) -> Parameter:
        return self._scattering_length_real

    @property
    def scattering_length_imag(self) -> Parameter:
        return self._scattering_length_imag

    @property
    def _dict_repr(self) -> dict[str, str]:
        """Dictionary representation of the instance."""
        mat_dict = super()._dict_repr
        mat_dict['chemical_structure'] = self._chemical_structure
        mat_dict['density'] = f'{self.density.value:.2e} {self.density.unit}'
        return mat_dict
