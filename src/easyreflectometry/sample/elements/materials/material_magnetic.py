# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from numbers import Number
from typing import Optional
from typing import Union

from easyscience import global_object
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easyreflectometry.constraints import unconstrain
from easyreflectometry.special.calculations import MAGNETIC_SCATTERING_LENGTH
from easyreflectometry.utils import get_as_parameter

from .material import DEFAULTS as MATERIAL_DEFAULTS
from .material_density import MaterialDensity

DEFAULTS = {
    'magnetic_moment': {
        'description': 'Effective in-plane net magnetic moment in Bohr magnetons per formula unit.',
        'url': 'https://www.reflectometry.org/projects/simple_model',
        'value': 2.2,
        'unit': 'dimensionless',
        # Non-negative: (mu, theta_m) and (-mu, theta_m + 180) are the same state,
        # the direction is the layer's theta_m.
        'min': 0.0,
        'max': 10.0,
        'fixed': True,
    },
    # Not named 'rho_m': calculators bind parameters by name, and a layer's
    # variables include its material's, so a 'rho_m' here would be bound to the
    # layer's magnetic SLD slot (read back from it, and written into it).
    'magnetic_sld': {
        'description': 'The magnetic scattering length density of the material in e-6 per squared angstrom.',
        'url': 'https://refl1d.readthedocs.io/en/latest/guide/magnetism.html',
        'value': 0.0,
        'unit': '1 / angstrom^2',
    },
}


def _moment_defaults(magnetic_moment) -> dict:
    """The moment's defaults, with the fit bounds widened to include an
    explicitly given value (the bounds are a fitting convenience)."""
    if not isinstance(magnetic_moment, Number) or isinstance(magnetic_moment, bool):
        return DEFAULTS
    defaults = deepcopy(DEFAULTS)
    entry = defaults['magnetic_moment']
    entry['min'] = min(entry['min'], float(magnetic_moment))
    entry['max'] = max(entry['max'], float(magnetic_moment))
    return defaults


# Number density times moment times the magnetic scattering length per Bohr
# magneton, in the library's 1e-6 SLD storage convention (see
# `SLD_DEPENDENCY_EXPRESSION`); the units are carried by the operands.
RHO_M_DEPENDENCY_EXPRESSION = 'na * d / mw * mu * p * 1e6'


class MaterialMagnetic(MaterialDensity):
    """A density-defined material carrying a magnetic moment.

    Adds :attr:`magnetic_moment` (µB per formula unit, the fit parameter) and
    a derived magnetic SLD

        rho_m = N * mu * p,    N = N_A * density / M

    with ``p`` the magnetic scattering length of one Bohr magneton
    (:data:`~easyreflectometry.special.calculations.MAGNETIC_SCATTERING_LENGTH`).
    The moment is the effective in-plane net moment per formula unit, which is
    the moment per atom only for a formula unit of one atom, such as Fe.

    A layer uses the material's ``rho_m`` when its
    :attr:`~easyreflectometry.sample.Layer.rho_m_coupled` is set. The magnetic
    SLD stays derived when :attr:`sld_coupled` is switched off, which affects
    the nuclear ``sld``/``isld`` only.
    """

    def __init__(
        self,
        chemical_structure: Union[str, None] = 'Fe',
        density: Union[Parameter, float, None] = 7.874,
        magnetic_moment: Union[Parameter, float, None] = None,
        name: str = 'EasyMaterialMagnetic',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        chemical_structure : Union[str, None], optional
            Chemical formula. By default, 'Fe'.
        density : Union[Parameter, float, None], optional
            Mass density in g/cm^3. By default, 7.874 (iron).
        magnetic_moment : Union[Parameter, float, None], optional
            Moment in Bohr magnetons per formula unit. By default, 2.2 (iron).
            The default fit bounds are [0, 10]; a number outside them widens
            the bounds to include it rather than being refused, since a
            signed moment (an imported reversed magnetic SLD) or a formula
            unit of many atoms is a valid state, not a fitting choice.
        name : str, optional
            Identifier. By default, 'EasyMaterialMagnetic'.
        unique_name : Optional[str], optional
            By default, None.
        interface :
            Interface object. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)
        super().__init__(
            chemical_structure=chemical_structure,
            density=density,
            name=name,
            unique_name=unique_name,
            interface=None,
        )
        self._magnetic_moment = get_as_parameter(
            name='magnetic_moment',
            value=magnetic_moment,
            default_dict=_moment_defaults(magnetic_moment),
            unique_name_prefix=f'{unique_name}_MagneticMoment',
        )
        self._rho_m = get_as_parameter(
            name='magnetic_sld',
            value=None,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_RhoM',
        )
        self._magnetic_scattering_length = DescriptorNumber(
            display_name='magnetic_scattering_length',
            value=MAGNETIC_SCATTERING_LENGTH,
            unit='angstrom',
            unique_name=global_object.generate_unique_name(f'{unique_name}_MagneticScatteringLength'),
        )
        self._setup_magnetic_dependency()

        if interface is not None:
            self.interface = interface

    def _setup_magnetic_dependency(self) -> None:
        """Wire `rho_m` to the current density, formula and moment.

        Idempotent, and independent of the nuclear coupling: called from
        `__init__` and, after the saved parameters are swapped in, from
        `from_dict` whatever `sld_coupled` is.
        """
        unconstrain(self._rho_m)
        self._rho_m.make_dependent_on(
            dependency_expression=RHO_M_DEPENDENCY_EXPRESSION,
            dependency_map={
                'na': self._avogadro,
                'd': self._density,
                'mw': self._molecular_weight,
                'mu': self._magnetic_moment,
                'p': self._magnetic_scattering_length,
            },
            desired_unit=MATERIAL_DEFAULTS['sld']['unit'],
        )

    @classmethod
    def from_dict(cls, obj_dict: dict) -> 'MaterialMagnetic':
        instance = super().from_dict(obj_dict)
        instance._setup_magnetic_dependency()
        return instance

    @property
    def magnetic_moment(self) -> Parameter:
        """Effective in-plane net moment in Bohr magnetons per formula unit."""
        return self._magnetic_moment

    @magnetic_moment.setter
    def magnetic_moment(self, value: float) -> None:
        self._magnetic_moment.value = value

    @property
    def rho_m(self) -> Parameter:
        """Magnetic SLD in 1e-6 / angstrom^2, derived from density, formula and moment (read-only)."""
        return self._rho_m

    @property
    def _dict_repr(self) -> dict[str, str]:
        mat_dict = super()._dict_repr
        mat_dict['magnetic_moment'] = f'{self._magnetic_moment.value:.3f} muB/f.u.'
        mat_dict['rho_m'] = f'{self._rho_m.value:.3f}e-6 {self._rho_m.unit}'
        return mat_dict
