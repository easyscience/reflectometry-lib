# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.utils import get_as_parameter

from ...base_core import BaseCore

DEFAULTS = {
    'rho_m': {
        'description': 'The magnetic scattering length density of the layer in e-6 per squared angstrom.',
        'url': 'https://refl1d.readthedocs.io/en/latest/guide/magnetism.html',
        'value': 0.0,
        'unit': '1 / angstrom^2',
        'min': -np.inf,
        'max': np.inf,
        'fixed': True,
    },
    'theta_m': {
        'description': 'The angle of the in-plane magnetic moment with respect to the beam direction in degrees. '
        'The default of 270 degrees aligns the moment with the default guide field (Aguide), giving no spin-flip.',
        'url': 'https://refl1d.readthedocs.io/en/latest/guide/magnetism.html',
        'value': 270.0,
        'unit': 'degree',
        'min': 0.0,
        'max': 360.0,
        'fixed': True,
    },
}


class LayerMagnetism(BaseCore):
    """Magnetic properties of a layer: magnetic SLD amplitude and in-plane moment angle.

    Attach to a `Layer` via its `magnetism` argument/property to make the layer magnetic.
    Both attributes are `Parameter` objects, so they can be fitted and serialized like any
    other sample parameter. A calculator that supports magnetism (refl1d) binds them as
    `magnetism_rhoM` / `magnetism_thetaM` on the corresponding slab.
    """

    def __init__(
        self,
        rho_m: Union[Parameter, float, None] = None,
        theta_m: Union[Parameter, float, None] = None,
        name: str = 'EasyLayerMagnetism',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        rho_m : Union[Parameter, float, None], optional
            Magnetic scattering length density in e-6 per squared angstrom. By default, 0.
        theta_m : Union[Parameter, float, None], optional
            In-plane moment angle in degrees. By default, 270 (aligned with the default
            guide field, i.e. no spin-flip).
        name : str, optional
            Name of the magnetism element. By default, 'EasyLayerMagnetism'.
        unique_name : Optional[str], optional
            By default, None.
        interface :
            Calculator interface. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        rho_m_value = rho_m
        rho_m = get_as_parameter(
            name='rho_m',
            value=rho_m,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_RhoM',
        )
        # The default bounds are infinite; `Project._sync_parameter_states`
        # narrows them to the shared SLD window unless the caller passed an
        # explicit Parameter with its own bounds (same contract as `Layer`).
        rho_m.default_limits_pending = not isinstance(rho_m_value, Parameter)
        theta_m = get_as_parameter(
            name='theta_m',
            value=theta_m,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_ThetaM',
        )

        super().__init__(name=name, unique_name=unique_name)
        self._rho_m = rho_m
        self._theta_m = theta_m

        if interface is not None:
            self.interface = interface

    @property
    def rho_m(self) -> Parameter:
        return self._rho_m

    @rho_m.setter
    def rho_m(self, value: float) -> None:
        self._rho_m.value = value

    @property
    def theta_m(self) -> Parameter:
        return self._theta_m

    @theta_m.setter
    def theta_m(self, value: float) -> None:
        self._theta_m.value = value

    # Representation
    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        return {
            self.name: {
                'rho_m': f'{self._rho_m.value:.3f}e-6 {self._rho_m.unit}',
                'theta_m': f'{self._theta_m.value:.3f} {self._theta_m.unit}',
            }
        }
