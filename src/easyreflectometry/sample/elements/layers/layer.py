# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.utils import get_as_parameter

from ...base_core import BaseCore
from ..materials.material import Material
from .layer_magnetism import LayerMagnetism

DEFAULTS = {
    'thickness': {
        'description': 'The thickness of the layer in angstroms',
        'url': 'https://github.com/reflectivity/edu_outreach/blob/master/refl_maths/paper.tex',
        'value': 10.0,
        'unit': 'angstrom',
        'min': 0.0,
        'max': np.inf,
        'fixed': True,
    },
    'roughness': {
        'description': 'The interfacial roughness, Nevot-Croce, for the layer in angstroms.',
        'url': 'https://doi.org/10.1051/rphysap:01980001503076100',
        'value': 3.3,
        'unit': 'angstrom',
        'min': 0.0,
        'max': np.inf,
        'fixed': True,
    },
}


class Layer(BaseCore):
    def __init__(
        self,
        material: Union[Material, None] = None,
        thickness: Union[Parameter, float, None] = None,
        roughness: Union[Parameter, float, None] = None,
        name: str = 'EasyLayer',
        unique_name: Optional[str] = None,
        interface=None,
        magnetism: Union[LayerMagnetism, None] = None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        material : Union[Material, None], optional
            The material for the layer. By default, None.
        thickness : Union[Parameter, float, None], optional
            Layer thickness in Angstrom. By default, None.
        roughness : Union[Parameter, float, None], optional
            Upper roughness on the layer in Angstrom. By default, None.
        magnetism : Union[LayerMagnetism, None], optional
            Magnetic properties of the layer; None for a non-magnetic layer.
            By default, None.
        name : str, optional
            Name of the layer. By default, 'EasyLayer'.
        interface :
            Interface object. By default, None.
        """
        if material is None:
            material = Material(interface=interface)

        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        thickness_value = thickness
        thickness = get_as_parameter(
            name='thickness',
            value=thickness,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Thickness',
        )
        thickness.default_limits_pending = not isinstance(thickness_value, Parameter)

        roughness_value = roughness
        roughness = get_as_parameter(
            name='roughness',
            value=roughness,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Roughness',
        )
        roughness.default_limits_pending = not isinstance(roughness_value, Parameter)

        super().__init__(name=name, unique_name=unique_name)
        self._material = material
        self._thickness = thickness
        self._roughness = roughness
        self._magnetism = magnetism

        if interface is not None:
            self.interface = interface

    # ----- interface (override BaseCore's to switch on calculator magnetism) -----

    @BaseCore.interface.setter
    def interface(self, new_interface) -> None:
        """Set the interface; runs `generate_bindings` and, for a magnetic layer,
        enables magnetism on the calculator (raising if it does not support it).
        """
        BaseCore.interface.fset(self, new_interface)
        if new_interface is not None and self._magnetism is not None:
            self._enable_calculator_magnetism()

    def _enable_calculator_magnetism(self) -> None:
        """Turn on magnetism on the attached calculator.

        Raises `NotImplementedError` when the active calculator cannot model
        magnetic samples (e.g. refnx or bornagain).
        """
        self.interface().include_magnetism = True

    @property
    def material(self) -> Material:
        return self._material

    @material.setter
    def material(self, value: Material) -> None:
        self._material = value

    @property
    def thickness(self) -> Parameter:
        return self._thickness

    @thickness.setter
    def thickness(self, value: float) -> None:
        self._thickness.value = value

    @property
    def roughness(self) -> Parameter:
        return self._roughness

    @roughness.setter
    def roughness(self, value: float) -> None:
        self._roughness.value = value

    @property
    def magnetism(self) -> Optional[LayerMagnetism]:
        return self._magnetism

    @magnetism.setter
    def magnetism(self, value: Optional[LayerMagnetism]) -> None:
        """Attach or remove the magnetic properties of this layer.

        Attaching regenerates the calculator bindings so `rho_m`/`theta_m` become
        live on the backend; removing drops the layer's magnetic state from the
        backend (and switches calculator magnetism off entirely when this was the
        last magnetic layer).
        """
        if value is None and self._magnetism is not None:
            if self.interface is not None:
                self.interface().remove_layer_magnetism(self.unique_name)
            # Detach the calculator callbacks of the removed parameters so later
            # value changes on the detached object no longer reach the backend.
            # `property()` (fget/fset/fdel all None) is easyscience's own "no
            # callback" sentinel -- `Parameter.__copy__` sets it the same way --
            # so a later `fset` guard in `Parameter` cleanly no-ops.
            self._magnetism.rho_m._callback = property()
            self._magnetism.theta_m._callback = property()
        self._magnetism = value
        if value is not None and self.interface is not None:
            self._enable_calculator_magnetism()
            self.generate_bindings()

    def assign_material(self, material: Material) -> None:
        """Assign a material to the layer interface.

        Parameters
        ----------
        material : Material
            The material to assign to the layer.
        """
        self._material = material
        if self.interface is not None:
            self.interface().assign_material_to_layer(self.material.unique_name, self.unique_name)

    # Representation
    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        this_dict = {
            self.name: {
                'material': self.material._dict_repr,
                'thickness': f'{self.thickness.value:.3f} {self.thickness.unit}',
                'roughness': f'{self.roughness.value:.3f} {self.roughness.unit}',
            }
        }
        if self._magnetism is not None:
            this_dict[self.name]['magnetism'] = self._magnetism._dict_repr
        return this_dict
