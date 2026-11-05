# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.constraints import USER_CONSTRAINT_FLAG
from easyreflectometry.constraints import unconstrain
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
        rho_m_coupled: bool = False,
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
        rho_m_coupled : bool, optional
            Take the magnetic SLD from the material (see :attr:`rho_m_coupled`).
            By default, False.
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
        self._rho_m_coupled = False
        if rho_m_coupled:
            self.rho_m_coupled = True

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
        coupled = self.rho_m_coupled
        self._material = value
        self._reconcile_rho_m_coupling(coupled)

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
        previous = self._magnetism
        coupled = self.rho_m_coupled
        if previous is not None and previous is not value:
            if value is None and self.interface is not None:
                self.interface().remove_layer_magnetism(self.unique_name)
            # The detached object must not follow the material any more.
            if coupled:
                _release(previous.rho_m)
        self._magnetism = value
        self._reconcile_rho_m_coupling(coupled)
        if value is not None and self.interface is not None:
            self._enable_calculator_magnetism()
            self.generate_bindings()

    @property
    def rho_m_coupled(self) -> bool:
        """Whether the layer's magnetic SLD is the material's ``rho_m``
        (a :class:`~easyreflectometry.sample.MaterialMagnetic`).

        When True, ``magnetism.rho_m`` is a dependent parameter following the
        current material; the layer keeps its own ``theta_m``. Opt-in: layers of
        one material may carry different net moments. Setting it attaches a
        default :class:`LayerMagnetism` when the layer has none, and requires a
        material with ``rho_m``. Uncoupling keeps the current value. Removing
        the magnetism, or a user ``unconstrain`` of ``rho_m``, uncouples.

        Only the layer's own wiring counts: a user constraint placed on
        ``rho_m`` (:func:`~easyreflectometry.constraints.constrain`) reads as
        uncoupled, is saved as the user constraint it is, and is left alone
        when the material is later replaced. Setting this to True explicitly
        replaces such a constraint with the coupling.
        """
        if not self._rho_m_coupled or self._magnetism is None:
            return False
        rho_m = self._magnetism.rho_m
        if rho_m.independent or getattr(rho_m, USER_CONSTRAINT_FLAG, False):
            return False
        source = getattr(self._material, 'rho_m', None)
        dependency_map = rho_m.dependency_map
        return source is not None and len(dependency_map) == 1 and dependency_map.get('rho_m') is source

    @rho_m_coupled.setter
    def rho_m_coupled(self, couple: bool) -> None:
        couple = bool(couple)
        if couple and getattr(self._material, 'rho_m', None) is None:
            raise ValueError(f'The material of layer {self.name!r} has no magnetic SLD to couple to.')
        if couple and self._magnetism is None:
            self.magnetism = LayerMagnetism()
        if not couple and self.rho_m_coupled:
            _release(self._magnetism.rho_m)
        self._reconcile_rho_m_coupling(couple)

    def _reconcile_rho_m_coupling(self, couple: bool) -> None:
        """Re-establish the invariant of :attr:`rho_m_coupled` after a change of
        material, magnetism or coupling: coupled means the current magnetism's
        ``rho_m`` follows the current material's ``rho_m``.

        `couple` is the state to keep, read *before* the change (afterwards the
        dependency may point at a replaced object).
        """
        self._rho_m_coupled = False
        if not couple or self._magnetism is None:
            return
        rho_m = self._magnetism.rho_m
        source = getattr(self._material, 'rho_m', None)
        if source is None:
            warnings.warn(
                f'The material of layer {self.name!r} has no magnetic SLD; '
                'the layer keeps its current rho_m and is no longer coupled.',
                UserWarning,
                stacklevel=3,
            )
            if not rho_m.independent:
                _release(rho_m)
            return
        # Drops any previous dependency, including a stale user-constraint marker
        # that would make the project persist this internal wiring.
        unconstrain(rho_m)
        rho_m.make_dependent_on(dependency_expression='rho_m', dependency_map={'rho_m': source})
        self._rho_m_coupled = True

    def assign_material(self, material: Material) -> None:
        """Assign a material to the layer interface.

        Parameters
        ----------
        material : Material
            The material to assign to the layer.
        """
        coupled = self.rho_m_coupled
        self._material = material
        self._reconcile_rho_m_coupling(coupled)
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


def _release(rho_m: Parameter) -> None:
    """Make a coupled magnetic SLD an ordinary parameter again, keeping its value.

    The dependency cleared `fixed` and narrowed the bounds; give back what a new
    `LayerMagnetism` starts with (fixed, project default limits), so uncoupling
    never silently adds a free parameter to the next fit.
    """
    rho_m.make_independent()
    rho_m.fixed = True
    rho_m.min, rho_m.max = -np.inf, np.inf
    rho_m.default_limits_pending = True
