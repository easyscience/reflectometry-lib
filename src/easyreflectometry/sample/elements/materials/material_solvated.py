# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.utils import get_as_parameter

from .material import Material
from .material_mixture import MaterialMixture

DEFAULTS = {
    'solvent_fraction': {
        'description': 'Fraction of solvent in layer.',
        'value': 0.2,
        'unit': 'dimensionless',
        'min': 0,
        'max': 1,
        'fixed': True,
    },
}


class MaterialSolvated(MaterialMixture):
    def __init__(
        self,
        material: Union[Material, None] = None,
        solvent: Union[Material, None] = None,
        solvent_fraction: Union[Parameter, float, None] = None,
        name=None,
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        material : Union[Material, None], optional
            The material being solvated. By default, None.
        solvent : Union[Material, None], optional
            The solvent material. By default, None.
        solvent_fraction : Union[Parameter, float, None], optional
            Fraction of solvent in layer. E.g. solvation or surface coverage. By default, None.
        name :
            Name of the material. By default, None.
        interface :
            Calculator interface. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if material is None:
            material = Material(sld=6.36, isld=0, name='D2O', interface=interface)
        if solvent is None:
            solvent = Material(sld=-0.561, isld=0, name='H2O', interface=interface)

        solvent_fraction = get_as_parameter(
            name='solvent_fraction',
            value=solvent_fraction,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Fraction',
        )

        # In super class, the fraction is the fraction of material b in material a
        super().__init__(
            material_a=material,
            material_b=solvent,
            fraction=solvent_fraction,
            name=name,
            unique_name=unique_name,
            interface=interface,
        )
        if name is None:
            self._update_name()

    @property
    def material(self) -> Material:
        """Get material."""
        return self._material_a

    @material.setter
    def material(self, new_material: Material) -> None:
        """Set the material."""
        self.material_a = new_material

    @property
    def solvent(self) -> Material:
        """Get solvent."""
        return self._material_b

    @solvent.setter
    def solvent(self, new_solvent: Material) -> None:
        """Set the solvent."""
        self.material_b = new_solvent

    @property
    def solvent_fraction_parameter(self) -> Parameter:
        """Get the parameter for the fraction of layer described by the solvent."""
        return self._fraction

    @property
    def solvent_fraction(self) -> Parameter:
        """The Parameter for the fraction of the layer described by the solvent.

        This might be the fraction of:
        - solvation where solvent is within the layer, or
        - patches of solvent in the layer where no material is present.
        """
        return self._fraction

    @solvent_fraction.setter
    def solvent_fraction(self, solvent_fraction: float) -> None:
        """Set the fraction of layer covered by the material."""
        if not isinstance(solvent_fraction, (int, float)):
            raise ValueError('solvent_fraction must be a float between 0 and 1')
        if solvent_fraction < 0 or solvent_fraction > 1:
            raise ValueError('solvent_fraction must be between 0 and 1')
        self._fraction.value = solvent_fraction

    def _update_name(self) -> None:
        """Update name."""
        self.name = self._material_a.name + ' in ' + self._material_b.name

    # ----- deserialization -----

    @classmethod
    def from_dict(cls, obj_dict: dict) -> 'MaterialSolvated':
        """Re-route the saved ``solvent_fraction`` Parameter onto ``_fraction``.

        :class:`ModelBase.from_dict` writes the saved Parameter to
        ``_solvent_fraction`` because that's the constructor-arg name, but
        the live `solvent_fraction` property returns ``self._fraction``
        (the field MaterialMixture maintains). Without this override the
        saved fit metadata (fixed/bounds/etc.) is stranded on the unused
        ``_solvent_fraction`` attribute and the active parameter keeps the
        defaults from `__init__`.

        Also re-runs `_materials_constraints` so the parent MaterialMixture's
        mixed `_sld` / `_isld` depend on the live `_fraction`, not the
        temporary Parameter created from the float kwarg.
        """
        instance = super().from_dict(obj_dict)
        saved = instance.__dict__.pop('_solvent_fraction', None)
        if saved is not None:
            old = instance._fraction
            instance._fraction = saved
            try:
                instance._global_object.registry.discard(old.unique_name)
            except (AttributeError, KeyError):
                pass
            instance._materials_constraints()
        return instance

    # Representation
    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        return {
            self.name: {
                'solvent_fraction': f'{self._fraction.value:.3f} {self._fraction.unit}',
                'sld': f'{self._sld.value:.3f}e-6 {self._sld.unit}',
                'isld': f'{self._isld.value:.3f}e-6 {self._isld.unit}',
                'material': self.material._dict_repr,
                'solvent': self.solvent._dict_repr,
            }
        }
