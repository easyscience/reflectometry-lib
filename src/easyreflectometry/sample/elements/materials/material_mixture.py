# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.special.calculations import weighted_average
from easyreflectometry.utils import get_as_parameter

from ...base_core import BaseCore
from .material import DEFAULTS as MATERIAL_DEFAULTS
from .material import Material

DEFAULTS = {
    'fraction': {
        'description': 'The fraction of material b in material a',
        'value': 0.5,
        'unit': 'dimensionless',
        'min': 0,
        'max': 1,
        'fixed': True,
    }
}
DEFAULTS.update(MATERIAL_DEFAULTS)


class MaterialMixture(BaseCore):
    def __init__(
        self,
        material_a: Union[Material, None] = None,
        material_b: Union[Material, None] = None,
        fraction: Union[Parameter, float, None] = None,
        name: Union[str, None] = None,
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        material_a : Union[Material, None], optional
            The first material. By default, None.
        material_b : Union[Material, None], optional
            The second material. By default, None.
        fraction : Union[Parameter, float, None], optional
            The fraction of material_b in material_a. By default, None.
        name : Union[str, None], optional
            Name of the material. By default, None.
        interface :
            Calculator interface. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if material_a is None:
            material_a = Material()
        if material_b is None:
            material_b = Material()

        fraction = get_as_parameter(
            name='fraction',
            value=fraction,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Fraction',
        )

        sld_value = weighted_average(
            a=material_a.sld.value,
            b=material_b.sld.value,
            p=fraction.value,
        )
        isld_value = weighted_average(
            a=material_a.isld.value,
            b=material_b.isld.value,
            p=fraction.value,
        )

        sld = get_as_parameter(
            name='sld',
            value=sld_value,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Sld',
        )
        isld = get_as_parameter(
            name='isld',
            value=isld_value,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Isld',
        )

        # `name` may be None to signal "derive from material names"; resolve
        # before super().__init__ since BaseCore stores `_name` directly.
        if name is None:
            resolved_name = material_a.name + '/' + material_b.name
        else:
            resolved_name = name

        super().__init__(name=resolved_name, unique_name=unique_name)
        self._material_a = material_a
        self._material_b = material_b
        self._fraction = fraction
        self._sld = sld
        self._isld = isld

        self._materials_constraints()

        if interface is not None:
            self.interface = interface

    # ----- constructor-arg accessors -----

    @property
    def material_a(self) -> Material:
        return self._material_a

    @material_a.setter
    def material_a(self, new_material_a: Material) -> None:
        self._material_a = new_material_a
        self._materials_constraints()
        self._update_name()

    @property
    def material_b(self) -> Material:
        return self._material_b

    @material_b.setter
    def material_b(self, new_material_b: Material) -> None:
        self._material_b = new_material_b
        self._materials_constraints()
        self._update_name()

    @property
    def fraction(self) -> Parameter:
        """The Parameter that controls the mixing fraction of material_b in material_a."""
        return self._fraction

    @fraction.setter
    def fraction(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError('fraction must be a float')
        self._fraction.value = value

    # ----- derived sld / isld parameters (shared shape with Material) -----
    #
    # These are *derived* via the constraints set up in `_materials_constraints`
    # (not constructor arguments) so we expose them as floats to match the
    # legacy MaterialMixture API. The underlying Parameter objects remain
    # available as `self._sld` / `self._isld`.

    @property
    def sld(self) -> float:
        return self._sld.value

    @property
    def isld(self) -> float:
        return self._isld.value

    # ----- internal helpers -----

    def _materials_constraints(self):
        """Wire the mixed `_sld` / `_isld` to depend on the current child
        material parameters and the current `_fraction`. Idempotent: callers
        invoke this once from ``__init__`` and again from ``from_dict`` after
        the saved Parameters have been reattached (so the dependency graph
        points at the right objects, not the temporary constructor params)."""
        # Detach any existing dependency before rebuilding so make_dependent_on
        # doesn't chain on top of stale references.
        for derived in (self._sld, self._isld):
            if not derived.independent:
                derived.make_independent()

        dependency_expression = 'a * (1 - p) + b * p'
        dependency_map = {
            'a': self._material_a.sld,
            'b': self._material_b.sld,
            'p': self._fraction,
        }
        self._sld.make_dependent_on(dependency_expression=dependency_expression, dependency_map=dependency_map)

        dependency_map = {
            'a': self._material_a.isld,
            'b': self._material_b.isld,
            'p': self._fraction,
        }
        self._isld.make_dependent_on(dependency_expression=dependency_expression, dependency_map=dependency_map)

    def _update_name(self) -> None:
        """Update name."""
        self.name = self._material_a.name + '/' + self._material_b.name

    # ----- deserialization -----

    @classmethod
    def from_dict(cls, obj_dict: dict) -> 'MaterialMixture':
        """Re-attach mixed-sld dependencies after :class:`ModelBase` swaps in
        the saved ``_fraction`` Parameter.

        :class:`ModelBase.from_dict` runs ``__init__`` (which builds the
        ``_sld`` / ``_isld`` constraints against the *temporary* ``_fraction``
        created from the float kwargs) and then re-points ``self._fraction``
        at the persisted Parameter. The constraint graph still references the
        temporary object, so subsequent ``mm.fraction = X`` mutations don't
        propagate to ``_sld`` / ``_isld``. Re-running ``_materials_constraints``
        here points the graph at the live objects.
        """
        instance = super().from_dict(obj_dict)
        instance._materials_constraints()
        return instance

    # Representation
    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        return {
            self.name: {
                'fraction': f'{self._fraction.value:.3f} {self._fraction.unit}',
                'sld': f'{self._sld.value:.3f}e-6 {self._sld.unit}',
                'isld': f'{self._isld.value:.3f}e-6 {self._isld.unit}',
                'material_a': self._material_a._dict_repr,
                'material_b': self._material_b._dict_repr,
            }
        }
