# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from typing import Optional

from easyscience.base_classes import ModelBase

from easyreflectometry.sample import references
from easyreflectometry.utils import yaml_dump

#: What `Parameter.to_dict` adds to record a dependency (see `BaseCore._convert_to_dict`).
_CORE_DEPENDENCY_FIELDS = (
    '_dependency_string',
    '_dependency_map_serializer_ids',
    '_independent',
    '_desired_unit',
    '__serializer_id',
)


class BaseCore(ModelBase):
    """Local base class for sample-tree objects (Material, Layer, assemblies).

    Built on top of `easyscience.base_classes.ModelBase` (the replacement for
    the deprecated `ObjBase`). On top of `ModelBase` this class adds:

    - a `name` property
    - an `interface` property whose setter propagates the calculator interface
      to child objects and (re)generates bindings
    - a yaml-formatted `__repr__` driven by an abstract `_dict_repr`
    - an `as_dict` alias for `to_dict`

    Subclass `__init__` convention:
        1. Build child Parameters / sub-objects.
        2. Call ``super().__init__(name=..., unique_name=...)``.
        3. Assign children to backing fields (``self._sld = sld`` etc.) or pass
           them as ``**kwargs`` to this base class (transitional path; each
           kwarg is stored as a plain instance attribute).
        4. Last: ``self.interface = interface`` (triggers ``generate_bindings``).
    """

    def __init__(
        self,
        name: str,
        interface: Any = None,
        unique_name: Optional[str] = None,
        display_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(unique_name=unique_name, display_name=display_name)
        self._name = name
        self._interface = None
        # `user_data` is part of the legacy `BasedBase` API — a free-form dict
        # callers stash arbitrary metadata in. Kept for back-compat with code
        # like `Project.replace_models_from_orso` which stores the ORSO sample
        # name on the model.
        self.user_data: dict = {}

        # Transitional path: subclasses still pass parameter / child objects via
        # **kwargs (legacy `ObjBase` accepted them and stashed in `_kwargs`).
        # Here we simply store each one as a plain instance attribute so
        # `obj.<key>` keeps working. Step 2 of the migration replaces this with
        # explicit assignments in each subclass.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Assign interface LAST so children exist when generate_bindings runs.
        if interface is not None:
            self.interface = interface

    # ----- name -----

    @property
    def name(self) -> str:
        """Common (display-friendly) name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    # ----- interface -----

    @property
    def interface(self) -> Any:
        """The calculator interface attached to this object (may be None)."""
        return self._interface

    @interface.setter
    def interface(self, new_interface: Any) -> None:
        self._interface = new_interface
        if new_interface is not None:
            self.generate_bindings()

    def generate_bindings(self) -> None:
        """Propagate the interface to child objects, then bind via the calculator.

        We propagate to any child whose class advertises an ``interface`` property
        with a setter. That includes both the new `BaseCore`-based children and
        legacy `BasedBase`-derived collections (which extend `SerializerComponent`,
        not `EasyBase`).
        """
        if self._interface is None:
            raise AttributeError('Interface error for generating bindings. `interface` has to be set.')
        for attr in self._iter_public_children():
            if self._has_interface_setter(type(attr)):
                attr.interface = self._interface
        self._interface.generate_bindings(self)

    def _iter_public_children(self):
        """Yield public child objects from both class-level (dir) and instance-level (__dict__) attrs.

        `EasyBase.__dir__` exposes only class attributes, which means plain instance
        attributes (the transitional `setattr(self, key, value)` path in
        `__init__`) are invisible to a pure `dir()` scan. To bridge both worlds —
        legacy subclasses that still use plain attrs, and migrated subclasses that
        expose children via `@property` accessors — this helper unions the two.
        Once all subclasses migrate to `@property`-backed children with `_field`
        backing storage, the `__dict__` branch becomes a no-op (private names are
        skipped).
        """
        seen_ids = {id(self)}
        # Class-level (properties, methods named like sld/isld/material).
        for attr_name in dir(self):
            if attr_name.startswith('_') or attr_name in ('interface', 'name'):
                continue
            try:
                attr = getattr(self, attr_name, None)
            except AttributeError:
                # A subclass @property may legitimately raise AttributeError
                # mid-construction (the `_field` backing isn't set yet); skip
                # those entries silently. Other exceptions should propagate.
                continue
            if attr is None or id(attr) in seen_ids:
                continue
            seen_ids.add(id(attr))
            yield attr
        # Instance-level (plain attrs set via the transitional kwargs path).
        for attr_name, attr in list(self.__dict__.items()):
            if attr_name.startswith('_') or attr_name in ('interface', 'name'):
                continue
            if attr is None or id(attr) in seen_ids:
                continue
            seen_ids.add(id(attr))
            yield attr

    @staticmethod
    def _has_interface_setter(obj_type: type) -> bool:
        for klass in obj_type.__mro__:
            prop = klass.__dict__.get('interface')
            if isinstance(prop, property):
                return prop.fset is not None
        return False

    def get_all_variables(self):
        """Discover Parameters/Descriptors across both class-level and instance-level attrs.

        `ModelBase.get_all_variables` walks `dir(self)`, which `EasyBase` restricts
        to class attributes only. During the transition some subclasses still
        store child Parameters as plain instance attributes (see the kwargs path
        in :meth:`__init__`); those are invisible to `dir()`. We therefore also
        scan `self.__dict__` for `DescriptorBase` instances and for child
        ModelBase objects whose own `get_all_variables` we recurse into.
        """
        from easyscience.variable.descriptor_base import DescriptorBase

        out: list = []
        seen_param_ids: set[int] = set()
        for attr in self._iter_public_children():
            if isinstance(attr, DescriptorBase):
                if id(attr) not in seen_param_ids:
                    seen_param_ids.add(id(attr))
                    out.append(attr)
            elif hasattr(attr, 'get_all_variables'):
                for v in attr.get_all_variables():
                    if id(v) not in seen_param_ids:
                        seen_param_ids.add(id(v))
                        out.append(v)
        return out

    def to_dict(self, skip: Optional[list[str]] = None) -> dict[str, Any]:
        """Serialize, skipping the calculator interface and unique_name by default.

        The calculator (`CalculatorFactory`) is not serializable and is not part
        of the model's persistent state — round-trip code that needs it back
        reattaches it after `from_dict`. The legacy `ObjBase`-based pipeline
        achieved the same by never including `interface` in its `_kwargs`
        encoding; we replicate that here.

        `unique_name` is also stripped by default, matching the legacy
        `BasedBase.as_dict` contract. The installed `SerializerBase` does *not*
        propagate per-object `_default_unique_name` to nested children — if
        we leave it in, child Parameters end up with explicit unique_names in
        the dict (e.g. `Parameter_0`) that subsequently collide on reload when
        the global counter restarts from 0.

        Pass a *copy* of `skip` to super since `EasyBase.to_dict` mutates the
        list (appends `unique_name` / `display_name` if those are default).
        """
        skip = list(skip or [])
        if 'interface' not in skip:
            skip.append('interface')
        if 'unique_name' not in skip:
            skip.append('unique_name')
        return super().to_dict(skip=list(skip))

    def as_dict(self, skip: Optional[list[str]] = None) -> dict[str, Any]:
        """Compatibility alias for :meth:`to_dict`."""
        return self.to_dict(skip=skip)

    def _convert_to_dict(self, d: dict, encoder=None, skip: Optional[list[str]] = None, **kwargs) -> dict:
        """Serializer hook, called for this object at any depth.

        Drops easyscience's own dependency record from this object's parameters:
        the library persists dependencies itself (owning classes rebuild the
        internal ones, `Project` saves user constraints), and a second record
        would be restored on top of them. Then tags the object, or refers to
        its first occurrence (see :mod:`easyreflectometry.sample.references`).
        """
        for value in d.values():
            if isinstance(value, dict):
                for key in _CORE_DEPENDENCY_FIELDS:
                    value.pop(key, None)
        return references.tag(self, d)

    @classmethod
    def from_dict(cls, obj_dict: dict) -> BaseCore:
        """Rebuild, resolving shared children; see :mod:`easyreflectometry.sample.references`."""
        if references.is_reference(obj_dict):
            # Unresolvable reference (reported by `references.resolve`): a default object.
            obj_dict = {'@module': obj_dict['@module'], '@class': obj_dict['@class']}
        instance = super().from_dict(references.build_children(obj_dict))
        references.register(obj_dict, instance)
        return instance

    # ----- repr -----

    @property
    @abstractmethod
    def _dict_repr(self) -> dict[str, Any]: ...

    def __repr__(self) -> str:
        """Yaml-formatted multi-line string built from :attr:`_dict_repr`."""
        return yaml_dump(self._dict_repr)
