# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any
from typing import List
from typing import Optional

from easyscience.base_classes import EasyList
from easyscience.base_classes.new_base import NewBase
from easyscience.io import SerializerBase
from easyscience.variable import Parameter

from easyreflectometry.sample import references
from easyreflectometry.utils import yaml_dump


class BaseCollection(EasyList):
    """Local base for sample-tree collections (Material/Layer/Assembly/Model collections).

    Built on top of `easyscience.base_classes.EasyList` (the replacement for
    the deprecated `CollectionBase`). On top of `EasyList` this class adds:

    - a `name` property
    - an `interface` property whose setter propagates the calculator interface
      to every contained item
    - propagation of the current `interface` to newly inserted items
    - a `populate_if_none` flag preserved for serialization round-trip
    - convenience helpers `names`, `move_up`, `move_down`, `remove_at`
    - a yaml-formatted `__repr__` driven by `_dict_repr`
    - an `as_dict` alias for `to_dict` with `skip=` support and `interface`
      excluded by default

    Subclasses (`LayerCollection`, `MaterialCollection`, `Sample`,
    `ModelCollection`) keep their existing constructor shape — they pass
    `name` and `interface` positionally to this class, items as `*args`, and
    additional configuration as kwargs.
    """

    def __init__(
        self,
        name: str,
        interface: Any = None,
        *args: Any,
        unique_name: Optional[str] = None,
        populate_if_none: bool = False,
        **kwargs: Any,
    ):
        # `_interface` must exist before `super().__init__` because `EasyList`
        # calls `self.append(item)` for each positional arg, which routes
        # through our `insert` override and reads `self._interface`.
        self._interface = None
        self._name = name
        # Legacy `CollectionBase` accepted items either positionally or as a
        # list-valued keyword (e.g. ``LayerCollection(layers=[a, b])``). Pull
        # any list-valued kwarg into the positional stream so callers using
        # that older pattern keep working. `data` and `protected_types` are
        # `EasyList`'s own list-valued arguments (`from_dict` passes the
        # latter as a list of classes) and must reach it untouched.
        extra_items = []
        for key in list(kwargs.keys()):
            if isinstance(kwargs[key], list) and kwargs[key] and key not in ('data', 'protected_types'):
                extra_items.extend(kwargs.pop(key))
        if extra_items:
            args = tuple(args) + tuple(extra_items)
        super().__init__(*args, unique_name=unique_name, **kwargs)
        # `populate_if_none` is a control flag, not state. It is serialized so
        # `from_dict` knows whether the original construction filled in
        # defaults; the value should be `False` once the data is restored.
        self.populate_if_none = populate_if_none

        # Assign interface LAST — propagates to all contained items.
        if interface is not None:
            self.interface = interface

    # ----- name -----

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    # ----- interface -----

    @property
    def interface(self) -> Any:
        return self._interface

    @interface.setter
    def interface(self, new_interface: Any) -> None:
        self._interface = new_interface
        if new_interface is None:
            return
        # Propagate to existing items.
        for item in self._data:
            if self._has_interface_setter(type(item)):
                item.interface = new_interface
        # Tell the calculator to bind to self (matches the legacy CollectionBase
        # behavior which called `interface.generate_bindings(self)` once per
        # collection).
        if hasattr(new_interface, 'generate_bindings'):
            new_interface.generate_bindings(self)

    def _get_key(self, obj):
        """Use the item's `name` for string-indexed lookups.

        Matches the legacy `CollectionBase.__getitem__` behaviour which
        searched by `item.name`. `EasyList` defaults to `unique_name`; the
        existing `Project` code (and callers) look items up by their pretty
        name (e.g. `materials['Air']`).
        """
        return getattr(obj, 'name', None) or obj.unique_name

    @staticmethod
    def _has_interface_setter(obj_type: type) -> bool:
        for klass in obj_type.__mro__:
            prop = klass.__dict__.get('interface')
            if isinstance(prop, property):
                return prop.fset is not None
        return False

    # ----- mutable-sequence overrides that propagate interface -----

    def insert(self, index: int, value: Any) -> None:
        """Insert and (if an interface is set) propagate it to the new item.

        Legacy `CollectionBase.insert` did `value.interface = self.interface`
        after registering the item; we replicate the same behaviour here so
        downstream calculator state stays in sync when items are appended
        after the collection's interface was already set.

        The type check from `EasyList.insert` is bypassed: each subclass
        accepts a single item type (Layer / Material / BaseAssembly / Model)
        and enforces it elsewhere, while the EasyList check would require
        every item to be a `NewBase` subclass — which was historically not
        guaranteed and forces an extra coupling we don't need.
        """
        if not isinstance(index, int):
            raise TypeError('Index must be an integer')
        # Skip the EasyList protected-types check; mimic the rest of its insert
        # (duplicate-name warning + append-to-_data).
        import warnings as _warnings

        if value in self:
            _warnings.warn(f'Item with unique name "{self._get_key(value)}" already in collection, it will be ignored')
            return
        self._data.insert(index, value)
        if self._interface is not None and self._has_interface_setter(type(value)):
            value.interface = self._interface

    # ----- helpers -----

    @property
    def names(self) -> list:
        """List of item names."""
        return [getattr(item, 'name', None) for item in self._data]

    @property
    def data(self) -> list:
        """Read-only view of the underlying item list.

        Provided for compatibility with code (and tests) written against the
        legacy `CollectionBase` shape, which exposed items via `.data`.
        """
        return list(self._data)

    def move_up(self, index: int) -> None:
        """Move the element at the given index up in the collection."""
        if index == 0:
            return
        self.insert(index - 1, self.pop(index))

    def move_down(self, index: int) -> None:
        """Move the element at the given index down in the collection."""
        if index == len(self) - 1:
            return
        self.insert(index + 1, self.pop(index))

    def remove_at(self, index: int) -> None:
        """Remove the item at *index* from the collection.

        Renamed from the legacy `BaseCollection.remove(index)` which shadowed
        `MutableSequence.remove(value)` (remove-by-value, inherited from
        `EasyList`). Callers that meant "remove by index" should use this; the
        standard `remove(value)` is still available with its usual semantics.
        """
        self.pop(index)

    # ----- compatibility shims (kept until call sites migrate) -----

    def get_parameters(self) -> List[Parameter]:
        """Compatibility alias for legacy callers; prefer `get_all_parameters`."""
        return self.get_all_parameters()

    def get_all_variables(self) -> List:
        """Flat list of every Parameter/Descriptor across all items.

        Walks each item in the collection and unions whatever each item
        exposes via its own `get_all_variables` (for `ModelBase` /
        `BaseCore` children) or, for objects that lack that hook,
        unions any direct `DescriptorBase` attributes.
        """
        from easyscience.variable.descriptor_base import DescriptorBase

        out: list = []
        seen: set[int] = set()
        for item in self._data:
            if hasattr(item, 'get_all_variables'):
                for v in item.get_all_variables():
                    if id(v) not in seen:
                        seen.add(id(v))
                        out.append(v)
            elif isinstance(item, DescriptorBase):
                if id(item) not in seen:
                    seen.add(id(item))
                    out.append(item)
        return out

    def get_all_parameters(self) -> List[Parameter]:
        return [v for v in self.get_all_variables() if isinstance(v, Parameter)]

    def get_free_parameters(self) -> List[Parameter]:
        return [p for p in self.get_all_parameters() if p.independent and not p.fixed]

    def get_fit_parameters(self) -> List[Parameter]:
        """Alias kept for the minimizer; matches `ModelBase.get_fit_parameters`."""
        return self.get_free_parameters()

    def _get_linkable_attributes(self) -> List[Parameter]:
        """Bridge for `easyscience.fitting.calculators.interface_factory.generate_bindings`."""
        return self.get_all_variables()

    # ----- repr -----

    @property
    def _dict_repr(self) -> dict:
        """A simplified dict representation."""
        return {self.name: [getattr(i, '_dict_repr', repr(i)) for i in self._data]}

    def __repr__(self) -> str:
        try:
            return yaml_dump(self._dict_repr)
        except Exception:
            return super().__repr__()

    # ----- serialization -----

    def _convert_to_dict(self, d: dict, encoder=None, skip: Optional[List[str]] = None, **kwargs) -> dict:
        """Serializer hook used when this collection is encoded as a *child*
        attribute (e.g. `Multilayer.layers`).

        `SerializerBase._convert_to_dict` iterates `_arg_spec` to populate the
        base dict and then calls `obj._convert_to_dict(d, ...)` if defined.
        Because `data` is supplied via `*args` (VAR_POSITIONAL — not part of
        `_arg_spec`), without this hook the nested encoding would miss the
        items entirely and round-trip would reconstruct an empty collection.
        """
        if skip is None:
            skip = []
        if self._protected_types != [NewBase] and 'protected_types' not in d:
            d['protected_types'] = [{'@module': c.__module__, '@class': c.__name__} for c in self._protected_types]
        # Encode each item. Defer to the encoder's recursive walk so nested
        # ModelBase / NewBase items get properly serialized.
        item_skip = list(skip)
        items: list = []
        for item in self._data:
            if encoder is not None and hasattr(encoder, '_recursive_encoder'):
                items.append(encoder._recursive_encoder(item, skip=item_skip, encoder=encoder, full_encode=False))
            elif hasattr(item, 'to_dict'):
                try:
                    items.append(item.to_dict(skip=list(item_skip)))
                except TypeError:
                    items.append(item.to_dict())
            else:
                items.append(item)
        d['data'] = items
        return d

    def to_dict(self, skip: Optional[List[str]] = None) -> dict:
        """Serialize with `skip` support; `interface` excluded by default.

        `EasyList.to_dict` doesn't accept a `skip` argument and is hard-wired
        to dump `data` plus the parent's `_arg_spec` view. We reimplement here
        so existing callers (`Project`, `Model.as_dict`, etc.) can keep
        passing `skip=['unique_name']` or similar.

        ``NewBase.to_dict`` mutates the ``skip`` list in-place (e.g. it
        appends ``'unique_name'`` when the collection's own unique_name is
        default-generated). We therefore pass a *copy* to it, otherwise the
        mutation would leak into the per-item serialization below and force
        every item Parameter dict to drop its ``unique_name`` — breaking the
        from_dict round-trip.
        """
        skip = list(skip or [])
        if 'interface' not in skip:
            skip.append('interface')
        # Matches legacy `BasedBase.as_dict`: drop unique_name from the
        # serialized form so nested Parameters don't get explicit names that
        # would collide with the auto-generated names produced when the global
        # counter restarts during reconstruction.
        if 'unique_name' not in skip:
            skip.append('unique_name')
        # The items this encodes (through `_convert_to_dict`) are replaced
        # below, so they must not be registered as shared-object occurrences.
        with references.suspended():
            dict_repr = NewBase.to_dict(self, skip=list(skip))
        if self._protected_types != [NewBase]:
            dict_repr['protected_types'] = [{'@module': c.__module__, '@class': c.__name__} for c in self._protected_types]
        dict_repr['data'] = []
        for item in self._data:
            # Items that are ModelBase / BaseCore subclasses accept `skip`;
            # other shapes use their no-arg `to_dict`/`as_dict`.
            if hasattr(item, 'to_dict'):
                try:
                    dict_repr['data'].append(item.to_dict(skip=list(skip)))
                except TypeError:
                    dict_repr['data'].append(item.to_dict())
            else:
                dict_repr['data'].append(item.as_dict(skip=list(skip)))
        return dict_repr

    def as_dict(self, skip: Optional[List[str]] = None) -> dict:
        """Compatibility alias for :meth:`to_dict`."""
        return self.to_dict(skip=skip)

    @classmethod
    def from_dict(cls, obj_dict: dict) -> BaseCollection:
        """Rebuild the collection, each item through its own class's ``from_dict``.

        ``EasyList.from_dict`` deserializes the ``data`` list with the
        serializer's generic path, which calls every item's constructor with
        the stored keys. An item whose class restores state in ``from_dict``
        (e.g. ``MaterialDensity`` and its ``sld_coupled`` key, or any nested
        item that does) then fails or loses that state. Apart from the items,
        this mirrors ``EasyList.from_dict``.
        """
        if not isinstance(obj_dict, dict) or '@module' not in obj_dict or '@class' not in obj_dict:
            raise ValueError('Input must be a dictionary representing an EasyScience collection.')
        if obj_dict['@class'] != cls.__name__:
            raise ValueError(f'Class name in dictionary does not match the expected class: {cls.__name__}.')
        temp_dict = copy.deepcopy(obj_dict)
        protected_types = None
        if 'protected_types' in temp_dict:
            protected_types = []
            for type_dict in temp_dict.pop('protected_types'):
                if '@module' not in type_dict or '@class' not in type_dict:
                    raise ValueError('Each protected type must be a serialized EasyScience class with @module and @class keys')
                module = __import__(type_dict['@module'], globals(), locals(), [type_dict['@class']], 0)
                if not hasattr(module, type_dict['@class']):
                    raise ImportError(f'Could not import class {type_dict["@class"]} from module {type_dict["@module"]}')
                protected_types.append(getattr(module, type_dict['@class']))
        raw_items = temp_dict.pop('data', [])
        kwargs = SerializerBase.deserialize_dict(temp_dict)
        data = [references.deserialize(item) for item in raw_items]
        return cls(data, protected_types=protected_types, **kwargs)

    def __deepcopy__(self, memo):
        """Round-trip via dict-skip-unique to get a fresh copy.

        `NewBase.__copy__` already does this; the override is kept (rather
        than deleted) to mirror the legacy `BaseCollection.__deepcopy__`
        semantics — callers that relied on `copy.deepcopy(collection)` still
        get a clone built from `from_dict(as_dict(skip=['unique_name']))`.
        """
        return self.from_dict(self.as_dict(skip=['unique_name']))
