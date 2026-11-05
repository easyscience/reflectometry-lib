# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The storage-based calculator, kept only for refl1d until its stateless port (CORE-3-DESIGN.md, PR 3)."""

from __future__ import annotations

import weakref
from typing import Callable

from easyreflectometry.model import Model
from easyreflectometry.sample import BaseAssembly
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer

from .calculator_base import CalculatorBase
from .wrapper_base import WrapperBase


class StorageCalculatorBase(CalculatorBase, is_abstract=True):
    """A calculator whose wrapper keeps backend objects in ``storage``.

    The sample tree mirrors every structural edit into the storage (the
    ``add_*``/``remove_*``/``assign_*`` hooks), and parameter values are
    written into it before each evaluation (see :meth:`bind`).
    """

    _material_link: dict[str, str]
    _layer_link: dict[str, str]
    _item_link: dict[str, str]
    _model_link: dict[str, str]

    def __init__(self):
        """Init function."""
        super().__init__()
        self._wrapper: WrapperBase
        # Which easyscience object each backend entry was created for, keyed
        # like the wrapper storage (``(kind, unique_name)``). Weak so the
        # calculator keeps nothing alive. See `_claim`.
        self._owners: dict[tuple[str, str], weakref.ReferenceType] = {}
        # Backend entry -> {parameter label: (parameter, backend key, setter)}.
        # See `bind`.
        self._links: dict[str, dict[str, tuple[weakref.ReferenceType, str, Callable]]] = {}

    def reset_storage(self) -> None:
        r"""Reset the storage area of the calculator."""
        self._wrapper.reset_storage()
        self._owners = {}
        self._links = {}

    def register_model(self, model: Model) -> None:
        """Serve *model*: create its backend objects."""
        self.bind(model)

    def discard_model(self, model_id: str) -> None:
        """Stop serving the model with *model_id*."""
        self._wrapper.remove_model(model_id)
        self._links.pop(model_id, None)
        self._owners.pop(('model', model_id), None)

    def bind(self, model: Material | MaterialMixture | Layer | BaseAssembly | Model) -> None:
        """Create the backend entry of *model* and link its parameters to it."""
        for key, conversion, setter in self.create(model):
            parameters = {}
            for parameter in model._get_linkable_attributes():
                parameters.setdefault(parameter.display_name, parameter)
            links = {}
            for label, backend_key in conversion.items():
                if label in parameters:
                    links[label] = (weakref.ref(parameters[label]), backend_key, setter)
            self._links[key] = links
            self._push(key, links)

    def _before_evaluation(self) -> None:
        self._push_values()

    def _push_values(self) -> None:
        """Write the current value of every linked parameter into the backend."""
        for key, links in self._links.items():
            self._push(key, links)

    @staticmethod
    def _push(key: str, links: dict[str, tuple[weakref.ReferenceType, str, Callable]]) -> None:
        for parameter_ref, backend_key, setter in links.values():
            parameter = parameter_ref()
            if parameter is not None:
                setter(key, **{backend_key: parameter.value})

    def _claim(self, kind: str, model: Material | Layer | Multilayer | Model) -> None:
        """Make *model* the owner of its backend entry, discarding a stale one first.

        Backend entries are keyed by ``unique_name``, and easyscience hands the
        name of a dead object out again (the next ``Prefix_N`` above the live
        ones), while nothing tells the calculator that the object died. A new
        object can thus be bound to the entry its dead namesake left behind:
        ``create`` must not recreate the entry for an object that is merely
        re-bound (the entry is referenced from its item's stack), so without
        this it would keep the old one. Parameter values are pushed again on
        binding, but state that is only pushed when present is not -- a
        non-magnetic layer landing on a dead magnetic layer's name became
        magnetic (GitHub issue #426). Anything not owned by *model* itself is
        removed from the backend here, so ``create`` starts from a blank entry.
        """
        key = model.unique_name
        owner = self._owners.get((kind, key))
        if owner is not None and owner() is model:
            return
        if owner is not None or key in self._wrapper.storage[kind]:
            getattr(self._wrapper, f'remove_{kind}')(key)
        self._owners[(kind, key)] = weakref.ref(model)

    def create(self, model: Material | Layer | Multilayer | Model) -> list[tuple[str, dict[str, str], Callable]]:
        """Creation function.

        Parameters
        ----------
        model : Material | Layer | Multilayer | Model
            Object to be created.
        """
        r_list = []
        t_ = type(model)
        if issubclass(t_, Material):
            key = model.unique_name
            self._claim('material', model)
            if key not in self._wrapper.storage['material'].keys():
                self._wrapper.create_material(key)
            r_list.append((key, self._material_link, self._wrapper.update_material))
        elif issubclass(t_, MaterialMixture):
            key = model.unique_name
            self._claim('material', model)
            if key not in self._wrapper.storage['material'].keys():
                self._wrapper.create_material(key)
            r_list.append((key, self._material_link, self._wrapper.update_material))
        elif issubclass(t_, Layer):
            key = model.unique_name
            self._claim('layer', model)
            if key not in self._wrapper.storage['layer'].keys():
                self._wrapper.create_layer(key)
            r_list.append((key, self._layer_link, self._wrapper.update_layer))
            self.assign_material_to_layer(model.material.unique_name, key)
        elif issubclass(t_, BaseAssembly):
            key = model.unique_name
            self._wrapper.create_item(key)
            r_list.append((key, self._item_link, self._wrapper.update_item))
            for i in model.layers:
                self.add_layer_to_item(i.unique_name, model.unique_name)
        elif issubclass(t_, Model):
            key = model.unique_name
            # `create_model` starts blank anyway; claiming drops what else is
            # kept under the name (resolution function, cached cross-sections).
            self._claim('model', model)
            self._wrapper.create_model(key)
            r_list.append((key, self._model_link, self._wrapper.update_model))
            for i in model.sample:
                self.add_item_to_model(i.unique_name, key)
        return r_list

    def assign_material_to_layer(self, material_id: str, layer_id: str) -> None:
        """Assign a material to a layer.

        Parameters
        ----------
        material_id : str
            The material name.
        layer_id : str
            The layer name.
        """
        self._wrapper.assign_material_to_layer(material_id, layer_id)

    def add_layer_to_item(self, layer_id: str, item_id: str) -> None:
        """Add a layer to the item stack.

        Parameters
        ----------
        item_id : str
            The item id.
        layer_id : str
            The layer id.
        """
        self._wrapper.add_layer_to_item(layer_id, item_id)

    def remove_layer_from_item(self, layer_id: str, item_id: str) -> None:
        """Remove a layer from an item stack.

        Parameters
        ----------
        item_id : str
            The item id.
        layer_id : str
            The layer id.
        """
        self._wrapper.remove_layer_from_item(layer_id, item_id)

    def add_item_to_model(self, item_id: str, model_id: str) -> None:
        """Add a layer to the item stack.

        Parameters
        ----------
        item_id : str
            The item id.
        model_id : str
            The model id.
        """
        self._wrapper.add_item(item_id, model_id)

    def remove_item_from_model(self, item_id: str, model_id: str) -> None:
        """Remove an item from the model.

        Parameters
        ----------
        item_id : str
            The item id.
        model_id : str
            The model id.
        """
        self._wrapper.remove_item(item_id, model_id)

    def remove_layer_magnetism(self, layer_id: str) -> None:
        """Remove the magnetic state of one layer; disables `include_magnetism`
        when no magnetic layer is left.

        Parameters
        ----------
        layer_id : str
            The layer id.
        """
        self._links.get(layer_id, {}).pop('rho_m', None)
        self._links.get(layer_id, {}).pop('theta_m', None)
        self._wrapper.remove_layer_magnetism(layer_id)
