# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Optional
from typing import Tuple

from easyreflectometry.model.model import COLORS
from easyreflectometry.sample.collections.base_collection import BaseCollection

from .model import Model


# Needs to be a function, elements are added to the global_object.map
def DEFAULT_ELEMENTS(interface):
    """Default elements."""
    return (Model(interface=interface),)


class ModelCollection(BaseCollection):
    def __init__(
        self,
        *models: Tuple[Model],
        name: str = 'Models',
        interface=None,
        unique_name: Optional[str] = None,
        populate_if_none: bool = True,
        next_color_index: Optional[int] = None,
        **kwargs,
    ):
        """Init function."""
        if not models:
            if populate_if_none:
                models = DEFAULT_ELEMENTS(interface)
            else:
                models = []

        # `_next_color_index` must exist before super().__init__ because each
        # `append` during construction routes through `_append_internal` →
        # `_advance_color_index`, which reads the attribute.
        self._next_color_index = next_color_index

        super().__init__(
            name,
            interface,
            *models,
            unique_name=unique_name,
            populate_if_none=False,
            **kwargs,
        )

        color_count = len(COLORS)
        if color_count == 0:
            self._next_color_index = 0
        elif next_color_index is None:
            self._next_color_index = len(self) % color_count
        else:
            self._next_color_index = next_color_index % color_count

    @property
    def next_color_index(self) -> Optional[int]:
        """Index of the next colour to assign — kept around so it round-trips."""
        return self._next_color_index

    def next_color(self) -> str:
        """Colour the next appended model should get.

        Appending advances the cycle, so callers that build a ``Model``
        themselves can keep the per-model colours distinct::

            model = Model(sample=sample, color=collection.next_color())
            collection.add_model(model)
        """
        return self._current_color()

    def add_model(self, model: Optional[Model] = None):
        """Add a model to the collection.

        Parameters
        ----------
        model : Optional[Model], optional
            Model to add. By default, None (a new model is created with
            the collection's next colour; a supplied model keeps its own
            colour — use :meth:`next_color` when building one).
        """
        if model is None:
            model = Model(name='Model', interface=self.interface, color=self._current_color())
        self.append(model)

    def duplicate_model(self, index: int):
        """Duplicate a model in the collection.

        Parameters
        ----------
        index : int
            Model to duplicate.
        """
        to_be_duplicated = self[index]
        duplicate = Model.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        duplicate.name = duplicate.name + ' duplicate'
        # A duplicate sharing its source's colour would be indistinguishable
        # in the plots; give it the collection's next colour instead.
        duplicate.color = self._current_color()
        self.append(duplicate)

    @classmethod
    def from_dict(cls, this_dict: dict) -> ModelCollection:
        """Create an instance of a collection from a dictionary."""
        collection_dict = dict(this_dict)
        dict_data = collection_dict.pop('data', [])
        next_color_index = collection_dict.pop('next_color_index', None)

        # Reconstruct empty collection via EasyList.from_dict (handles
        # protected_types and assigns name/unique_name/populate_if_none).
        collection = super().from_dict(collection_dict)

        # Append each model without advancing the colour index — the saved
        # `next_color_index` below is the source of truth.
        for model_data in dict_data:
            collection._append_internal(Model.from_dict(model_data), advance=False)

        if len(collection) != len(dict_data):
            raise ValueError(f'Expected {len(dict_data)} models, got {len(collection)}')

        color_count = len(COLORS)
        if color_count == 0:
            collection._next_color_index = 0
        elif next_color_index is None:
            collection._next_color_index = len(collection) % color_count
        else:
            collection._next_color_index = next_color_index % color_count

        return collection

    def append(self, model: Model) -> None:  # type: ignore[override]
        """Append function."""
        self._append_internal(model, advance=True)

    def _append_internal(self, model: Model, advance: bool) -> None:
        """Append internal."""
        # Bypass our own `append` override and go straight to EasyList's
        # `MutableSequence.append` → `insert` path. Calling `super().append`
        # would dispatch back to `ModelCollection.append` because Python
        # resolves `append` via MRO from MutableSequence which doesn't
        # define it on a class higher than ModelCollection.
        from collections.abc import MutableSequence

        MutableSequence.append(self, model)
        if advance:
            self._advance_color_index()

    def _advance_color_index(self) -> None:
        """Advance color index."""
        if not COLORS:
            self._next_color_index = 0
            return
        if self._next_color_index is None:
            self._next_color_index = len(self) % len(COLORS)
            return
        self._next_color_index = (self._next_color_index + 1) % len(COLORS)

    def _current_color(self) -> str:
        """Current color."""
        if not COLORS:
            raise ValueError('No colors defined for models.')
        if self._next_color_index is None:
            self._next_color_index = 0
        return COLORS[self._next_color_index]
