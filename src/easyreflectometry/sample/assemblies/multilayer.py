# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Optional
from typing import Union

from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer import Layer
from .base_assembly import BaseAssembly


class Multilayer(BaseAssembly):
    """A multi layer is build from a single or a list of `Layer` or `LayerCollection`.

    The multi layer will arrange the layers as slabs, allowing the reflectometry to be determined from them.
    The front layer is where the neutron beam starts in, it has an index of 0.

    More information about the usage of this assembly is available in the
    `multilayer documentation`_

    .. _`multilayer documentation`: ../sample/assemblies_library.html#multilayer
    """

    def __init__(
        self,
        layers: Union[Layer, list[Layer], LayerCollection, None] = None,
        name: str = 'EasyMultilayer',
        unique_name: Optional[str] = None,
        interface=None,
        type: str = 'Multi-layer',
        populate_if_none: Optional[bool] = True,
        conformal_thickness: bool = False,
        conformal_roughness: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        populate_if_none : Optional[bool], optional
            By default, True.
        unique_name : Optional[str], optional
            By default, None.
        layers : Union[Layer, list[Layer], LayerCollection, None], optional
            The layers that make up the multi-layer. By default, None.
        name : str, optional
            Name for multi layer. By default, 'EasyMultilayer'.
        interface :
            Calculator interface. By default, None.
        type : str, optional
            Type of the constructed instance. By default, 'Multi-layer'.
        conformal_thickness : bool, optional
            Tie every layer's thickness to the front layer's. Serialization
            reads the current graph state through the matching property, so
            the ties are rebuilt on ``from_dict``. By default, False.
        conformal_roughness : bool, optional
            Tie every layer's roughness to the front layer's, likewise
            persistent. By default, False.
        """
        if layers is None:
            if populate_if_none:
                layers = LayerCollection([Layer(interface=interface)])
            else:
                layers = LayerCollection()
        elif isinstance(layers, Layer):
            layers = LayerCollection(layers, name=layers.name)
        elif isinstance(layers, list):
            layers = LayerCollection(*layers, name='/'.join([layer.name for layer in layers]))
        # Needed to ensure an empty list is created when saving and instatiating the object as_dict -> from_dict
        # Else collisions might occur in global_object.map
        self.populate_if_none = False

        super().__init__(
            name=name,
            type=type,
            interface=interface,
            layers=layers,
            unique_name=unique_name,
        )
        if conformal_thickness:
            self.conformal_thickness = True
        if conformal_roughness:
            self.conformal_roughness = True

    def add_layer(self, *layers: tuple[Layer]) -> None:
        """Add a layer to the multi layer.

        Parameters
        ----------
        *layers : tuple[Layer]
            Layers to add to the multi layer.
        """
        for arg in layers:
            if issubclass(arg.__class__, Layer):
                self.layers.append(arg)
                if self.interface is not None:
                    self.interface().add_layer_to_item(arg.unique_name, self.unique_name)

    def duplicate_layer(self, idx: int) -> None:
        """Duplicate a given layer.

        Parameters
        ----------
        idx : int
            Index of layer to duplicate.
        """
        to_duplicate = self.layers[idx]
        duplicate_layer = Layer(
            material=to_duplicate.material,
            thickness=to_duplicate.thickness.value,
            roughness=to_duplicate.roughness.value,
            name=to_duplicate.name + ' duplicate',
        )
        self.add_layer(duplicate_layer)

    def remove_layer(self, idx: int) -> None:
        """Remove a layer from the item.

        Parameters
        ----------
        idx : int
            Index of layer to remove.
        """
        if self.interface is not None:
            self.interface().remove_layer_from_item(self.layers[idx].unique_name, self.unique_name)
        del self.layers[idx]

    # Representation
    @property
    def _dict_repr(self) -> dict:
        """A simplified dict representation."""
        return {self.name: self.layers._dict_repr}

    @classmethod
    def from_dict(cls, data: dict) -> Multilayer:
        """Create a Multilayer from a dictionary."""
        multilayer = super().from_dict(data)
        return multilayer
