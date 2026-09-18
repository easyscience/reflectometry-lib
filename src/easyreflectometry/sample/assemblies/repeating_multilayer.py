# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.utils import get_as_parameter

from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer import Layer
from .multilayer import Multilayer

DEFAULTS = {
    'repetitions': {
        'description': 'Number of repetitions of the given series of layers',
        'value': 1,
        'min': 1,
        'max': 9999,
        'fixed': True,
    }
}


class RepeatingMultilayer(Multilayer):
    """A repeating multi layer is build from a `Multilayer` and which it repeats
    for a given number of times. This enables a computational efficiency in many
    reflectometry engines as the operation can be performed for a single
    `Multilayer` and cheaply combined for the appropriate number of
    `repetitions`.

    More information about the usage of this assembly is available in the
    `repeating multilayer documentation`_

    .. _`repeating multilayer documentation`: ../sample/assemblies_library.html#repeatingmultilayer
    """

    def __init__(
        self,
        layers: Union[LayerCollection, Layer, list[Layer], None] = None,
        repetitions: Union[Parameter, int, None] = None,
        name: str = 'EasyRepeatingMultilayer',
        unique_name: Optional[str] = None,
        interface=None,
        populate_if_none: bool = True,
        conformal_thickness: bool = False,
        conformal_roughness: bool = False,
    ):
        """Constructor.

        Parameters
        ----------
        populate_if_none : bool, optional
            By default, True.
        unique_name : Optional[str], optional
            By default, None.
        layers : Union[LayerCollection, Layer, list[Layer], None], optional
            The layers that make up the multi-layer that will be repeated. By default, None.
        repetitions : Union[Parameter, int, None], optional
            Number of repetitions of the given series of layers. By default, None.
        name : str, optional
            Name for the repeating multi layer. By default, 'EasyRepeatingMultilayer'.
        interface :
            Calculator interface. By default, None.
        conformal_thickness : bool, optional
            Tie every layer's thickness to the front layer's (persisted). By default, False.
        conformal_roughness : bool, optional
            Tie every layer's roughness to the front layer's (persisted). By default, False.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if layers is None:
            if populate_if_none:
                layers = LayerCollection([Layer(interface=interface)])
            else:
                layers = LayerCollection()
        elif isinstance(layers, Layer):
            layers = LayerCollection(layers, name=layers.name)
        elif isinstance(layers, list):
            layers = LayerCollection(*layers, name='/'.join([layer.name for layer in layers]))

        repetitions = get_as_parameter(
            name='repetitions',
            value=repetitions,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Repetitions',
        )

        super().__init__(
            layers=layers,
            name=name,
            unique_name=unique_name,
            interface=None,
            type='Repeating Multi-layer',
            populate_if_none=False,
            conformal_thickness=conformal_thickness,
            conformal_roughness=conformal_roughness,
        )
        self._repetitions = repetitions

        if interface is not None:
            self.interface = interface

    @property
    def repetitions(self) -> Parameter:
        return self._repetitions

    @repetitions.setter
    def repetitions(self, value) -> None:
        self._repetitions.value = value

    # Representation
    @property
    def _dict_repr(self) -> dict:
        """A simplified dict representation."""
        d_dict = {self.name: self.layers._dict_repr}
        d_dict[self.name]['repetitions'] = float(self.repetitions.value)
        return d_dict
