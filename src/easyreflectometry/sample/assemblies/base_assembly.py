# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from ..base_core import BaseCore
from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer import Layer


def follows_equal(follower, leader) -> bool:
    """Whether `follower` is constrained to be exactly equal to `leader`."""
    if getattr(follower, 'independent', True):
        return False
    dependency_map = getattr(follower, '_dependency_map', None) or {}
    expression = (getattr(follower, '_clean_dependency_string', None) or '').strip()
    return len(dependency_map) == 1 and expression in dependency_map and dependency_map[expression] is leader


class BaseAssembly(BaseCore):
    """Assembly of layers.

    The front layer (front_layer) is the layer the neutron beam starts in, it has an index of 0.
    The back layer (back_layer) is the final layer from which the unreflected neutron beam is transmitted,
    its index number depends on the number of finite layers in the system, but it might be accessed at index -1.
    """

    def __init__(
        self,
        name: str,
        type: str,
        interface,
        layers: LayerCollection,
        unique_name: Optional[str] = None,
    ):
        super().__init__(name=name, unique_name=unique_name)
        self._layers = layers

        # Type is needed when fitting in easyscience
        self._type = type
        self._roughness_constraints_setup = False
        self._thickness_constraints_setup = False

        if interface is not None:
            self.interface = interface

    @property
    def layers(self) -> LayerCollection:
        return self._layers

    @layers.setter
    def layers(self, value: LayerCollection) -> None:
        self._layers = value

    @property
    def type(self) -> str:
        """Get type of the assembly.

        Needed by the GUI.
        """
        return self._type

    @property
    def front_layer(self) -> Optional[Layer]:
        """Get the front layer in the assembly."""
        if len(self.layers) == 0:
            return None
        return self.layers[0]

    @front_layer.setter
    def front_layer(self, layer: Layer) -> None:
        """Set the front layer in the assembly.

        Parameters
        ----------
        layer : Layer
            Layer to set as the front layer.
        """
        if len(self.layers) == 0:
            self.layers.append(layer)
        else:
            self.layers[0] = layer

    @property
    def back_layer(self) -> Optional[Layer]:
        """Get the back layer in the assembly."""

        if len(self.layers) < 2:
            return None
        return self.layers[-1]

    @back_layer.setter
    def back_layer(self, layer: Layer) -> None:
        """Set the back layer in the assembly.

        Parameters
        ----------
        layer : Layer
            Layer to set as the back layer.
        """

        if len(self.layers) == 0:
            raise Exception('There is no front layer to add the back layer to. Please add a front layer first.')
        if len(self.layers) == 1:
            self.layers.append(layer)
        else:
            self.layers[-1] = layer

    # ----- public constraint toggles -----

    def _layers_follow_front(self, attribute: str) -> bool:
        """True when every layer's `attribute` is tied *equal* to the front layer's.

        Only the conformal idiom counts (expression ``'a'`` over ``{'a': leader}``,
        as set up by ``_setup_*_constraints``); a parameter that merely *uses*
        the front layer's value in some other expression is not conformal.
        """
        if len(self.layers) < 2:
            return False
        leader = getattr(self.front_layer, attribute)
        for layer in list(self.layers)[1:]:
            follower = getattr(layer, attribute)
            if not follows_equal(follower, leader):
                return False
        return True

    @property
    def conformal_thickness(self) -> bool:
        """Whether every layer shares the front layer's thickness."""
        return self._layers_follow_front('thickness')

    @conformal_thickness.setter
    def conformal_thickness(self, status: bool) -> None:
        """Tie (or release) every layer's thickness to the front layer's."""
        if status:
            self._setup_thickness_constraints()
        elif self._layers_follow_front('thickness'):
            for layer in list(self.layers)[1:]:
                layer.thickness.make_independent()

    @property
    def conformal_roughness(self) -> bool:
        """Whether every layer shares the front layer's roughness."""
        return self._layers_follow_front('roughness')

    @conformal_roughness.setter
    def conformal_roughness(self, status: bool) -> None:
        """Tie (or release) every layer's roughness to the front layer's."""
        if status:
            self._setup_roughness_constraints()
        elif self._layers_follow_front('roughness'):
            self._disable_roughness_constraints()

    def _setup_thickness_constraints(self) -> None:
        """Setup thickness constraint, front layer is the deciding layer."""
        independent_param = self.front_layer.thickness
        for i in range(1, len(self.layers)):
            self.layers[i].thickness.make_dependent_on(dependency_expression='a', dependency_map={'a': independent_param})
        self._thickness_constraints_setup = True

    def _enable_thickness_constraints(self):
        """Enable the thickness constraint."""
        if self._thickness_constraints_setup:
            # Make sure that the thickness constraint is enabled
            self._setup_thickness_constraints()
            # Make sure that the thickness parameter is enabled
        else:
            raise Exception('Thickness constraints not setup')

    def _disable_thickness_constraints(self):
        """Disable the thickness constraint."""
        if self._thickness_constraints_setup:
            for i in range(1, len(self.layers)):
                self.layers[i].thickness.make_independent()
        else:
            raise Exception('Thickness constraints not setup')

    def _setup_roughness_constraints(self) -> None:
        """Setup roughness constraint, front layer is the deciding layer."""
        independent_parameter = self.front_layer.roughness
        for i in range(1, len(self.layers)):
            self.layers[i].roughness.make_dependent_on(dependency_expression='a', dependency_map={'a': independent_parameter})
        self._roughness_constraints_setup = True

    def _enable_roughness_constraints(self):
        """Enable the roughness constraint."""
        independent_parameter = self.front_layer.roughness
        for i in range(1, len(self.layers)):
            self.layers[i].roughness.make_dependent_on(dependency_expression='a', dependency_map={'a': independent_parameter})

    def _disable_roughness_constraints(self):
        """Disable the roughness constraint."""
        for i in range(1, len(self.layers)):
            self.layers[i].roughness.make_independent()
