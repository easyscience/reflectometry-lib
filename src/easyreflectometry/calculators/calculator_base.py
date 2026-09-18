# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABCMeta
from typing import Callable

import numpy as np
from easyscience.fitting.calculators.interface_factory import ItemContainer
from easyscience.io import SerializerComponent

# if TYPE_CHECKING:
from easyreflectometry.model import Model
from easyreflectometry.sample import BaseAssembly
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer

from .wrapper_base import WrapperBase


class CalculatorBase(SerializerComponent, metaclass=ABCMeta):
    """This class is a template and defines all properties that a calculator should have."""

    _calculators: list[CalculatorBase] = []  # class variable to store all calculators
    _material_link: dict[str, str]
    _layer_link: dict[str, str]
    _item_link: dict[str, str]
    _model_link: dict[str, str]

    def __init_subclass__(cls, is_abstract: bool = False, **kwargs) -> None:
        r"""Initialise all subclasses so that they can be created in the factory.

        Parameters
        ----------
        cls :
        is_abstract : bool, optional
            Is this a subclass which shouldn't be dded. By default, False.
        **kwargs :
            Key word arguments.
        """
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            cls._calculators.append(cls)

    def __init__(self):
        """Init function."""
        self._namespace = {}
        self._wrapper: WrapperBase

    def reset_storage(self) -> None:
        r"""Reset the storage area of the calculator."""
        self._wrapper.reset_storage()

    def create(self, model: Material | Layer | Multilayer | Model) -> list[ItemContainer]:
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
            if key not in self._wrapper.storage['material'].keys():
                self._wrapper.create_material(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._material_link,
                    self._wrapper.get_material_value,
                    self._wrapper.update_material,
                )
            )
        elif issubclass(t_, MaterialMixture):
            key = model.unique_name
            if key not in self._wrapper.storage['material'].keys():
                self._wrapper.create_material(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._material_link,
                    self._wrapper.get_material_value,
                    self._wrapper.update_material,
                )
            )
        elif issubclass(t_, Layer):
            key = model.unique_name
            if key not in self._wrapper.storage['layer'].keys():
                self._wrapper.create_layer(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._layer_link,
                    self._wrapper.get_layer_value,
                    self._wrapper.update_layer,
                )
            )
            self.assign_material_to_layer(model.material.unique_name, key)
        elif issubclass(t_, BaseAssembly):
            key = model.unique_name
            self._wrapper.create_item(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._item_link,
                    self._wrapper.get_item_value,
                    self._wrapper.update_item,
                )
            )
            for i in model.layers:
                self.add_layer_to_item(i.unique_name, model.unique_name)
        elif issubclass(t_, Model):
            key = model.unique_name
            self._wrapper.create_model(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._model_link,
                    self._wrapper.get_model_value,
                    self._wrapper.update_model,
                )
            )
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

    def reflectity_profile(self, x_array: np.ndarray, model_id: str) -> np.ndarray:
        """Determines the reflectivity profile for the given range and model.

        Parameters
        ----------
        x_array : np.ndarray
            Points to be calculated at.
        model_id : str
            The model id.
        """
        return self._wrapper.calculate(x_array, model_id)

    def reflectivity_profile_channel(self, x_array: np.ndarray, model_id: str, channel) -> np.ndarray:
        """Determine the reflectivity profile of one explicit spin channel.

        Unlike `polarization_channel` (global calculator state), the channel is an
        argument, so several channels can be evaluated against the same model —
        one per dataset in a simultaneous multi-channel fit.

        Parameters
        ----------
        x_array : np.ndarray
            Points to be calculated at.
        model_id : str
            The model id.
        channel : PolarizationChannel | str
            One of 'pp', 'pm', 'mp', 'mm' (or the corresponding enum member).

        Returns
        -------
        np.ndarray
            Reflectivity of the requested channel at q.
        """
        return self._wrapper.calculate_channel(x_array, model_id, channel)

    def polarized_reflectivity_profiles(self, x_array: np.ndarray, model_id: str) -> dict[str, np.ndarray]:
        """Determines the reflectivity profiles of all four spin channels for the given range and model.

        Requires `include_magnetism` to be enabled and a calculator that supports it (refl1d).

        Parameters
        ----------
        x_array : np.ndarray
            Points to be calculated at.
        model_id : str
            The model id.

        Returns
        -------
        dict[str, np.ndarray]
            Reflectivity per spin channel, keyed 'pp', 'pm', 'mp', 'mm' (in that order).
        """
        return self._wrapper.calculate_polarized(x_array, model_id)

    def sld_profile(self, model_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_id : str
            The model id.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            z and sld(z).
        """
        return self._wrapper.sld_profile(model_id)

    def magnetic_sld_profile(self, model_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the nuclear and magnetic scattering length density profiles.

        Requires `include_magnetism` to be enabled and a calculator that supports it (refl1d).

        Parameters
        ----------
        model_id : str
            The model id.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            z, sld(z), magnetic sld rhoM(z) and magnetic angle thetaM(z).
        """
        return self._wrapper.magnetic_sld_profile(model_id)

    def set_resolution_function(self, resolution_function: Callable[[np.array], np.array]) -> None:
        """Set resolution function."""
        return self._wrapper.set_resolution_function(resolution_function)

    @property
    def supports_magnetism(self) -> bool:
        """Whether this calculator backend can model magnetic samples."""
        return self._wrapper.supports_magnetism

    def remove_layer_magnetism(self, layer_id: str) -> None:
        """Remove the magnetic state of one layer; disables `include_magnetism`
        when no magnetic layer is left.

        Parameters
        ----------
        layer_id : str
            The layer id.
        """
        self._wrapper.remove_layer_magnetism(layer_id)

    @property
    def include_magnetism(self):
        """Include magnetism."""
        return self._wrapper.magnetism

    @include_magnetism.setter
    def include_magnetism(self, magnetism: bool):
        """Set the magnetism flag for the calculator.

        Parameters
        ----------
        magnetism : bool
            True if the calculator should include magnetism.
        """
        self._wrapper.magnetism = magnetism

    @property
    def polarization_channel(self):
        """The spin channel ('pp', 'pm', 'mp' or 'mm') used by `reflectity_profile` when magnetism is enabled.

        Note: this state belongs to the currently-active calculator instance; switching
        calculators via the factory constructs a fresh instance and resets it.
        """
        return self._wrapper.polarization_channel

    @polarization_channel.setter
    def polarization_channel(self, channel) -> None:
        """Set the spin channel for reflectivity calculations.

        Parameters
        ----------
        channel : PolarizationChannel | str
            One of 'pp', 'pm', 'mp', 'mm' (or the corresponding enum member).
        """
        self._wrapper.polarization_channel = channel
