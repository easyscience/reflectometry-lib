# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod

import numpy as np

from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import ResolutionFunction

from .polarization import PolarizationChannel


class WrapperBase:
    #: Whether this calculator backend can model magnetic samples.
    supports_magnetism = False

    def __init__(self):
        """Constructor."""
        self._magnetism = False
        self._polarization_channel = PolarizationChannel.PP
        self.storage = {
            'material': {},
            'layer': {},
            'item': {},
            'model': {},
        }
        # Default resolution, used for every model that has not registered its own.
        self._resolution_function = PercentageFwhm()
        # Per-model resolution functions, keyed by model name. One wrapper serves
        # every model of a project, so the resolution cannot be a single shared
        # value: two contrasts measured with different q-resolution would
        # otherwise both be smeared with whichever was set last.
        self._model_resolution_functions: dict[str, ResolutionFunction] = {}

    def reset_storage(self):
        """Reset the storage area to blank."""
        self.storage = {
            'material': {},
            'layer': {},
            'item': {},
            'model': {},
        }
        self._model_resolution_functions = {}

    def remove_material(self, name: str) -> None:
        """Forget a material, with everything the backend keeps under its name.

        Called when the easyscience object the entry was created for is gone
        and its name is being given to a new one; the new object's values are
        pushed after it is created afresh.

        Parameters
        ----------
        name : str
            The material name.
        """
        self.storage['material'].pop(name, None)

    def remove_layer(self, name: str) -> None:
        """Forget a layer, with everything the backend keeps under its name.

        Backends that keep per-layer state outside the layer object (e.g.
        magnetism) extend this.

        Parameters
        ----------
        name : str
            The layer name.
        """
        self.storage['layer'].pop(name, None)

    def remove_model(self, name: str) -> None:
        """Forget a model, with everything the backend keeps under its name.

        Parameters
        ----------
        name : str
            The model name.
        """
        self.storage['model'].pop(name, None)
        self._model_resolution_functions.pop(name, None)

    @abstractmethod
    def create_material(self, name: str):
        """Create a material using SLD.

        Parameters
        ----------
        name : str
            The name of the material.
        """
        ...

    @abstractmethod
    def create_layer(self, name: str):
        """Create a layer using Slab.

        Parameters
        ----------
        name : str
            The name of the layer.
        """
        ...

    @abstractmethod
    def create_item(self, name: str):
        """Create an item using Stack.

        Parameters
        ----------
        name : str
            The name of the item.
        """
        ...

    @abstractmethod
    def create_model(self, name: str):
        """Create a model for analysis.

        Parameters
        ----------
        name : str
            Name for the model.
        """
        ...

    @abstractmethod
    def update_model(self, name: str, **kwargs):
        """Update the non-structural parameters of the model.

        Parameters
        ----------
        name : str
            Name for the model.
        **kwargs :
        """
        ...

    @abstractmethod
    def get_model_value(self, name: str, key: str) -> float:
        """A function to get a given model value.

        Parameters
        ----------
        name : str
            Name for the model.
        key : str
            The given value keys.
        """
        ...

    @abstractmethod
    def assign_material_to_layer(self, material_name: str, layer_name: str):
        """Assign a material to a layer.

        Parameters
        ----------
        material_name : str
            The material name.
        layer_name : str
            The layer name.
        """
        ...

    @abstractmethod
    def add_layer_to_item(self, layer_name: str, item_name: str):
        """Create a layer from the material of the same name, in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        ...

    @abstractmethod
    def add_item(self, item_name: str, model_name: str):
        """Add an item to the model.

        Parameters
        ----------
        item_name : str
            Items to add to model.
        model_name : str
            Name for the model.
        """
        ...

    @abstractmethod
    def remove_layer_from_item(self, layer_name: str, item_name: str):
        """Remove a layer in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        ...

    @abstractmethod
    def remove_item(self, item_name: str, model_name: str):
        """Remove a given item.

        Parameters
        ----------
        item_name : str
            The item name.
        model_name : str
            Name of the model.
        """
        ...

    @abstractmethod
    def calculate(
        self,
        q_array: np.ndarray,
        model_name: str,
        resolution_function: ResolutionFunction | None = None,
    ) -> np.ndarray:
        """For a given q array calculate the corresponding reflectivity.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_name : str
            The model name.
        resolution_function : ResolutionFunction | None, optional
            Resolution to smear with; by default the one registered for the
            model (see :meth:`set_resolution_function`). A dataset measured
            with its own q-resolution passes it here.

        Returns
        -------
        np.ndarray
            Reflectivity calculated at q.
        """
        ...

    @abstractmethod
    def sld_profile(self, model_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_name : str
            Name for the model.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Z and sld(z).
        """
        ...

    def update_material(self, name: str, **kwargs):
        """Update a material.

        Parameters
        ----------
        name : str
            The name of the material.
        **kwargs :
            Key-value pairs of attributes to update.
        """
        material = self.storage['material'][name]
        for key in kwargs.keys():
            item = getattr(material, key)
            setattr(item, 'value', kwargs[key])

    def get_material_value(self, name: str, key: str) -> float:
        """A function to get a given material value.

        Parameters
        ----------
        name : str
            The material name.
        key : str
            The given value keys.

        Returns
        -------
        float
            The desired value.
        """
        material = self.storage['material'][name]
        item = getattr(material, key)
        return getattr(item, 'value')

    def update_layer(self, name: str, **kwargs):
        """Update a layer in a given item.

        Parameters
        ----------
        name : str
            The layer name.
        **kwargs :
        """
        layer = self.storage['layer'][name]
        for key in kwargs.keys():
            ii = getattr(layer, key)
            setattr(ii, 'value', kwargs[key])

    def get_layer_value(self, name: str, key: str) -> float:
        """A function to get a given layer value.

        Parameters
        ----------
        name : str
            The layer name.
        key : str
            The given value keys.
        """
        layer = self.storage['layer'][name]
        ii = getattr(layer, key)
        return getattr(ii, 'value')

    def update_item(self, name: str, **kwargs):
        """Update a layer.

        Parameters
        ----------
        **kwargs :
        name : str
            The item name.
        """
        item = self.storage['item'][name]
        for key in kwargs.keys():
            ii = getattr(item, key)
            setattr(ii, 'value', kwargs[key])

    def get_item_value(self, name: str, key: str) -> float:
        """A function to get a given item value.

        Parameters
        ----------
        name : str
            The item name.
        key : str
            The given value keys.

        Returns
        -------
        float
            The desired value.
        """
        item = self.storage['item'][name]
        item = getattr(item, key)
        return getattr(item, 'value')

    def set_resolution_function(self, resolution_function: ResolutionFunction, model_name: str | None = None) -> None:
        """Set the resolution function for one model, or the default for all others.

        Parameters
        ----------
        resolution_function : ResolutionFunction
            The resolution function.
        model_name : str | None, optional
            Name of the model the resolution belongs to. ``None`` sets the
            default used by every model without a resolution of its own.
        """
        if model_name is None:
            self._resolution_function = resolution_function
        else:
            self._model_resolution_functions[model_name] = resolution_function

    def resolution_function_for(self, model_name: str | None = None) -> ResolutionFunction:
        """The resolution function used when calculating *model_name*.

        Parameters
        ----------
        model_name : str | None, optional
            The model name; ``None`` (or an unknown name) gives the default.
        """
        if model_name is None:
            return self._resolution_function
        return self._model_resolution_functions.get(model_name, self._resolution_function)

    def _resolution_for(self, model_name: str, resolution_function: ResolutionFunction | None) -> ResolutionFunction:
        """An explicit resolution function, or else the one registered for the model."""
        if resolution_function is None:
            return self.resolution_function_for(model_name)
        return resolution_function

    @property
    def magnetism(self) -> bool:
        """Magnetism function."""
        return self._magnetism

    @magnetism.setter
    def magnetism(self, magnetism: bool) -> None:
        """Set the magnetism flag.

        Parameters
        ----------
        magnetism : bool
            The magnetism flag.
        """
        if magnetism and not self.supports_magnetism:
            raise NotImplementedError(f'Magnetism is not supported by {self.__class__.__name__}')
        self._magnetism = magnetism
        if magnetism:
            # Attach any magnetic values set (or restored) while magnetism was off.
            self._apply_magnetism_to_layers()
        else:
            # A non-pp channel is only meaningful on the polarized probe path.
            self._polarization_channel = PolarizationChannel.PP
            # Leave no magnetic residue behind: the unpolarized calculation path
            # must work on the layers that already exist.
            self._remove_magnetism_from_layers()

    def _remove_magnetism_from_layers(self) -> None:
        """Strip backend magnetism state from existing layers when magnetism is disabled.

        No-op by default; overridden by backends that attach magnetic objects to layers.
        """

    def _apply_magnetism_to_layers(self) -> None:
        """Attach stored magnetic values to existing layers when magnetism is enabled.

        No-op by default; overridden by backends that attach magnetic objects to layers.
        """

    def remove_layer_magnetism(self, name: str) -> None:
        """Remove the magnetic state of one layer; disable magnetism when none is left.

        No-op by default; overridden by backends that support magnetism.

        Parameters
        ----------
        name : str
            The layer name.
        """

    @property
    def polarization_channel(self) -> PolarizationChannel:
        """The spin channel returned by `calculate` when magnetism is enabled."""
        return self._polarization_channel

    @polarization_channel.setter
    def polarization_channel(self, channel: PolarizationChannel | str) -> None:
        """Set the spin channel returned by `calculate`.

        Parameters
        ----------
        channel : PolarizationChannel | str
            One of 'pp', 'pm', 'mp', 'mm' (or the corresponding enum member).
        """
        channel = PolarizationChannel(channel)
        if channel is not PolarizationChannel.PP and not self._magnetism:
            raise ValueError(f"Selecting the '{channel.value}' channel requires magnetism to be enabled.")
        self._polarization_channel = channel

    def calculate_channel(
        self,
        q_array: np.ndarray,
        model_name: str,
        channel: PolarizationChannel | str,
        resolution_function: ResolutionFunction | None = None,
    ) -> np.ndarray:
        """For a given q array calculate the reflectivity of one explicit spin channel.

        Unlike the `polarization_channel` property (global calculator state used by
        `calculate`), the channel is passed explicitly, so several channels can be
        evaluated against the same model, e.g. one per dataset in a simultaneous
        multi-channel fit.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_name : str
            The model name.
        channel : PolarizationChannel | str
            One of 'pp', 'pm', 'mp', 'mm' (or the corresponding enum member).
        resolution_function : ResolutionFunction | None, optional
            Resolution to smear with; by default the one registered for the
            model. Each spin channel is measured with its own q-resolution, so
            a channel dataset passes its own here.

        Returns
        -------
        np.ndarray
            Reflectivity of the requested channel at q.
        """
        channel = PolarizationChannel(channel)
        if not self._magnetism:
            if channel is PolarizationChannel.PP:
                # No explicit `.copy()` needed here: `calculate()` always returns
                # a fresh array (a cached, shared array only exists on the
                # magnetism-enabled `calculate_polarized` path below).
                return self.calculate(q_array, model_name, resolution_function=resolution_function)
            raise ValueError(f"Calculating the '{channel.value}' channel requires magnetism to be enabled.")
        return self.calculate_polarized(q_array, model_name, resolution_function=resolution_function)[channel.value]

    def calculate_polarized(
        self,
        q_array: np.ndarray,
        model_name: str,
        resolution_function: ResolutionFunction | None = None,
    ) -> dict[str, np.ndarray]:
        """For a given q array calculate the reflectivity of all four spin channels.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_name : str
            The model name.
        resolution_function : ResolutionFunction | None, optional
            Resolution to smear with; by default the one registered for the model.

        Returns
        -------
        dict[str, np.ndarray]
            Reflectivity per spin channel, keyed 'pp', 'pm', 'mp', 'mm' (in that order).
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support polarized reflectivity.')

    def magnetic_sld_profile(self, model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the nuclear and magnetic scattering length density profiles.

        Parameters
        ----------
        model_name : str
            The model name.

        Returns
        -------
            z, sld(z), magnetic sld rhoM(z) and magnetic angle thetaM(z).
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support magnetic sld profiles.')
