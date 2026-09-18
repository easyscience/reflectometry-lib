# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from typing import Tuple

import numpy as np
from refnx import reflect

from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM

from ..wrapper_base import WrapperBase


class RefnxWrapper(WrapperBase):
    def create_material(self, name: str):
        """Create a material using SLD.

        Parameters
        ----------
        name : str
            The name of the material.
        """
        self.storage['material'][name] = reflect.SLD(0, name=name)

    def create_layer(self, name: str):
        """Create a layer using Slab.

        Parameters
        ----------
        name : str
            The name of the layer.
        """
        self.storage['layer'][name] = reflect.Slab(0, 0, 0, name=name)

    def create_item(self, name: str):
        """Create an item using Stack.

        Parameters
        ----------
        name : str
            The name of the item.
        """
        self.storage['item'][name] = reflect.Stack(name=name)

    def create_model(self, name: str):
        """Create a model for analysis.

        Parameters
        ----------
        name : str
            Name for the model.
        """
        self.storage['model'][name] = reflect.ReflectModel(reflect.Structure())

    def update_model(self, name: str, **kwargs):
        """Update the non-structural parameters of the model.

        Parameters
        ----------
        **kwargs :
        name : str
            Name for the model.
        """
        model = self.storage['model'][name]
        for key in kwargs.keys():
            item = getattr(model, key)
            setattr(item, 'value', kwargs[key])

    def get_model_value(self, name: str, key: str) -> float:
        """A function to get a given model value.

        Parameters
        ----------
        name : str
            Name for the model.
        key : str
            The given value keys.

        Returns
        -------
        float
            The desired value.
        """
        model = self.storage['model'][name]
        item = getattr(model, key)
        return getattr(item, 'value')

    def assign_material_to_layer(self, material_name: str, layer_name: str):
        """Assign a material to a layer.

        Parameters
        ----------
        material_name : str
            The material name.
        layer_name : str
            The layer name.
        """
        self.storage['layer'][layer_name].sld = self.storage['material'][material_name]

    def add_layer_to_item(self, layer_name: str, item_name: str):
        """Create a layer from the material of the same name, in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        item = self.storage['item'][item_name]
        item.append(self.storage['layer'][layer_name])

    def add_item(self, item_name: str, model_name: str):
        """Add an item to the model.

        Parameters
        ----------
        item_name : str
            Items to add to model.
        model_name : str
            Name for the model.
        """
        self.storage['model'][model_name].structure.components.append(self.storage['item'][item_name])

    def remove_layer_from_item(self, layer_name: str, item_name: str):
        """Remove a layer in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        layer_idx = self.storage['item'][item_name].components.index(self.storage['layer'][layer_name])
        del self.storage['item'][item_name].components[layer_idx]

    def remove_item(self, item_name: str, model_name: str):
        """Remove a given item.

        Parameters
        ----------
        item_name : str
            The item name.
        model_name : str
            Name of the model.
        """
        item_idx = self.storage['model'][model_name].structure.components.index(self.storage['item'][item_name])
        del self.storage['model'][model_name].structure.components[item_idx]
        del self.storage['item'][item_name]

    def calculate(self, q_array: np.ndarray, model_name: str) -> np.ndarray:
        """For a given q array calculate the corresponding reflectivity.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_name : str
            The model name.

        Returns
        -------
        np.ndarray
            Reflectivity calculated at q.
        """
        structure = _remove_unecessary_stacks(self.storage['model'][model_name].structure)
        model = reflect.ReflectModel(
            structure,
            scale=self.storage['model'][model_name].scale.value,
            bkg=self.storage['model'][model_name].bkg.value,
            dq_type='pointwise',
        )

        dq_vector = self._resolution_function.smearing(q_array)
        if isinstance(self._resolution_function, PercentageFwhm):
            # refnx interprets a scalar x_err as a constant dq/q (FWHM percentage),
            # so pass the percentage directly rather than a per-point vector.
            dq_vector = self._resolution_function.constant
        else:
            # smearing() returns sigma; refnx expects the FWHM at each point.
            dq_vector = dq_vector * SIGMA_TO_FWHM

        return model(x=q_array, x_err=dq_vector)

    def sld_profile(self, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_name : str
            Name for the model.

        Returns
        -------

            Z and sld(z).
        """
        return _remove_unecessary_stacks(self.storage['model'][model_name].structure).sld_profile()


def _remove_unecessary_stacks(current_structure: reflect.Structure) -> reflect.Structure:
    """Removed unnecessary reflect.Stack objects from the structure.

    Parameters
    ----------
    current_structure : reflect.Structure
        The current structure.

    Returns
    -------
    reflect.structure
        The structre without the unnecessary Stacks.
    """
    structure = []
    for i in current_structure.components:
        if i.repeats.value == 1:
            for j in i.components:
                structure.append(j)
        else:
            structure.append(i)
    return reflect.Structure(structure)
