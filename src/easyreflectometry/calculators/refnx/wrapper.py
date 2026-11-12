# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from refnx import reflect

from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import ResolutionFunction
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM

from ..stateless_wrapper_base import StatelessWrapperBase
from ..stateless_wrapper_base import repeats_of
from ..stateless_wrapper_base import value_of

if TYPE_CHECKING:
    from easyreflectometry.model import Model
    from easyreflectometry.sample import Layer


class RefnxWrapper(StatelessWrapperBase):
    """Builds the refnx structure from the model on every evaluation.

    Assemblies in sample order, layers in assembly order. An assembly that is
    not repeated (no ``repetitions``, or exactly one) is flattened into its
    slabs; a repeated one becomes a ``Stack``.
    """

    def calculate(
        self, q_array: np.ndarray, model_id: str | None = None, resolution_function: ResolutionFunction | None = None
    ) -> np.ndarray:
        """Reflectivity of the model at *q_array*; see :meth:`StatelessWrapperBase.calculate`."""
        model = self._get_model(model_id)
        refl_model = reflect.ReflectModel(
            self._build_structure(model),
            scale=model.scale.value,
            bkg=model.background.value,
            dq_type='pointwise',
        )
        resolution_function = self._resolution_for(model.unique_name, resolution_function)
        return refl_model(x=q_array, x_err=_x_err(resolution_function, q_array))

    def sld_profile(self, model_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """z and sld(z) of the model."""
        return self._build_structure(self._get_model(model_id)).sld_profile()

    @staticmethod
    def _build_structure(model: Model) -> reflect.Structure:
        components = []
        for assembly in model.sample:
            slabs = [_slab(layer) for layer in assembly.layers]
            repeats = repeats_of(assembly)
            if repeats == 1:
                components.extend(slabs)
            else:
                components.append(reflect.Stack(slabs, name=assembly.unique_name, repeats=repeats))
        return reflect.Structure(components)


def _slab(layer: Layer) -> reflect.Slab:
    material = layer.material
    sld = reflect.SLD(complex(value_of(material, 'sld'), value_of(material, 'isld')), name=material.unique_name)
    return reflect.Slab(value_of(layer, 'thickness'), sld, value_of(layer, 'roughness'), name=layer.unique_name)


def _x_err(resolution_function: ResolutionFunction, q_array: np.ndarray) -> np.ndarray | float:
    """The resolution in refnx's convention.

    refnx reads a scalar ``x_err`` as a constant dq/q FWHM percentage, and a
    per-point one as the FWHM at each q; ``smearing`` returns sigma.
    """
    if isinstance(resolution_function, PercentageFwhm):
        return resolution_function.constant
    return resolution_function.smearing(q_array) * SIGMA_TO_FWHM
