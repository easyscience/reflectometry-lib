# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

__author__ = 'github.com/wardsimon'

from typing import Optional
from typing import Union

import numpy as np
from easyscience.io import SerializerComponent

from easyreflectometry.model import Model


class DataSet1D(SerializerComponent):
    def __init__(
        self,
        name: str = 'Series',
        x: Optional[Union[np.ndarray, list]] = None,
        y: Optional[Union[np.ndarray, list]] = None,
        ye: Optional[Union[np.ndarray, list]] = None,
        xe: Optional[Union[np.ndarray, list]] = None,
        model: Optional['Model'] = None,  # delay type checking until runtime (quotes)
        x_label: str = 'x',
        y_label: str = 'y',
        auto_background: bool = True,
    ):
        """Init function."""
        self._model = model
        if y is not None and model is not None and auto_background:
            self._model.background = max(np.min(y), 1e-10)

        if x is None:
            x = np.array([])
        if y is None:
            y = np.array([])
        if ye is None:
            ye = np.zeros_like(x)
        if xe is None:
            xe = np.zeros_like(x)

        if len(x) != len(y):
            raise ValueError('x and y must be the same length')

        self.name = name
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if not isinstance(ye, np.ndarray):
            ye = np.array(ye)
        if not isinstance(xe, np.ndarray):
            xe = np.array(xe)

        self.x = x
        self.y = y
        self.ye = ye
        self.xe = xe

        self.x_label = x_label
        self.y_label = y_label

        self._color = None

    @property
    def model(self) -> 'Model':  # delay type checking until runtime (quotes)
        """Model function."""
        return self._model

    @model.setter
    def model(self, new_model: 'Model') -> None:
        """Model function."""
        self._model = new_model

    @property
    def is_experiment(self) -> bool:
        """Is experiment."""
        return self._model is not None

    @property
    def is_simulation(self) -> bool:
        """Is simulation."""
        return self._model is None

    def data_points(self) -> tuple[float, float, float, float]:
        """Data points."""
        return zip(self.x, self.y, self.ye, self.xe)

    def __repr__(self) -> str:
        """Repr function."""
        return "1D DataStore of '{:s}' Vs '{:s}' with {} data points".format(self.x_label, self.y_label, len(self.x))
