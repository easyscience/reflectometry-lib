# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

__author__ = 'github.com/wardsimon'
from typing import Callable

from easyscience.fitting.calculators.interface_factory import InterfaceFactoryTemplate

from easyreflectometry.calculators import CalculatorBase

from .polarization import PolarizationChannel


class CalculatorFactory(InterfaceFactoryTemplate):
    def __init__(self):
        """Init function."""
        super().__init__(interface_list=CalculatorBase._calculators)

    def reset_storage(self) -> None:
        """Reset storage."""
        return self().reset_storage()

    def sld_profile(self, model_id: str) -> tuple:
        """Sld profile."""
        return self().sld_profile(model_id)

    def polarized_reflectivity_profiles(self, x_array, model_id: str) -> dict:
        """Reflectivity profiles of all four spin channels ('pp', 'pm', 'mp', 'mm')."""
        return self().polarized_reflectivity_profiles(x_array, model_id)

    def reflectivity_profile_channel(self, x_array, model_id: str, channel: PolarizationChannel | str):
        """Reflectivity profile of one explicit spin channel ('pp', 'pm', 'mp' or 'mm')."""
        return self().reflectivity_profile_channel(x_array, model_id, channel)

    def magnetic_sld_profile(self, model_id: str) -> tuple:
        """Nuclear and magnetic sld profiles: z, sld(z), rhoM(z) and thetaM(z)."""
        return self().magnetic_sld_profile(model_id)

    @property
    def fit_func(self) -> Callable:
        """Fit func."""
        """
        Pass through to the underlying interfaces fitting function.

        :param x_array: points to be calculated at
        :type x_array: np.ndarray
        :param args: positional arguments for the fitting function
        :type args: Any
        :param kwargs: key/value pair arguments for the fitting function.
        :type kwargs: Any
        :return: points calculated at positional values `x`
        :rtype: np.ndarray
        #"""

        def __fit_func(*args, **kwargs):
            """Fit func."""
            return self().reflectity_profile(*args, **kwargs)

        return __fit_func

    def fit_func_for_channel(self, channel: PolarizationChannel | str) -> Callable:
        """A fit function evaluating one explicit spin channel.

        Used for simultaneous multi-channel fitting: each channel dataset gets its
        own fit function while all of them share the same model (and hence the
        same parameters).

        Parameters
        ----------
        channel : PolarizationChannel | str
            One of 'pp', 'pm', 'mp', 'mm' (or the corresponding enum member).

        Returns
        -------
        Callable
            Function of (x_array, model_id) returning the channel reflectivity.
        """
        channel = PolarizationChannel(channel)

        def __fit_func(x_array, model_id):
            """Fit func for one spin channel."""
            return self().reflectivity_profile_channel(x_array, model_id, channel)

        return __fit_func
