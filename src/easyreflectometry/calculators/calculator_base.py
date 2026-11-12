# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABCMeta

import numpy as np

from easyreflectometry.model import Model
from easyreflectometry.model import ResolutionFunction


class CalculatorBase(metaclass=ABCMeta):
    """A calculator: a thin layer over its wrapper, which reads the models it serves.

    Every non-abstract subclass registers itself in ``_calculators`` and so
    becomes available in :class:`CalculatorFactory`.
    """

    _calculators: list[type[CalculatorBase]] = []  # class variable to store all calculators

    def __init_subclass__(cls, is_abstract: bool = False, **kwargs) -> None:
        r"""Register every subclass, unless *is_abstract*, so that the factory can create it."""
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            cls._calculators.append(cls)

    def __init__(self):
        """Init function."""
        self._wrapper = None  # set by each subclass

    # ----- membership -----

    def bind(self, obj) -> None:
        """Called for every object of a sample tree attached to this calculator; only models are kept."""
        if isinstance(obj, Model):
            self.register_model(obj)

    def register_model(self, model: Model) -> None:
        """Serve *model*."""
        self._wrapper.register_model(model)

    def discard_model(self, model_id: str) -> None:
        """Stop serving the model with *model_id*."""
        self._wrapper.discard_model(model_id)

    def reset_storage(self) -> None:
        """Stop serving every model."""
        self._wrapper.reset_storage()

    @property
    def models(self) -> dict[str, Model]:
        """The models served, keyed by ``unique_name``."""
        return self._wrapper.models

    def reconcile_magnetism(self) -> None:
        """Switch `include_magnetism` off when the served models have just stopped being magnetic."""
        self._wrapper.reconcile_magnetism()

    # ----- evaluation -----

    def _before_evaluation(self) -> None:
        """Run at the start of every evaluation and profile.

        Covers structural removals (a magnetic layer, an assembly) that nothing
        reports to the calculator.
        """
        self.reconcile_magnetism()

    def reflectity_profile(
        self,
        x_array: np.ndarray,
        model_id: str,
        resolution_function: ResolutionFunction | None = None,
    ) -> np.ndarray:
        """Determines the reflectivity profile for the given range and model.

        Parameters
        ----------
        x_array : np.ndarray
            Points to be calculated at.
        model_id : str
            The model id.
        resolution_function : ResolutionFunction | None, optional
            Resolution to smear with; by default the one registered for the
            model through :meth:`set_resolution_function`.
        """
        self._before_evaluation()
        return self._wrapper.calculate(x_array, model_id, resolution_function=resolution_function)

    def reflectivity_profile_channel(
        self,
        x_array: np.ndarray,
        model_id: str,
        channel,
        resolution_function: ResolutionFunction | None = None,
    ) -> np.ndarray:
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
        resolution_function : ResolutionFunction | None, optional
            Resolution to smear with; by default the one registered for the model.

        Returns
        -------
        np.ndarray
            Reflectivity of the requested channel at q.
        """
        self._before_evaluation()
        return self._wrapper.calculate_channel(x_array, model_id, channel, resolution_function=resolution_function)

    def polarized_reflectivity_profiles(
        self,
        x_array: np.ndarray,
        model_id: str,
        resolution_function: ResolutionFunction | None = None,
    ) -> dict[str, np.ndarray]:
        """Determines the reflectivity profiles of all four spin channels for the given range and model.

        Requires `include_magnetism` to be enabled and a calculator that supports it (refl1d).

        Parameters
        ----------
        x_array : np.ndarray
            Points to be calculated at.
        model_id : str
            The model id.
        resolution_function : ResolutionFunction | None, optional
            Resolution to smear with; by default the one registered for the model.

        Returns
        -------
        dict[str, np.ndarray]
            Reflectivity per spin channel, keyed 'pp', 'pm', 'mp', 'mm' (in that order).
        """
        self._before_evaluation()
        return self._wrapper.calculate_polarized(x_array, model_id, resolution_function=resolution_function)

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
        self._before_evaluation()
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
        self._before_evaluation()
        return self._wrapper.magnetic_sld_profile(model_id)

    def set_resolution_function(self, resolution_function: ResolutionFunction, model_id: str | None = None) -> None:
        """Set the resolution function of one model, or the default for all others.

        Parameters
        ----------
        resolution_function : ResolutionFunction
            The resolution function.
        model_id : str | None, optional
            The model id. ``None`` sets the default used by every model that
            has not registered a resolution of its own.
        """
        return self._wrapper.set_resolution_function(resolution_function, model_id)

    def resolution_function_for(self, model_id: str | None = None) -> ResolutionFunction:
        """The resolution function used when calculating the model with *model_id*."""
        return self._wrapper.resolution_function_for(model_id)

    @property
    def supports_magnetism(self) -> bool:
        """Whether this calculator backend can model magnetic samples."""
        return self._wrapper.supports_magnetism

    @property
    def include_magnetism(self):
        """Include magnetism."""
        return self._wrapper.magnetism

    @include_magnetism.setter
    def include_magnetism(self, magnetism: bool):
        """Set the magnetism flag for the calculator.

        It stays as set until the served models stop being magnetic (see
        :meth:`reconcile_magnetism`).

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
