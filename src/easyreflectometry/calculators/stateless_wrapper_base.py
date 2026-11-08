# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import weakref
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import ResolutionFunction

from .polarization import PolarizationChannel

if TYPE_CHECKING:
    from easyreflectometry.model import Model
    from easyreflectometry.sample import BaseAssembly


def value_of(obj: object, name: str) -> float:
    """The current value of ``obj.<name>``.

    Most attributes are parameters; ``MaterialMixture`` exposes its derived
    ``sld``/``isld`` as plain floats.
    """
    value = getattr(obj, name)
    return float(getattr(value, 'value', value))


def repeats_of(assembly: BaseAssembly) -> float:
    """How often *assembly* is repeated; 1 for an assembly without ``repetitions``."""
    return 1.0 if getattr(assembly, 'repetitions', None) is None else value_of(assembly, 'repetitions')


@dataclass
class _ModelEntry:
    """Everything the calculator keeps for one model; dropping the entry drops all of it."""

    model: weakref.ReferenceType
    resolution_function: ResolutionFunction | None = None
    #: The model state that the results in `cache` were calculated at.
    cache_state: tuple | None = None
    #: Calculated results at `cache_state`, keyed by grid.
    cache: dict = field(default_factory=dict)


class StatelessWrapperBase:
    """Backend-agnostic part of a wrapper that reads the model at evaluation time.

    No backend object holds a parameter value: the backend structure is built
    from the model inside every calculation and dropped when it returns, so
    value and structural edits need no mirroring. The wrapper owns only its
    configuration (default resolution, magnetism flag, polarization channel)
    and one :class:`_ModelEntry` per registered model.

    Membership is explicit (:meth:`register_model`, :meth:`discard_model`,
    :meth:`reset_storage`). Entries hold their model weakly and are removed
    when it dies, but only while they still refer to that same model: core
    hands the ``unique_name`` of a dead object out again, and a new model
    registered under it keeps its entry.

    The magnetism flag is explicit state; :meth:`reconcile_magnetism` only
    switches it off when the served models stop being magnetic.
    """

    #: Whether this calculator backend can model magnetic samples.
    supports_magnetism = False

    def __init__(self):
        """Constructor."""
        self._models: dict[str, _ModelEntry] = {}
        self._magnetism = False
        # Whether a served model was magnetic when the flag was last set or reconciled.
        self._magnetic_seen = False
        self._polarization_channel = PolarizationChannel.PP
        # Used for every model that has not registered a resolution of its own.
        self._resolution_function = PercentageFwhm()

    # ----- membership -----

    @property
    def models(self) -> dict[str, Model]:
        """The registered models, keyed by ``unique_name``."""
        return {name: entry.model() for name, entry in self._models.items() if entry.model() is not None}

    def register_model(self, model: Model) -> None:
        """Serve *model*; registering it again keeps its entry."""
        name = model.unique_name
        entry = self._models.get(name)
        if entry is not None and entry.model() is model:
            return
        wrapper = weakref.ref(self)

        def forget(dead: weakref.ReferenceType) -> None:
            owner = wrapper()
            if owner is not None and name in owner._models and owner._models[name].model is dead:
                del owner._models[name]

        self._models[name] = _ModelEntry(weakref.ref(model, forget))
        self.reconcile_magnetism()

    def discard_model(self, model_id: str) -> None:
        """Stop serving the model with *model_id* (unknown ids are ignored)."""
        self._models.pop(model_id, None)
        self.reconcile_magnetism()

    def reset_storage(self) -> None:
        """Stop serving every model."""
        self._models = {}
        self.reconcile_magnetism()

    def _get_model(self, model_id: str | None = None) -> Model:
        """The registered model with *model_id*; ``None`` is allowed when exactly one is registered.

        Raises
        ------
        ValueError
            If no model is registered, if several are and none was selected,
            or if *model_id* is not registered.
        """
        models = self.models
        if not models:
            raise ValueError(
                'No model is attached to the calculator. Attach one with '
                '`model.interface = calculator_factory` before calculating.'
            )
        if model_id is None:
            if len(models) > 1:
                raise ValueError(
                    f'Several models are attached to the calculator ({sorted(models)}); '
                    'pass `model_id` to select the one to evaluate.'
                )
            return next(iter(models.values()))
        if model_id not in models:
            raise ValueError(
                f'No model with unique name {model_id!r} is attached to the calculator. Attached models: {sorted(models)}.'
            )
        return models[model_id]

    # ----- resolution -----

    def set_resolution_function(self, resolution_function: ResolutionFunction, model_id: str | None = None) -> None:
        """Set the resolution function of one registered model, or the default for all others.

        Parameters
        ----------
        resolution_function : ResolutionFunction
            The resolution function.
        model_id : str | None, optional
            The model it belongs to. ``None`` sets the default used by every
            model without a resolution of its own. A model that is not
            registered is not served, so nothing is kept for it.
        """
        if model_id is None:
            self._resolution_function = resolution_function
        elif model_id in self._models:
            self._models[model_id].resolution_function = resolution_function

    def resolution_function_for(self, model_id: str | None = None) -> ResolutionFunction:
        """The resolution function used when calculating *model_id*; the default for ``None`` or an unknown id."""
        entry = self._models.get(model_id)
        if entry is None or entry.resolution_function is None:
            return self._resolution_function
        return entry.resolution_function

    def _resolution_for(self, model_id: str, resolution_function: ResolutionFunction | None) -> ResolutionFunction:
        """An explicit (per-dataset) resolution function, or else the one registered for the model."""
        if resolution_function is None:
            return self.resolution_function_for(model_id)
        return resolution_function

    # ----- magnetism and polarization -----

    @property
    def magnetism(self) -> bool:
        """Whether magnetism is included in the calculation."""
        return self._magnetism

    @magnetism.setter
    def magnetism(self, magnetism: bool) -> None:
        if magnetism and not self.supports_magnetism:
            raise NotImplementedError(f'Magnetism is not supported by {self.__class__.__name__}')
        self._magnetism = magnetism
        # An explicit choice is never undone by the next reconciliation; only a
        # later removal of magnetism from the models can switch it off.
        self._magnetic_seen = self._has_magnetic_model()
        if not magnetism:
            # A non-pp channel is only meaningful on the polarized probe path.
            self._polarization_channel = PolarizationChannel.PP

    def reconcile_magnetism(self) -> None:
        """Switch magnetism off when the served models have just stopped being magnetic.

        Only that transition counts: a flag switched on before any layer is
        magnetic, or switched off while some are, stays as it was set. Never
        switches magnetism on.
        """
        if not self._magnetism:
            return
        magnetic = self._has_magnetic_model()
        if self._magnetic_seen and not magnetic:
            self.magnetism = False
        self._magnetic_seen = magnetic

    def _has_magnetic_model(self) -> bool:
        return any(model.has_magnetism for model in self.models.values())

    @property
    def polarization_channel(self) -> PolarizationChannel:
        """The spin channel returned by `calculate` when magnetism is enabled."""
        return self._polarization_channel

    @polarization_channel.setter
    def polarization_channel(self, channel: PolarizationChannel | str) -> None:
        channel = PolarizationChannel(channel)
        # First, so that a channel picked after a structural edit is not reset by the next evaluation.
        self.reconcile_magnetism()
        if channel is not PolarizationChannel.PP and not self._magnetism:
            raise ValueError(f"Selecting the '{channel.value}' channel requires magnetism to be enabled.")
        self._polarization_channel = channel

    # ----- calculation -----

    @abstractmethod
    def calculate(
        self, q_array: np.ndarray, model_id: str | None = None, resolution_function: ResolutionFunction | None = None
    ) -> np.ndarray:
        """Reflectivity of the model at *q_array*.

        Parameters
        ----------
        q_array : np.ndarray
            Points to calculate at.
        model_id : str | None, optional
            The ``unique_name`` of the model; may be omitted when exactly one
            model is registered.
        resolution_function : ResolutionFunction | None, optional
            Resolution to smear with (a dataset's own); by default the one
            registered for the model.
        """

    @abstractmethod
    def sld_profile(self, model_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """z and sld(z) of the model."""

    def calculate_channel(
        self,
        q_array: np.ndarray,
        model_id: str | None,
        channel: PolarizationChannel | str,
        resolution_function: ResolutionFunction | None = None,
    ) -> np.ndarray:
        """Reflectivity of one explicit spin channel.

        Without magnetism only 'pp' exists and it is the ordinary reflectivity.
        """
        channel = PolarizationChannel(channel)
        if not self._magnetism:
            if channel is PolarizationChannel.PP:
                return self.calculate(q_array, model_id, resolution_function=resolution_function)
            raise ValueError(f"Calculating the '{channel.value}' channel requires magnetism to be enabled.")
        return self.calculate_polarized(q_array, model_id, resolution_function=resolution_function)[channel.value]

    def calculate_polarized(
        self, q_array: np.ndarray, model_id: str | None, resolution_function: ResolutionFunction | None = None
    ) -> dict[str, np.ndarray]:
        """Reflectivity of all four spin channels, keyed 'pp', 'pm', 'mp', 'mm'."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support polarized reflectivity.')

    def magnetic_sld_profile(self, model_id: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """z, sld(z), magnetic sld rhoM(z) and magnetic angle thetaM(z)."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support magnetic sld profiles.')
