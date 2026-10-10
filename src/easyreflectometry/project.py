# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import json
import logging
import os
import stat
import tempfile
import warnings
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.fitting import AvailableMinimizers
from easyscience.variable import DescriptorNumber as DescriptorNumberType
from easyscience.variable import Parameter
from easyscience.variable.parameter_dependency_resolver import resolve_all_parameter_dependencies
from scipp import DataGroup

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.calculators.calculator_base import CalculatorBase
from easyreflectometry.constraints import SUM_PARTNER_MAX_BACKUP
from easyreflectometry.constraints import USER_CONSTRAINT_FLAG
from easyreflectometry.constraints import constrain
from easyreflectometry.constraints import constrain_equal
from easyreflectometry.constraints import unconstrain
from easyreflectometry.contrasts import LinkPlan
from easyreflectometry.contrasts import LinkRecord
from easyreflectometry.contrasts import ReplaceFormula
from easyreflectometry.contrasts import ReplaceMaterial
from easyreflectometry.contrasts import derive_contrast
from easyreflectometry.contrasts import follows
from easyreflectometry.contrasts import kept_children
from easyreflectometry.contrasts import plan_link
from easyreflectometry.contrasts import restore
from easyreflectometry.contrasts import snapshot
from easyreflectometry.contrasts import tie_corresponding
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.data import detect_polarization_channel
from easyreflectometry.data import detect_polarization_channels_per_dataset
from easyreflectometry.data import load_as_dataset
from easyreflectometry.data import merge_datasets
from easyreflectometry.data import resolution_from_dataset
from easyreflectometry.data.measurement import dataset_from_datagroup
from easyreflectometry.data.measurement import extract_orso_title
from easyreflectometry.data.measurement import load as load_measurement_file
from easyreflectometry.data.measurement import load_data_from_orso_file
from easyreflectometry.fit_settings import DEFAULT_MINIMIZER  # noqa: F401 (re-exported)
from easyreflectometry.fit_settings import FitSettings
from easyreflectometry.fitting import FitRun
from easyreflectometry.fitting import FitScopeError
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.fitting import PreparedFit
from easyreflectometry.inequality_constraints import InequalityEvaluation
from easyreflectometry.inequality_constraints import InequalitySpec
from easyreflectometry.inequality_constraints import build_constraints_factory
from easyreflectometry.inequality_constraints import check_units
from easyreflectometry.inequality_constraints import evaluate_spec
from easyreflectometry.limits import apply_default_limits
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import ResolutionFunction
from easyreflectometry.orso_utils import is_orso_file
from easyreflectometry.orso_utils import save_orso_experiment
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialCollection
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import paths
from easyreflectometry.sample import references
from easyreflectometry.sample import volume_fraction_profile
from easyreflectometry.sample.collections.base_collection import BaseCollection

logger = logging.getLogger(__name__)

Q_MIN = 0.001
Q_MAX = 0.3
Q_RESOLUTION = 500

# Guide-field angle A in degrees, used to project the moment onto the neutron
# quantisation axis for the spin-up/spin-down potentials and nothing else.
# refl1d's default (`Aguide = 270`), which is what every model the library can
# currently build uses — `Aguide` is not exposed as a parameter yet. Keep this
# the single source of the value: exposing it later is a change here only.
GUIDE_FIELD_ANGLE = 270.0

# Points whose spin-asymmetry denominator (R⁺⁺ + R⁻⁻) is smaller than this
# multiple of its own uncertainty are dropped: there SA is noise divided by
# noise and would swamp the axis with ±values of no physical meaning.
SPIN_ASYMMETRY_SIGNIFICANCE = 3.0

# Points whose spin-asymmetry denominator (R⁺⁺ + R⁻⁻) is smaller than this
# fraction of |R⁺⁺| + |R⁻⁻| are dropped too: there the sum is what is left after
# cancellation between the two channels, so SA is a ratio of rounding noise.
# Unlike the significance rule above this needs no uncertainties, which is what
# keeps a two-column file from putting ±10³ values on the axis.
SPIN_ASYMMETRY_CANCELLATION_FRACTION = 1e-3

# A depth whose magnetic SLD is below this fraction of the largest one in the
# profile carries no moment worth speaking of, and its moment *angle* is
# meaningless — the direction of a (nearly) zero-length vector. Used to restrict
# the reported theta_m profile; a relative floor because it has to work both for
# a weak 0.1 and a strong 5 (1e-6 A^-2) moment, and because the interface
# roughness leaves a small erf tail everywhere.
MAGNETIC_MOMENT_FLOOR_FRACTION = 0.01


@dataclass(frozen=True)
class ModelRemoval:
    """What removing a model affects (see :meth:`Project.plan_model_removal`)."""

    #: Keys of the experiments bound to the model.
    experiments: List[int]
    #: Parameters of other models that follow a parameter only this model owns.
    dependents: List[Parameter]
    #: Inequality constraints on parameters only this model owns.
    inequality_constraints: List[InequalitySpec]


def _weak_constraints_provider(project: 'Project'):
    project_ref = weakref.ref(project)

    def provider():
        target = project_ref()
        return None if target is None else target.build_constraints_factory()

    return provider


class Project:
    def __init__(self):
        """Init function."""
        self._info = self._default_info()
        self._path_project_parent = Path(os.path.expanduser('~'))
        self._models = ModelCollection(populate_if_none=False, unique_name='project_models')
        self._materials = MaterialCollection(populate_if_none=False, unique_name='project_materials')
        self._calculator = CalculatorFactory()
        self._experiments: Dict[DataGroup] = {}
        self._fitter: MultiFitter = None
        self._fit_settings = FitSettings()
        #: Warnings from the last :meth:`from_dict`, for the user; cleared by each load.
        self.load_report: list[str] = []
        self._colors: list[str] = None
        self._report = None
        self._q_min: float = None
        self._q_max: float = None
        self._q_resolution: int = None
        self._current_material_index = 0
        self._current_model_index = 0
        self._current_assembly_index = 0
        self._current_layer_index = 0
        self._fitter_model_index = None
        self._current_experiment_index = 0
        self._inequality_constraints: List[InequalitySpec] = []
        self._last_fit = None
        #: (derived model, the model it was derived from), see :meth:`add_contrast`.
        self._contrasts: List[tuple] = []
        self._links: List[LinkRecord] = []

        # Project flags
        self._created = False
        self._with_experiments = False

    def reset(self):
        """Reset function."""
        del self._models
        del self._materials
        global_object.map._clear()

        self.__init__()

    @property
    def parameters(self) -> List[Parameter]:
        """Get all unique parameters from all models in the project.

        Parameters shared across multiple models (e.g. material SLD) are
        only included once to avoid double-counting.
        """
        parameters = []
        seen_ids: set[int] = set()
        if self._models is not None:
            for model in self._models:
                for param in model.get_all_parameters():
                    pid = id(param)
                    if pid not in seen_ids:
                        seen_ids.add(pid)
                        parameters.append(param)
        return parameters

    def _sync_parameter_states(self) -> None:
        """Apply project-level parameter enablement and default limits.

        Superphase thickness/roughness and subphase thickness are physically
        meaningless and are marked as disabled. Thickness and roughness default
        ranges are then applied only to enabled parameters created from project
        defaults, leaving explicit user-provided bounds untouched.
        """
        if self._models is None:
            return

        disabled_ids: set[int] = set()
        for model in self._models:
            sample = model.sample
            if sample is None or len(sample) == 0:
                continue
            superphase = sample.superphase
            if superphase is not None:
                disabled_ids.add(id(superphase.thickness))
                disabled_ids.add(id(superphase.roughness))
            subphase = sample.subphase
            if subphase is not None:
                disabled_ids.add(id(subphase.thickness))

        for model in self._models:
            sample = model.sample
            if sample is None or len(sample) == 0:
                continue
            for assembly in sample:
                for layer in assembly.layers:
                    self._sync_layer_parameter_state(layer.thickness, 'thickness', disabled_ids)
                    self._sync_layer_parameter_state(layer.roughness, 'roughness', disabled_ids)
                    magnetism = getattr(layer, 'magnetism', None)
                    if magnetism is not None:
                        # Magnetic parameters exist only on magnetic layers, so
                        # they are never in `disabled_ids`; theta_m carries
                        # explicit 0-360 bounds and needs no default window.
                        self._sync_layer_parameter_state(magnetism.rho_m, 'rho_m', disabled_ids)

    def _sync_layer_parameter_state(self, parameter: Parameter, kind: str, disabled_ids: set[int]) -> None:
        """Update a layer parameter's enabled state and pending default limits."""
        if id(parameter) in disabled_ids:
            parameter.enabled = False
            return

        if getattr(parameter, 'default_limits_pending', False):
            delattr(parameter, 'default_limits_pending')
            # A dependent parameter (a conformal follower) takes its bounds from its leader.
            if getattr(parameter, 'enabled', True) and parameter.independent:
                parameter.min = -np.inf
                parameter.max = np.inf
                apply_default_limits(parameter, kind)

    @property
    def q_min(self):
        """Q min."""
        if self._q_min is None:
            return Q_MIN
        return self._q_min

    @q_min.setter
    def q_min(self, value: float) -> None:
        """Q min."""
        self._q_min = value

    @property
    def q_max(self):
        """Q max."""
        if self._q_max is None:
            return Q_MAX
        return self._q_max

    @q_max.setter
    def q_max(self, value: float) -> None:
        """Q max."""
        self._q_max = value

    @property
    def q_resolution(self):
        """Q resolution."""
        if self._q_resolution is None:
            return Q_RESOLUTION
        return self._q_resolution

    @q_resolution.setter
    def q_resolution(self, value: int) -> None:
        """Q resolution."""
        self._q_resolution = value

    @property
    def current_material_index(self) -> Optional[int]:
        """Current material index."""
        return self._current_material_index

    @current_material_index.setter
    def current_material_index(self, value: int) -> None:
        """Current material index."""
        if value < 0 or value >= len(self._materials):
            raise ValueError(f'Index {value} out of range')
        if self._current_material_index != value:
            self._current_material_index = value

    @property
    def current_model_index(self) -> Optional[int]:
        """Current model index."""
        return self._current_model_index

    @current_model_index.setter
    def current_model_index(self, value: int) -> None:
        """Current model index."""
        if value < 0 or value >= len(self._models):
            raise ValueError(f'Index {value} out of range')
        if self._current_model_index != value:
            self._current_model_index = value
            self._current_assembly_index = 0
            self._current_layer_index = 0

    @property
    def current_assembly_index(self) -> Optional[int]:
        """Current assembly index."""
        return self._current_assembly_index

    @current_assembly_index.setter
    def current_assembly_index(self, value: int) -> None:
        """Current assembly index."""
        if value < 0 or value >= len(self._models[self._current_model_index].sample):
            raise ValueError(f'Index {value} out of range')
        if self._current_assembly_index != value:
            self._current_assembly_index = value
            self._current_layer_index = 0

    @property
    def current_layer_index(self) -> Optional[int]:
        """Current layer index."""
        return self._current_layer_index

    @current_layer_index.setter
    def current_layer_index(self, value: int) -> None:
        """Current layer index."""
        if value < 0 or value >= len(self._models[self._current_model_index].sample[self._current_assembly_index].layers):
            raise ValueError(f'Index {value} out of range')
        if self._current_layer_index != value:
            self._current_layer_index = value

    @property
    def current_experiment_index(self) -> Optional[int]:
        """Current experiment index."""
        return self._current_experiment_index

    @current_experiment_index.setter
    def current_experiment_index(self, value: int) -> None:
        """Current experiment index."""
        if value < 0 or value >= len(self._experiments):
            raise ValueError(f'Index {value} out of range')
        if self._current_experiment_index != value:
            self._current_experiment_index = value
            # Resetting the model index to 0 when changing the experiment
            # self.current_model_index = 0

    @property
    def created(self) -> bool:
        """Created function."""
        return self._created

    @property
    def path(self):
        """Path function."""
        return self._path_project_parent / self._info['name']

    def set_path_project_parent(self, path: Union[Path, str]):
        """Set path project parent."""
        self._path_project_parent = Path(path)

    @property
    def models(self) -> ModelCollection:
        """Models function."""
        return self._models

    @models.setter
    def models(self, models: ModelCollection) -> None:
        """Models function."""
        self._replace_collection(models, self._models)
        self._invalidate_fitter()
        # Use setter to update indicies for current model, assembly and layer
        self.current_model_index = 0
        # Only track materials not already in the project's material collection
        # (e.g. layers built from self._materials, as in default_model(), would
        # otherwise be re-added and trigger a spurious duplicate-item warning).
        self._extend_palette_from_models()
        for model in self._models:
            model.interface = self._calculator
        self._sync_parameter_states()

    @property
    def fitter(self) -> MultiFitter:
        """The fitter of the current model, configured from :attr:`fit_settings`.

        Every hand-out writes the project's settings onto it, so configure the
        fit through :attr:`fit_settings` (or :attr:`minimizer`), not through
        the returned fitter's core fitter.
        """
        if len(self._models):
            if (self._fitter is None) or (self._fitter_model_index != self._current_model_index):
                self._fitter = MultiFitter(self._models[self._current_model_index])
                self._fitter.settings = self._fit_settings
                self._fitter_model_index = self._current_model_index
                # Fits run through this fitter pick up the project's inequality
                # constraints automatically (resolved at fit time). A weak
                # reference avoids a project -> fitter -> project cycle that
                # would keep a discarded project (and its unique names) alive.
                self._fitter.constraints_factory_provider = _weak_constraints_provider(self)
            self._fit_settings.configure(self._fitter.easy_science_multi_fitter)
        return self._fitter

    def _invalidate_fitter(self) -> None:
        """Drop the cached fitter; the next :attr:`fitter` access builds a new one."""
        self._fitter = None
        self._fitter_model_index = None

    @property
    def fit_settings(self) -> FitSettings:
        """The project's fit configuration; edits apply to the next fit prepared."""
        return self._fit_settings

    @fit_settings.setter
    def fit_settings(self, settings: FitSettings) -> None:
        settings.validate()
        self._fit_settings = settings
        if self._fitter is not None:
            self._fitter.settings = settings

    def prepare_fit(
        self,
        experiments: Optional[list] = None,
        *,
        objective: Optional[str] = None,
        skip_invalid: bool = False,
    ) -> PreparedFit:
        """Prepare a fit of the experiments with a snapshot of :attr:`fit_settings`.

        The run is not executed: call :meth:`PreparedFit.execute`, or run its
        ``core_fitter`` in a worker thread, then hand the results to
        :meth:`record_fit`. Inequality constraints are resolved when it executes.

        A parameter of a fitted model that follows a free parameter of a model
        left out of the fit keeps following it, and that parameter is varied
        too (``prepared.added_roots``); the left-out data does not enter the fit.

        Parameters
        ----------
        experiments : list, optional
            The experiments to fit, in order. By default, the loaded ones
            whose ``include_in_fit`` is set.
        objective : str, optional
            Zero-variance objective overriding the settings' one.
        skip_invalid : bool, optional
            Leave out experiments that have no model in this project (listed
            in ``prepared.skipped``) instead of raising. By default, False.

        Returns
        -------
        PreparedFit
            The prepared run.

        Raises
        ------
        FitScopeError
            Nothing to fit, or an experiment has no model in this project.
        FitPreconditionError
            If a free parameter's bounds are unusable for the minimizer.
        """
        if experiments is None:
            experiments = [self._experiments[key] for key in sorted(self._experiments) if self._experiments[key].include_in_fit]
        invalid = [experiment for experiment in experiments if self._model_index(experiment.model) is None]
        if invalid and not skip_invalid:
            names = ', '.join(f"'{experiment.name}'" for experiment in invalid)
            raise FitScopeError(f'{names} cannot be fitted: no model of this project is assigned.')
        experiments = [experiment for experiment in experiments if experiment not in invalid]
        if not experiments:
            raise FitScopeError('No experiment to fit: include at least one experiment that has a model.')
        fitter = MultiFitter.for_experiments(experiments, constraints_factory_provider=_weak_constraints_provider(self))
        fitter.settings = self._fit_settings
        prepared = fitter.prepare(objective=objective)
        prepared.skipped = [experiment.name for experiment in invalid]
        keys = {id(experiment): key for key, experiment in self._experiments.items()}
        prepared.labels = [
            (keys.get(id(experiment)), experiment.name, None if channel is None else channel.value)
            for experiment in experiments
            for channel in (getattr(experiment, 'available_channels', None) or [None])
        ]
        return prepared

    def record_fit(self, prepared: PreparedFit, results: Optional[list] = None, status: str = 'completed') -> FitRun:
        """Record the outcome of a run prepared by :meth:`prepare_fit`; see :attr:`last_fit`.

        A completed run's results also reach :attr:`fitter`'s goodness-of-fit
        properties. A failed or cancelled one is recorded without statistics.
        """
        completed = status == 'completed' and bool(results)
        metrics = prepared.finalize(results) if completed else None
        self._last_fit = FitRun.from_results(prepared, results, status, self._fit_settings.minimizer.name, metrics)
        if completed:
            self.fitter.record_fit_results(results, metrics)
        return self._last_fit

    @property
    def last_fit(self) -> Optional[FitRun]:
        """The record of the latest fit recorded with :meth:`record_fit`, or None."""
        return self._last_fit

    # ----- structural parameter paths -----

    def _walk_parameters(self):
        """Yield ``(structural path, parameter)`` for every parameter under the models;
        a parameter of an object shared by several models at its first path only."""
        yield from paths.walk(self._models, ['models'])

    def parameter_path(self, parameter) -> Optional[str]:
        """Structural path of `parameter` within this project, e.g.
        ``models/0/sample/1/layers/0/thickness``.

        Paths are stable across save/load (unlike unique names, which are
        regenerated) and are the way inequality constraints reference
        parameters. Returns ``None`` when the parameter is not reachable
        from the project's models.
        """
        return next((path for path, candidate in self._walk_parameters() if candidate is parameter), None)

    def resolve_parameter_path(self, path: str):
        """Return the parameter at a structural `path` (see :meth:`parameter_path`)."""
        tokens = [t for t in str(path).split('/') if t != '']
        if not tokens or tokens[0] != 'models':
            raise KeyError(f"Parameter path must start with 'models': '{path}'.")
        obj = self._models
        for token in tokens[1:]:
            if token.lstrip('-').isdigit():
                try:
                    obj = obj[int(token)]
                except (IndexError, KeyError, TypeError):
                    raise KeyError(f"Parameter path '{path}': index {token} is out of range.") from None
            else:
                if token.startswith('_') or not hasattr(obj, token):
                    raise KeyError(f"Parameter path '{path}': unknown attribute '{token}'.")
                obj = getattr(obj, token)
        if not isinstance(obj, DescriptorNumberType):
            raise KeyError(f"Parameter path '{path}' does not point to a parameter.")
        return obj

    # ----- inequality constraints -----

    @property
    def inequality_constraints(self) -> List[InequalitySpec]:
        """The project's inequality constraints (a copy of the list)."""
        return list(self._inequality_constraints)

    def add_inequality_constraint(self, spec: InequalitySpec, validate: bool = True) -> InequalitySpec:
        """Register an inequality constraint; returns it.

        With ``validate`` the paths are resolved and the units of both sides
        compared, raising ``KeyError``/``ValueError`` on problems.
        """
        if not isinstance(spec, InequalitySpec):
            raise TypeError('spec must be an InequalitySpec')
        if validate:
            check_units(spec, self.resolve_parameter_path)
        self._inequality_constraints.append(spec)
        return spec

    def remove_inequality_constraint(self, which: Union[int, str, InequalitySpec]) -> None:
        """Remove a constraint by index, by name or by identity.

        A name removes **every** spec carrying that name; use the index or
        the spec object to remove a single one when names are shared.
        """
        if isinstance(which, InequalitySpec):
            self._inequality_constraints = [s for s in self._inequality_constraints if s is not which]
            return
        if isinstance(which, str):
            matches = [s for s in self._inequality_constraints if s.name == which]
            if not matches:
                raise KeyError(f"No inequality constraint named '{which}'.")
            for spec in matches:
                self._inequality_constraints.remove(spec)
            return
        del self._inequality_constraints[int(which)]

    def clear_inequality_constraints(self) -> None:
        self._inequality_constraints = []

    def evaluate_inequality_constraints(self) -> List[InequalityEvaluation]:
        """Evaluate every constraint (enabled or not) at the current values."""
        return [evaluate_spec(spec, self.resolve_parameter_path) for spec in self._inequality_constraints]

    def violated_inequality_constraints(self) -> List[InequalitySpec]:
        """Enabled constraints that the *current* parameter values violate.

        A fit started from an infeasible point begins on the BUMPS penalty
        plateau; callers should refuse or warn before launching.
        """
        violated = []
        for spec in self._inequality_constraints:
            if spec.enabled and not evaluate_spec(spec, self.resolve_parameter_path).satisfied:
                violated.append(spec)
        return violated

    def build_constraints_factory(self):
        """``constraints_factory`` hook for the enabled inequality constraints, or ``None``."""
        return build_constraints_factory(self._inequality_constraints, self.resolve_parameter_path)

    # ----- equality constraints (parameter dependencies) -----

    @staticmethod
    def _is_user_constrained(parameter) -> bool:
        """Whether `parameter` carries a constraint this project should persist.

        Both halves are needed: a constraint removed with the raw
        ``make_independent()`` leaves the marker behind, and re-applying it on
        load would resurrect what the user removed.
        """
        return getattr(parameter, USER_CONSTRAINT_FLAG, False) and not parameter.independent

    def _constraint_candidates(self) -> List[Parameter]:
        """Every parameter the project owns, model-reachable or not.

        Deliberately wider than :attr:`parameters`: a constraint on a material
        that no model uses cannot be addressed by a structural path, and has to
        be seen here so that :meth:`_constraint_record` can refuse it out loud
        rather than let it disappear at save time.
        """
        candidates = list(self.parameters)
        seen = {id(parameter) for parameter in candidates}
        if self._materials is not None:
            for parameter in self._materials.get_all_parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    candidates.append(parameter)
        return candidates

    def _user_constraints(self) -> List[dict]:
        """Records describing the constraints created via :mod:`easyreflectometry.constraints`.

        Parameters are addressed by structural path rather than by EasyScience
        serializer id: an id is minted lazily when a parameter first gains an
        observer and deleted again when it loses its last one, so it is not a
        durable handle.

        A parameter is recorded only when it is both marked *and* still
        dependent. A constraint removed with the raw ``make_independent()``
        leaves the marker behind, and re-applying that on load would resurrect
        something the user removed. Internal constraints carry no marker at all
        — their owning class rebuilds them in its own ``from_dict``.

        Raises
        ------
        ValueError
            If a constrained parameter, or a live parameter it depends on, is
            not reachable from the models, so no path can address it.
        """
        # Which parameters are constrained is decided from `_constraint_candidates`,
        # which enumerates them without walking properties. The structural walk has
        # to walk properties and leaves reference cycles behind (delaying collection
        # of the project and its unique names), so it runs only when there is
        # something to record, and only to supply the paths. Driving both from
        # one list keeps a constraint from being dropped because the two
        # enumerations disagree.
        constrained = [parameter for parameter in self._constraint_candidates() if self._is_user_constrained(parameter)]
        if not constrained:
            return []
        paths = {id(parameter): path for path, parameter in self._walk_parameters()}
        return [self._constraint_record(parameter, paths) for parameter in constrained]

    @staticmethod
    def _constraint_record(parameter, paths: dict) -> dict:
        """One save record for `parameter`, addressing everything through `paths`."""

        def _addressable(target, described_as: str) -> str:
            path = paths.get(id(target))
            if path is None:
                raise ValueError(
                    f"Cannot save the constraint on '{parameter.name}': {described_as} is not "
                    "reachable from the project's models. Only parameters that belong to a "
                    'model can take part in a saved constraint.'
                )
            return path

        dependencies = {}
        for alias, dependency in parameter._dependency_map.items():
            if isinstance(dependency, Parameter):
                # Embedding a live parameter by value would silently turn a
                # dependency into a frozen constant on load.
                dependencies[alias] = {'path': _addressable(dependency, f"its dependency '{dependency.name}'")}
            else:
                # An object-less constant built for the expression (the explicit
                # total of `constrain_to_sum`); nothing else serializes it, so it
                # is embedded here.
                dependencies[alias] = {
                    'name': dependency.name,
                    'value': float(dependency.value),
                    'unit': str(dependency.unit),
                }
        return {
            'target': _addressable(parameter, 'the parameter itself'),
            'expression': parameter._clean_dependency_string,
            'dependencies': dependencies,
        }

    def _sum_partner_max_backups(self) -> Dict[str, float]:
        """``{structural path: original max}`` for every parameter whose maximum
        :func:`easyreflectometry.constraints.clamp_sum_partners` narrowed.

        The backups live as an attribute on the parameters, which no other
        serialization records; without this, removing a sum constraint after a
        save/load could never widen the bounds back.
        """
        flagged = [parameter for parameter in self.parameters if hasattr(parameter, SUM_PARTNER_MAX_BACKUP)]
        if not flagged:
            return {}
        paths = {id(parameter): path for path, parameter in self._walk_parameters()}
        backups: Dict[str, float] = {}
        for parameter in flagged:
            path = paths.get(id(parameter))
            if path is not None:
                backups[path] = float(getattr(parameter, SUM_PARTNER_MAX_BACKUP))
        return backups

    def _restore_sum_partner_max_backups(self, backups: Dict[str, float]) -> None:
        """Re-attach the backups written by :meth:`_sum_partner_max_backups`."""
        for path, previous in (backups or {}).items():
            try:
                parameter = self.resolve_parameter_path(path)
            except KeyError:
                logger.warning('Cannot restore the bound backup for %r: the parameter no longer exists.', path)
                continue
            setattr(parameter, SUM_PARTNER_MAX_BACKUP, float(previous))

    def _restore_user_constraints(self, records: List[dict]) -> None:
        """Re-apply the constraint records written by :meth:`_user_constraints`.

        Records are applied in the order they were written. A chain
        (``a`` follows ``b`` follows ``c``) resolves whichever order it is
        restored in, because re-constraining a parameter propagates the new
        value to anything already following it.
        """
        for record in records:
            dependencies = {}
            for alias, reference in record['dependencies'].items():
                if 'path' not in reference:
                    dependencies[alias] = DescriptorNumberType(**reference)
                    continue
                try:
                    dependencies[alias] = self.resolve_parameter_path(reference['path'])
                except KeyError as error:
                    raise KeyError(
                        f'Cannot restore the constraint on {record["target"]!r}: its dependency '
                        f'{reference["path"]!r} does not exist in this project.'
                    ) from error
            constrain(self.resolve_parameter_path(record['target']), record['expression'], **dependencies)

    @property
    def calculator(self) -> str:
        """Calculator function."""
        return self._calculator.current_interface_name

    @calculator.setter
    def calculator(self, calculator: str) -> None:
        """Calculator function."""
        if calculator == self._calculator.current_interface_name:
            return

        self._calculator.switch(calculator)
        self._calculator.reset_storage()

        for model in self._models:
            model.generate_bindings()

        # The cached fitter holds fit functions bound to the previous backend.
        self._invalidate_fitter()

    @property
    def calculator_supports_magnetism(self) -> bool:
        """Whether the active calculator can model magnetic samples.

        The GUI uses this to gate magnetism-related controls (e.g. when the
        refnx or bornagain backend is selected).
        """
        return self._calculator().supports_magnetism

    @property
    def calculators_supporting_magnetism(self) -> List[str]:
        """Names of the available calculators that can model magnetic samples.

        Lets an application offer the switch a magnetic sample needs ("this
        requires refl1d — change to it?") instead of only reporting that the
        current calculator cannot do it. The active calculator is not touched.
        """
        supporting = []
        for name in self._calculator.available_interfaces:
            calculator = next(
                (candidate for candidate in CalculatorBase._calculators if candidate.name == name),
                None,
            )
            if calculator is not None and calculator().supports_magnetism:
                supporting.append(name)
        return supporting

    @property
    def models_have_magnetism(self) -> bool:
        """Whether any model in the project carries a magnetic layer.

        A calculator without magnetism cannot be selected while this holds — the
        binding it would have to build does not exist.
        """
        return any(model.has_magnetism for model in self._models)

    @property
    def minimizer(self) -> AvailableMinimizers:
        """The selected minimizer; an alias for ``fit_settings.minimizer``."""
        return self._fit_settings.minimizer

    @minimizer.setter
    def minimizer(self, minimizer: AvailableMinimizers) -> None:
        """Select the minimizer; it applies to the next fit."""
        if not isinstance(minimizer, AvailableMinimizers):
            raise ValueError(f'minimizer must be an AvailableMinimizers member, got {minimizer!r}.')
        previous = self._fit_settings.minimizer
        self._fit_settings.minimizer = minimizer
        logger.info('Minimizer changed from %s to %s', previous.name, self._fit_settings.minimizer.name)

    @property
    def experiments(self) -> Dict[int, Union[DataSet1D, PolarizedDataSet]]:
        """Experiments function."""
        return self._experiments

    @experiments.setter
    def experiments(self, experiments: Dict[int, Union[DataSet1D, PolarizedDataSet]]) -> None:
        """Experiments function."""
        self._experiments = experiments

    def remove_experiment(self, index: int) -> Dict[int, int]:
        """Remove the experiment at `index`; the later ones move down a key.

        Keys stay ``0..n-1`` and equal to display positions, so a caller that
        keeps per-experiment state (a selection, visibility) remaps it with
        the returned table.

        Returns
        -------
        Dict[int, int]
            Old key to new key of every remaining experiment.

        Raises
        ------
        IndexError
            No experiment at `index`.
        """
        if index not in self._experiments:
            raise IndexError(f'No experiment at index {index}')
        remaining = [key for key in sorted(self._experiments) if key != index]
        remap = {old: new for new, old in enumerate(remaining)}
        self._experiments = {remap[key]: self._experiments[key] for key in remaining}
        current = self._current_experiment_index
        self._current_experiment_index = remap.get(current, min(current, len(remaining) - 1)) if remaining else 0
        self._with_experiments = bool(self._experiments)
        if self._last_fit is not None:
            self._last_fit = self._last_fit.remapped(remap)
        return remap

    def model_index_for_experiment(self, index: int) -> Optional[int]:
        """Index of the model the experiment at `index` is bound to, or None when it has none."""
        return self._model_index_for_experiment(self._experiments[index])

    def set_model_for_experiment(self, index: int, model_index: int) -> None:
        """Bind the experiment at `index` to the model at `model_index`."""
        self._experiments[index].model = self._models[model_index]

    def experiments_for_model(self, model_index: int) -> List[int]:
        """Keys of the experiments bound to the model at `model_index`."""
        model = self._models[model_index]
        return [key for key, experiment in sorted(self._experiments.items()) if experiment.model is model]

    # ----- contrasts -----

    def add_contrast(
        self,
        reference_index: int,
        name: str,
        substitutions: Optional[Sequence[Union[ReplaceMaterial, ReplaceFormula]]] = None,
    ) -> int:
        """Add a contrast of the model at `reference_index`; see :func:`easyreflectometry.contrasts.derive_contrast`.

        Replacement materials join the materials palette. Returns the index of the new model.
        """
        reference = self._models[reference_index]
        model = derive_contrast(reference, name=name, substitutions=substitutions)
        model.color = self._models.next_color()
        self._models.append(model)
        model.interface = self._calculator
        self._extend_palette_from_models()
        self._sync_parameter_states()
        self._contrasts.append((model, reference))
        self._invalidate_fitter()
        return len(self._models) - 1

    def contrast_reference(self, model_index: int) -> Optional[int]:
        """Index of the model the model at `model_index` was derived from, if it was and still exists."""
        model = self._models[model_index]
        return self._model_index(next((reference for derived, reference in self._contrasts if derived is model), None))

    def parameter_models(self, parameter: Parameter) -> List[int]:
        """Indices of the models `parameter` belongs to (several when an object is shared)."""
        return self._parameter_owners().get(id(parameter), [])

    def _parameter_owners(self) -> Dict[int, List[int]]:
        owners: Dict[int, List[int]] = {}
        for index, model in enumerate(self._models):
            for parameter in {id(p): p for p in model.get_all_parameters()}.values():
                owners.setdefault(id(parameter), []).append(index)
        return owners

    @property
    def links(self) -> List[LinkRecord]:
        """The links made by :meth:`apply_link` (a copy of the list)."""
        return list(self._links)

    def plan_link(self, follower_index: int, reference_index: int, materials: Optional[str] = None) -> LinkPlan:
        """What tying the model at `follower_index` to the one at `reference_index` would do; see
        :func:`easyreflectometry.contrasts.plan_link`. Nothing is changed."""
        owners = self._parameter_owners()
        return plan_link(
            self._models[follower_index],
            self._models[reference_index],
            owners=lambda parameter: owners.get(id(parameter), []),
            follower_index=follower_index,
            materials=materials,
        )

    def apply_link(self, plan: LinkPlan) -> LinkRecord:
        """Tie the rows of `plan` marked ``'tie'``, all or none.

        Raises
        ------
        ValueError
            A row is still ``'undecided'``. Nothing is tied then.
        """
        undecided = [row.path for row in plan.rows if row.action == 'undecided']
        if undecided:
            raise ValueError(f'Decide whether to tie {", ".join(undecided)} before linking.')
        pairs = []
        try:
            for row in plan.rows:
                if row.action == 'tie':
                    state = snapshot(row.follower)
                    constrain_equal(row.follower, row.reference)
                    pairs.append((row.follower, row.reference, state))
        except Exception:
            for follower, _, state in pairs:
                unconstrain(follower)
                restore(follower, state)
            raise
        record = LinkRecord(plan.follower, plan.reference, pairs)
        self._links.append(record)
        return record

    def unlink(self, record: LinkRecord, restore_state: bool = True) -> List[int]:
        """Remove the ties `record` made. Each follower gets back its value, bounds and
        fixed state from before the link (`restore_state`), or keeps its current ones.
        Returns the indices of the models whose parameters were freed."""
        freed = []
        for follower, reference, state in record.pairs:
            if follows(follower, reference):
                unconstrain(follower)
                if restore_state:
                    restore(follower, state)
                freed.append(follower)
        self._links = [link for link in self._links if link is not record]
        owners = self._parameter_owners()
        return sorted({index for parameter in freed for index in owners.get(id(parameter), [])})

    def detach(self, parameter: Parameter, model_index: Optional[int] = None) -> None:
        """Make `parameter` independent (in the model at `model_index` only, when it is shared).

        A tie is removed (with the state from before it, when a link made it).
        A parameter of an assembly shared with other models gets a copy of that
        assembly in the model at `model_index`, with every other parameter tied
        to the shared one, so only `parameter`'s counterpart is free.

        Raises
        ------
        ValueError
            The parameter is derived; or it is shared and `model_index` is not
            given or not one of its models; or it belongs to a material the
            model shares (assign the layer another material instead).
        """
        if not parameter.independent:
            if not getattr(parameter, USER_CONSTRAINT_FLAG, False):
                raise ValueError(f"'{parameter.name}' is derived from other parameters; it cannot be detached.")
            self._untie(parameter)
            return
        if len(self.parameter_models(parameter)) < 2:
            return
        if model_index is None:
            raise ValueError(f"'{parameter.name}' is shared by several models; say which one to detach it in.")
        model = self._models[model_index]
        found = next(
            (
                (position, assembly)
                for position, assembly in enumerate(model.sample)
                if any(p is parameter for p in assembly.get_all_parameters())
            ),
            None,
        )
        if found is None:
            raise ValueError(f"'{parameter.name}' is not part of '{model.name}'.")
        position, assembly = found
        kept = kept_children(assembly, {}, set())
        if any(parameter is p for material in kept.values() for p in material.get_all_parameters()):
            raise ValueError(f"'{parameter.name}' belongs to a shared material; give the layer its own material instead.")
        copy = references.rebuild(assembly, kept)
        twin = dict(paths.walk(copy, []))[next(path for path, p in paths.walk(assembly, []) if p is parameter)]
        tie_corresponding(copy, assembly, excluded={id(twin)})
        model.sample[position] = copy
        model.interface = self._calculator
        self._sync_parameter_states()
        self._invalidate_fitter()

    def _untie(self, parameter: Parameter) -> None:
        """Make `parameter` independent; when a link tied it, with its state from before the link."""
        unconstrain(parameter)
        for record in self._links:
            for pair in record.pairs:
                if pair[0] is parameter:
                    record.pairs.remove(pair)
                    restore(parameter, pair[2])
                    self._links = [link for link in self._links if link.pairs]
                    return

    def _extend_palette_from_models(self) -> None:
        """Add the models' materials the palette does not hold yet."""
        new_materials = [material for material in self._get_materials_in_models() if material not in self._materials]
        self._materials.extend(new_materials)

    @property
    def path_json(self):
        """Path json."""
        return self.path / 'project.json'

    def get_index_air(self) -> int:
        """Get index air."""
        if 'Air' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='Air', sld=0.0, isld=0.0))
        return [material.name for material in self._materials].index('Air')

    def get_index_si(self) -> int:
        """Get index si."""
        if 'Si' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='Si', sld=2.07, isld=0.0))
        return [material.name for material in self._materials].index('Si')

    def get_index_sio2(self) -> int:
        """Get index sio2."""
        if 'SiO2' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='SiO2', sld=3.47, isld=0.0))
        return [material.name for material in self._materials].index('SiO2')

    def get_index_d2o(self) -> int:
        """Get index d2o."""
        if 'D2O' not in [material.name for material in self._materials]:
            self._materials.add_material(Material(name='D2O', sld=6.36, isld=0.0))
        return [material.name for material in self._materials].index('D2O')

    def load_orso_file(self, path: Union[Path, str]) -> None:
        """Load an ORSO file, creating a model from its ``sample.model`` (when present) and an experiment.

        .. deprecated::
            Use :meth:`load_new_experiment` (data) together with
            :meth:`set_sample_from_orso` /
            :func:`easyreflectometry.orso_utils.load_orso_model` (model)
            instead. This wrapper now routes through the same
            ``DataSet1D`` + title + resolution path as every other importer.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the ORSO file.
        """
        warnings.warn(
            'Project.load_orso_file is deprecated; use load_new_experiment (data) and '
            'set_sample_from_orso/load_orso_model (model) instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from easyreflectometry.orso_utils import _load_orso_any
        from easyreflectometry.orso_utils import load_orso_model

        orso_data = _load_orso_any(str(path))
        sample = load_orso_model(orso_data)
        if sample is not None:
            self.models = ModelCollection([Model(sample=sample, name=sample.name)])
        else:
            self.default_model()
        self.load_experiment_for_model_at_index(path, 0)

    def set_sample_from_orso(self, sample: Sample) -> None:
        """Replace the current project model collection with a single model built from an ORSO-parsed sample.

        This is a convenience helper for the ORSO import pipeline where a complete
        :class:`~easyreflectometry.sample.Sample` is constructed elsewhere.

        Parameters
        ----------
        sample : Sample
            Sample to set as the project's (single) model.

        Returns
        -------
        None
            ``None``.
        """
        model = Model(sample=sample)
        self.models = ModelCollection([model])

    def add_sample_from_orso(self, sample: Sample) -> None:
        """Add a new model with the given sample to the existing model collection.

        The created model is appended to :attr:`models`, its calculator interface is
        set to the project's current calculator, and any materials referenced in the
        sample are added to the project's material collection.

        After adding the model, :attr:`current_model_index` is updated to point to
        the newly added model.

        Parameters
        ----------
        sample : Sample
            Sample to add as a new model.

        Returns
        -------
        None
            ``None``.
        """
        if sample is None:
            raise ValueError('The ORSO file does not contain a valid sample model definition.')
        # Take the collection's next colour: a supplied model keeps its own
        # colour, and the Model default would give every loaded sample the
        # same first palette colour.
        model = Model(sample=sample, color=self._models.next_color())
        self.models.add_model(model)
        # Set interface after adding to collection
        model.interface = self._calculator
        # Extract materials from the new model and add to project materials
        self._materials.extend(self._get_materials_from_model(model))
        self._sync_parameter_states()
        # Switch to the newly added model so its data is visible in the UI
        self.current_model_index = len(self._models) - 1

    def replace_models_from_orso(self, sample: Sample) -> None:
        """Replace all models and materials with a single model from an ORSO sample.

        All existing models and their associated materials are removed. A new
        model is created from *sample*, assigned to the project's calculator,
        and the material collection is rebuilt from the new model only.

        Parameters
        ----------
        sample : Sample
            Sample to set as the project's only model.

        Returns
        -------
        None
            ``None``.
        """
        if sample is None:
            raise ValueError('The ORSO file does not contain a valid sample model definition.')
        model = Model(sample=sample)
        if sample.name:
            model.user_data['original_name'] = sample.name  # Store original name for reference
        self.models = ModelCollection([model])
        model.interface = self._calculator
        self._materials = self._get_materials_from_model(model)
        self.current_model_index = 0

    def _get_materials_from_model(self, model: Model) -> 'MaterialCollection':
        """Get all materials from a single model's sample."""
        materials_in_model = MaterialCollection(populate_if_none=False)
        for assembly in model.sample:
            for layer in assembly.layers:
                if layer.material not in materials_in_model:
                    materials_in_model.append(layer.material)
        return materials_in_model

    def _apply_experiment_metadata(
        self,
        path: Union[Path, str],
        experiment: DataSet1D,
        fallback_name: str,
        data_group=None,
        data_key: Optional[str] = None,
    ) -> None:
        """Set experiment name from ORSO title and configure the resolution function.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the experiment data file.
        experiment : DataSet1D
            The loaded experiment dataset to configure.
        fallback_name : str
            Name to use when no ORSO title is available.
        data_group :
            Pre-loaded scipp DataGroup (avoids reloading the file). By default, None.
        data_key : Optional[str], optional
            Specific dataset key to use for title extraction (e.g. ``'R_1'``). By default, None.
        """
        # Prefer ORSO title when available (keeps UI descriptive)
        title = None
        try:
            if data_group is None:
                data_group = load_data_from_orso_file(str(path))
            if data_key is None:
                data_key = list(data_group['data'].keys())[0]
            title = extract_orso_title(data_group, data_key)
        except (KeyError, AttributeError, ValueError, IndexError):
            title = None

        if title:
            experiment.name = title
        elif not experiment.name or experiment.name == 'Series':
            experiment.name = fallback_name

    def _apply_resolution_function(
        self,
        experiment: DataSet1D,
        model: Model,
    ) -> None:
        """Set the resolution of *experiment*, and of *model*, from the experiment's variance data.

        Uses the measured per-point q-resolution (``Pointwise``) when the
        experiment carries q-variance data (``xe``, i.e. sQz²); otherwise
        falls back to the default 5% FWHM percentage resolution.

        The measured resolution is stored on the experiment itself, so every
        dataset is fitted and plotted with the resolution it was measured
        with even when several datasets share one model. The model's
        resolution is set as well: it is what simulations, the GUI's
        resolution field and datasets without a measured resolution use.

        Parameters
        ----------
        experiment : DataSet1D
            The experiment whose variance data drives the choice.
        model : Model
            The model whose resolution function is set.
        """
        measured = resolution_from_dataset(experiment)
        experiment.resolution_function = measured
        model.resolution_function = PercentageFwhm(5.0) if measured is None else measured

    @staticmethod
    def _auto_set_background(experiment: DataSet1D) -> None:
        """Set the model background to the minimum y-value of the experiment data."""
        if experiment.model is not None and len(experiment.y) > 0:
            experiment.model.background = max(np.min(experiment.y), 1e-10)

    def _model_in_use(self, model: Model, ignoring=None) -> bool:
        """Whether a loaded experiment other than `ignoring` is bound to `model`."""
        return any(experiment.model is model for experiment in self._experiments.values() if experiment is not ignoring)

    def _bind_loaded_dataset(self, experiment: DataSet1D, model: Model, replacing=None) -> None:
        """Bind a freshly loaded dataset to `model`.

        The dataset keeps the resolution it was measured with. The model takes
        its background and resolution from the dataset only while no other
        experiment uses it, so loading another contrast or angle onto a model
        leaves what its first dataset (or the user) set. A background set by
        hand before the first dataset is loaded is replaced by that dataset's.
        """
        configure_model = not self._model_in_use(model, ignoring=replacing)
        experiment.model = model
        if configure_model:
            self._auto_set_background(experiment)
            self._apply_resolution_function(experiment, model)
        else:
            experiment.resolution_function = resolution_from_dataset(experiment)

    def load_new_experiment(self, path: Union[Path, str], data_group=None) -> None:
        """Load new experiment.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the experiment data file.
        data_group :
            Pre-loaded scipp DataGroup for *path* (avoids re-parsing the
            file). By default, None.
        """
        if data_group is None:
            data_group = load_measurement_file(str(path))
        new_experiment = load_as_dataset(str(path), data_group=data_group)
        new_index = len(self._experiments)

        model_index = 0
        if new_index < len(self.models):
            model_index = new_index

        self._apply_experiment_metadata(path, new_experiment, f'Experiment {new_index}', data_group=data_group)
        self._bind_loaded_dataset(new_experiment, self.models[model_index])
        self._experiments[new_index] = new_experiment
        self._with_experiments = True

    def count_datasets_in_file(self, path: Union[Path, str]) -> int:
        """Return the number of datasets contained in the file at *path*.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the data file.

        Returns
        -------
        int
            Number of datasets found; 1 if a non-ORSO file cannot be
            introspected. A corrupt ORSO file (banner present but unparsable)
            raises instead of being silently miscounted.
        """
        try:
            data_group = load_measurement_file(str(path))
            return len(data_group['data'])
        except Exception:
            if is_orso_file(str(path)):
                raise
            return 1

    def load_all_experiments_from_file(self, path: Union[Path, str]) -> int:
        """Load all datasets from a file as separate experiments sharing the current model.

        For a multi-dataset ORSO file (e.g. a multi-angle measurement), each dataset is
        registered as an independent experiment.  All experiments share the model that is
        currently selected.  Single-dataset files go through
        :meth:`load_new_experiment`.  A corrupt ORSO file raises (no silent
        plain-text fallback).

        Parameters
        ----------
        path : Union[Path, str]
            Path to the data file.

        Returns
        -------
        int
            Number of experiments that were added.
        """
        data_group = load_measurement_file(str(path))

        data_keys = sorted(data_group['data'].keys())
        if len(data_keys) <= 1:
            self.load_new_experiment(path, data_group=data_group)
            return 1

        # Every dataset is built before any is added, so a failure adds none.
        new_experiments = []
        for offset, data_key in enumerate(data_keys):
            fallback_name = f'Experiment {len(self._experiments) + offset}'
            new_experiment = dataset_from_datagroup(data_group, data_key=data_key)
            new_experiment.name = fallback_name
            self._apply_experiment_metadata(path, new_experiment, fallback_name, data_group=data_group, data_key=data_key)
            new_experiments.append(new_experiment)

        model = self.models[self._current_model_index]
        for new_experiment in new_experiments:
            self._bind_loaded_dataset(new_experiment, model)
            self._experiments[len(self._experiments)] = new_experiment
        self._with_experiments = True
        return len(data_keys)

    def _load_single_dataset(self, path: Union[Path, str]) -> DataSet1D:
        """Load a file that holds exactly one dataset, keeping its ORSO title as name."""
        data_group = load_measurement_file(str(path))
        if len(data_group['data']) > 1:
            raise ValueError(
                f"File '{path}' contains multiple datasets; a contrast is combined from single-dataset files "
                '(use load_all_experiments_from_file to load each dataset of this file as its own experiment).'
            )
        dataset = load_as_dataset(str(path), data_group=data_group)
        self._apply_experiment_metadata(path, dataset, Path(path).name, data_group=data_group)
        return dataset

    def load_experiment_from_files(
        self,
        paths: List[Union[Path, str]],
        model_index: Optional[int] = None,
        name: Optional[str] = None,
    ) -> int:
        """Load several files measured on one contrast as a single experiment.

        A contrast is often measured as several curves -- one per incident
        angle or wavelength band -- each with its own q-resolution. The files
        are concatenated (sorted by q) into one experiment whose every point
        keeps the resolution it was measured with, so the fit smears each
        point correctly. See :func:`easyreflectometry.data.merge_datasets`.

        Parameters
        ----------
        paths : List[Union[Path, str]]
            The files, one dataset each.
        model_index : Optional[int], optional
            Index of the model the experiment belongs to. By default, the
            current model.
        name : Optional[str], optional
            Name of the experiment. By default, the ORSO title of the first
            file, or ``'Experiment N'``.

        Returns
        -------
        int
            Index of the newly loaded experiment.
        """
        if not paths:
            raise ValueError('At least one file is required.')
        if model_index is None:
            model_index = self._current_model_index
        model = self.models[model_index]
        new_index = len(self._experiments)

        datasets = [self._load_single_dataset(path) for path in paths]
        # 'Series' is the DataSet1D default that _apply_experiment_metadata
        # replaces by the ORSO title of the first file, or else the fallback.
        experiment = merge_datasets(
            datasets, name='Series' if name is None else name, fill_resolution=self._fill_resolution_for(model)
        )
        if name is None:
            self._apply_experiment_metadata(paths[0], experiment, f'Experiment {new_index}')

        # The merged resolution is kept as merge_datasets built it from the
        # inputs' own resolutions; only a model no other experiment uses follows it.
        configure_model = not self._model_in_use(model)
        experiment.model = model
        if configure_model:
            self._auto_set_background(experiment)
            model.resolution_function = (
                PercentageFwhm(5.0) if experiment.resolution_function is None else experiment.resolution_function
            )
        self._experiments[new_index] = experiment
        self._with_experiments = True
        return new_index

    def append_to_experiment_at_index(self, index: int, path: Union[Path, str]) -> None:
        """Add another measured curve of the same contrast to an existing experiment.

        The file's points are merged into the experiment at *index* (sorted by
        q), each keeping its own resolution: the file's measured one, and the
        experiment's ``resolution_function`` as currently assigned. The
        experiment's name and model are kept.

        Parameters
        ----------
        index : int
            Index of the experiment to extend.
        path : Union[Path, str]
            The file to add, one dataset.

        Raises
        ------
        IndexError
            No experiment is loaded at *index*.
        ValueError
            The experiment is polarized (extend its channels one file at a
            time instead), or the file holds several datasets.
        """
        if index not in self._experiments:
            raise IndexError(f'No experiment at index {index}')
        experiment = self._experiments[index]
        if isinstance(experiment, PolarizedDataSet):
            raise ValueError('Cannot append a file to a polarized experiment; its channels are separate datasets.')
        model = experiment.model
        addition = self._load_single_dataset(path)
        merged = merge_datasets([experiment, addition], name=experiment.name, fill_resolution=self._fill_resolution_for(model))
        merged.model = model
        merged.include_in_fit = experiment.include_in_fit
        # merge_datasets kept the experiment's own resolution (measured or
        # explicitly assigned) for its points; the model follows the merged
        # one unless another experiment uses it. Without any, the experiment
        # keeps using the model's as it is.
        if model is not None and merged.resolution_function is not None and not self._model_in_use(model, ignoring=experiment):
            model.resolution_function = merged.resolution_function
        self._experiments[index] = merged

    @staticmethod
    def _fill_resolution_for(model: Optional[Model]) -> Optional[ResolutionFunction]:
        """The resolution to assume for merged points that carry none: the model's, unless that is itself measured."""
        if model is None:
            return None
        resolution_function = model.resolution_function
        if isinstance(resolution_function, PercentageFwhm):
            return resolution_function
        return None

    def suggest_polarized_channel_assignment(self, paths: List[Union[Path, str]]) -> Dict[str, Optional[PolarizationChannel]]:
        """Suggest a spin-channel assignment for a set of data files.

        For each file, the ORSO header polarization is used when present, falling
        back to filename heuristics ('_uu'/'_up' → pp, '_dd'/'_down' → mm, ...).
        Files that cannot be identified map to None; a GUI should present the
        result as an editable file → channel table.

        Parameters
        ----------
        paths : List[Union[Path, str]]
            Paths of the per-channel data files.

        Returns
        -------
        Dict[str, Optional[PolarizationChannel]]
            Detected channel (or None) per path.
        """
        return {str(path): detect_polarization_channel(str(path)) for path in paths}

    def load_polarized_experiment(
        self,
        paths: Dict[Union[PolarizationChannel, str], Union[Path, str]],
        model_index: Optional[int] = None,
    ) -> int:
        """Load a polarized experiment from one data file per spin channel.

        The channel datasets form a single :class:`PolarizedDataSet` experiment
        sharing one model. Two-channel NSF-only experiments simply pass 'pp' and
        'mm' entries; spin-flip channels are optional.

        Parameters
        ----------
        paths : Dict[Union[PolarizationChannel, str], Union[Path, str]]
            Explicit channel → file assignment (e.g. from the GUI import dialog,
            pre-filled via :meth:`suggest_polarized_channel_assignment`).
        model_index : Optional[int], optional
            Index of the model the experiment belongs to. By default, the
            current model.

        Returns
        -------
        int
            Index of the newly loaded experiment, so a caller can make it
            current.
        """
        paths = {PolarizationChannel(channel): path for channel, path in paths.items()}
        channels = {}
        title_data_group = None
        for channel, path in paths.items():
            # Parse each file once: dataset count, data, and title all come
            # from the same DataGroup (previously up to 3-4 parses per file).
            data_group = load_measurement_file(str(path))
            # One file per channel means one dataset per file; a multi-dataset ORSO
            # file has no defined channel-to-dataset assignment here.
            if len(data_group['data']) > 1:
                raise ValueError(
                    f"File '{path}' contains multiple datasets; use load_polarized_experiment_from_file "
                    f'for a single multi-dataset file, or export the {channel.value} channel to its own file.'
                )
            dataset = load_as_dataset(str(path), data_group=data_group)
            # Keep the source file visible per channel (file → channel provenance).
            dataset.name = f'{channel.value}: {Path(path).name}'
            channels[channel] = dataset
            if title_data_group is None:
                title_data_group = data_group

        first_path = next(iter(paths.values()))
        return self._register_polarized_experiment(channels, model_index, str(first_path), title_data_group)

    def load_polarized_experiment_from_file(
        self,
        path: Union[Path, str],
        model_index: Optional[int] = None,
    ) -> int:
        """Load a polarized experiment from a single multi-dataset ORSO file.

        Multi-dataset packing is the ORSO format's intended way to store spin
        states: each ``data_set:`` block is classified by its **own** header
        (``instrument_settings.polarization``), so per-dataset overrides are
        honoured. Only the fully-analysed cross-sections ``pp/pm/mp/mm`` are
        mapped; ``po/mo/op/om/unpolarized/vector`` are not coerced.

        Parameters
        ----------
        path : Union[Path, str]
            Path to the multi-dataset ORSO file.
        model_index : Optional[int], optional
            Index of the model the experiment belongs to. By default, the
            current model.

        Returns
        -------
        int
            Index of the newly loaded experiment.

        Raises
        ------
        ValueError :
            If any dataset lacks a mappable polarization header, or two
            datasets declare the same channel. Such files cannot be classified
            unambiguously; use :meth:`load_all_experiments_from_file` (e.g. for
            multi-angle files) or per-channel files instead.
        """
        from easyreflectometry.orso_utils import _load_orso_any
        from easyreflectometry.orso_utils import _orso_dataset_key
        from easyreflectometry.orso_utils import load_orso_data

        orso_data = _load_orso_any(str(path))
        classified = detect_polarization_channels_per_dataset(orso_data)

        channels = {}
        for i, (declared, channel) in enumerate(classified):
            label = _orso_dataset_key(orso_data[i], i)
            if channel is None:
                reason = 'declares no mappable spin channel' if declared else 'declares no polarization'
                raise ValueError(
                    f"Dataset '{label}' in '{path}' {reason} "
                    f'(only pp/pm/mp/mm are mapped; po/mo/op/om/unpolarized are not coerced). '
                    f'Use load_all_experiments_from_file for non-polarized multi-dataset files, '
                    f'or assign channels explicitly via load_polarized_experiment.'
                )
            if channel in channels:
                raise ValueError(
                    f"Duplicate spin channel '{channel.value}' in '{path}': more than one dataset "
                    f'declares it. Assign channels explicitly via load_polarized_experiment.'
                )
            channels[channel] = label

        data_group = load_orso_data(orso_data)
        channel_datasets = {}
        for channel, label in channels.items():
            dataset = dataset_from_datagroup(data_group, data_key=f'R_{label}')
            dataset.name = f'{channel.value}: {label}'
            channel_datasets[channel] = dataset

        return self._register_polarized_experiment(channel_datasets, model_index, str(path), data_group)

    def _register_polarized_experiment(self, channels, model_index, title_path, title_data_group) -> int:
        """Shared tail of the polarized loaders: build, name, and register the experiment.

        The background follows the first (in canonical order) channel. Every
        channel keeps the resolution it was measured with (its own
        ``Pointwise`` when the file carries sQz), which the fit and the
        per-channel model curves use; the model's resolution follows the
        first channel for simulations and the GUI's resolution field.

        Parameters
        ----------
        channels :
            Channel → DataSet1D mapping.
        model_index :
            Index of the model, or None for the current one.
        title_path :
            Path used for the fallback experiment title lookup.
        title_data_group :
            Pre-loaded DataGroup for the title lookup (avoids re-parsing).

        Returns
        -------
        int
            Index of the newly loaded experiment.
        """
        new_index = len(self._experiments)
        if model_index is None:
            model_index = self._current_model_index
        model = self.models[model_index]

        experiment = PolarizedDataSet(
            name=f'Polarized experiment {new_index}',
            channels=channels,
            model=model,
        )
        # Name from the ORSO title of the first file, when available.
        self._apply_experiment_metadata(
            title_path, experiment, f'Polarized experiment {new_index}', data_group=title_data_group
        )

        configure_model = not self._model_in_use(model)
        for channel in experiment.available_channels:
            experiment[channel].resolution_function = resolution_from_dataset(experiment[channel])
        if configure_model:
            # The model follows the first channel, see `_bind_loaded_dataset`.
            first_dataset = experiment[experiment.available_channels[0]]
            self._auto_set_background(first_dataset)
            self._apply_resolution_function(first_dataset, model)

        self._experiments[new_index] = experiment
        self._with_experiments = True
        return new_index

    def save_experiment_as_orso(self, path: Union[Path, str], index: Optional[int] = None) -> None:
        """Write an experiment to an ORSO file (`.ort` text, or `.orb` binary).

        The experiment's model (when set) is exported as the ORSO
        ``sample.model`` (slab representation); data columns are written as
        sigma. A polarized experiment becomes one file with one ``data_set:``
        block per spin channel.

        Parameters
        ----------
        path : Union[Path, str]
            Destination path; a ``.orb`` extension selects the binary format.
        index : Optional[int], optional
            Index of the experiment to save. By default, the current one.
        """
        if index is None:
            index = self._current_experiment_index
        experiment = self._experiments[index]
        save_orso_experiment(experiment, str(path), model=experiment.model)

    def load_experiment_for_model_at_index(self, path: Union[Path, str], index: Optional[int] = 0) -> None:
        """Load experiment for model at index."""
        data_group = load_measurement_file(str(path))
        experiment = load_as_dataset(str(path), data_group=data_group)

        self._apply_experiment_metadata(path, experiment, f'Experiment {index}', data_group=data_group)
        self._bind_loaded_dataset(experiment, self.models[index], replacing=self._experiments.get(index))
        self._experiments[index] = experiment
        self._with_experiments = True

    def _bind_calculator(self, model) -> None:
        """Bind the project's calculator to a model unless it already is.

        Reassigning ``model.interface`` re-propagates the interface over the
        whole sample tree — an expensive rebuild. The plot getters run on every
        chart refresh, so an already-bound model must be left alone; engine
        switches go through the ``calculator`` setter, which regenerates the
        bindings itself.
        """
        if model.interface is not self._calculator:
            model.interface = self._calculator

    def sld_data_for_model_at_index(self, index: int = 0) -> DataSet1D:
        """Sld data for model at index."""
        self._bind_calculator(self.models[index])
        sld = self.models[index].interface().sld_profile(self._models[index].unique_name)
        return DataSet1D(
            name=f'SLD for Model {index}',
            x=sld[0],
            y=sld[1],
        )

    def model_has_magnetism_at_index(self, index: int = 0) -> bool:
        """Whether the model at index carries magnetism (False when there is no such model)."""
        try:
            return bool(self.models[index].has_magnetism)
        except (IndexError, AttributeError):
            return False

    def magnetic_sld_data_for_model_at_index(self, index: int = 0) -> Dict[str, DataSet1D]:
        """Nuclear and magnetic depth profiles of a magnetic model.

        Parameters
        ----------
        index : int
            Index of the model.

        Returns
        -------
        Dict[str, DataSet1D]
            Profiles versus depth z, keyed:

            - ``'sld'``: nuclear ρ(z), the same curve as
              :meth:`sld_data_for_model_at_index`;
            - ``'rho_m'``: magnetic SLD ρM(z);
            - ``'theta_m'``: in-plane moment angle θM(z) in degrees, restricted
              to the depths that carry a moment — the angle of a zero-length
              vector is meaningless, so points below
              :data:`MAGNETIC_MOMENT_FLOOR_FRACTION` of the largest ρM are left
              out instead of drawing an arbitrary angle through vacuum;
            - ``'spin_up'`` / ``'spin_down'``: the potentials each spin state
              sees, ρ(z) ± ρM(z)·cos(θM(z) − A), where A is the guide-field
              angle :data:`GUIDE_FIELD_ANGLE`.

        Raises
        ------
        ValueError
            The model has no magnetic layer, so there is no magnetic profile.
        NotImplementedError
            The active calculator cannot model magnetism.
        """
        model = self.models[index]
        if not model.has_magnetism:
            raise ValueError(
                f'Model {index} has no magnetic layer; there is no magnetic SLD profile to show. '
                'Attach magnetism to a layer first.'
            )
        self._bind_calculator(model)
        z, sld, rho_m, theta_m = model.interface().magnetic_sld_profile(model.unique_name)
        z = np.asarray(z, dtype=float)
        sld = np.asarray(sld, dtype=float)
        rho_m = np.asarray(rho_m, dtype=float)
        theta_m = np.asarray(theta_m, dtype=float)
        # Component of the moment along the guide field: what the neutron spin
        # states add to / subtract from the nuclear potential.
        projection = rho_m * np.cos(np.radians(theta_m - GUIDE_FIELD_ANGLE))
        # Only report the angle where there is a moment to have an angle, and
        # make it continuous along z: 359° followed by 1° is a 2° turn, but a
        # plotted line through the wrapped values sweeps the whole circle.
        magnitude = np.abs(rho_m)
        has_moment = magnitude > MAGNETIC_MOMENT_FLOOR_FRACTION * magnitude.max(initial=0.0)
        theta_display = _unwrapped_angle(theta_m, has_moment)
        return {
            'sld': DataSet1D(name=f'SLD for Model {index}', x=z, y=sld),
            'rho_m': DataSet1D(name=f'Magnetic SLD for Model {index}', x=z, y=rho_m),
            'theta_m': DataSet1D(name=f'Moment angle for Model {index}', x=z[has_moment], y=theta_display[has_moment]),
            'spin_up': DataSet1D(name=f'Spin-up potential for Model {index}', x=z, y=sld + projection),
            'spin_down': DataSet1D(name=f'Spin-down potential for Model {index}', x=z, y=sld - projection),
        }

    def volume_fraction_data_for_model_at_index(
        self,
        index: int = 0,
        *,
        z: Optional[np.ndarray] = None,
        max_delta_z: Optional[float] = None,
        merge_equivalent: bool = True,
        groups: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> Dict[str, DataSet1D]:
        """Volume fraction (occupancy) depth profiles of the components of a model.

        Built from the sample alone, so no calculator binding is needed. See
        :func:`easyreflectometry.sample.volume_fraction.volume_fraction_profile`
        for the depth convention and the caveats.

        Parameters
        ----------
        index : int
            Index of the model.
        z : np.ndarray, optional
            Depths (Å) to evaluate at; ``z = 0`` is the superphase / first
            layer interface. By default a 500-point grid spanning the sample,
            the same grid the refnx calculator uses for its SLD profile.
        max_delta_z : float, optional
            Maximum spacing of the default grid. Cannot be combined with ``z``.
        merge_equivalent : bool, optional
            Merge materials with equal name and SLD into one component. By default, True.
        groups : Mapping[str, Sequence[str]], optional
            Presentation groups forwarded to
            :meth:`~easyreflectometry.sample.VolumeFractionProfile.grouped`,
            e.g. ``{'Water': ['D2O', 'H2O']}``.

        Returns
        -------
        Dict[str, DataSet1D]
            One profile per component, keyed by its stable component key (or
            group label) in front-to-back order; the display label is in the
            dataset's ``name``. The ``y`` values sum to one at every ``x``.
        """
        model = self.models[index]
        profile = volume_fraction_profile(model.sample, z, max_delta_z=max_delta_z, merge_equivalent=merge_equivalent)
        if groups:
            profile = profile.grouped(groups)
        return {
            key: DataSet1D(
                name=f'{profile.labels[key]} volume fraction for Model {index}',
                x=profile.z,
                y=values,
                x_label='z (Å)',
                y_label='Volume fraction',
            )
            for key, values in profile.fractions.items()
        }

    def sample_data_for_model_at_index(self, index: int = 0, q_range: Optional[np.array] = None) -> DataSet1D:
        """Sample data for model at index."""
        original_resolution_function = self.models[index].resolution_function
        self.models[index].resolution_function = PercentageFwhm(0)
        reflectivity_data = self.model_data_for_model_at_index(index, q_range)
        self.models[index].resolution_function = original_resolution_function

        return reflectivity_data

    def model_data_for_model_at_index(
        self,
        index: int = 0,
        q_range: Optional[np.array] = None,
        channel: Optional[Union[PolarizationChannel, str]] = None,
        resolution_function: Optional[ResolutionFunction] = None,
    ) -> DataSet1D:
        """Model data for model at index.

        Parameters
        ----------
        index : int
            Index of the model.
        q_range : Optional[np.array]
            Points to calculate at; the project q range by default.
        channel : Optional[Union[PolarizationChannel, str]]
            Spin cross-section to calculate. `None` (the default) gives the
            ordinary, channel-agnostic reflectivity. A channel requires a
            magnetic model (except 'pp', which falls back to the unpolarized
            calculation) and a calculator supporting magnetism, otherwise the
            calculator raises.
        resolution_function : Optional[ResolutionFunction]
            Resolution to smear with; the model's by default. Pass a dataset's
            own resolution to get the curve the fit compares that dataset to
            (see :meth:`model_data_for_experiment_at_index`).
        """
        if q_range is None:
            q_range = np.linspace(self.q_min, self.q_max, self.q_resolution)
        self._bind_calculator(self.models[index])
        interface = self.models[index].interface()
        unique_name = self._models[index].unique_name
        if channel is None:
            reflectivity = interface.reflectity_profile(q_range, unique_name, resolution_function=resolution_function)
            name = f'Reflectivity for Model {index}'
        else:
            channel = PolarizationChannel(channel)
            reflectivity = interface.reflectivity_profile_channel(
                q_range, unique_name, channel, resolution_function=resolution_function
            )
            name = f'Reflectivity ({channel.value}) for Model {index}'
        return DataSet1D(
            name=name,
            x=q_range,
            y=reflectivity,
        )

    def model_data_for_experiment_at_index(
        self,
        index: int = 0,
        channel: Optional[Union[PolarizationChannel, str]] = None,
        q_range: Optional[np.array] = None,
    ) -> DataSet1D:
        """The model curve an experiment is compared to: its model, at its q, with its resolution.

        Parameters
        ----------
        index : int
            Index of the experiment.
        channel : Optional[Union[PolarizationChannel, str]]
            Spin channel of a polarized experiment; required for one, not
            allowed for an unpolarized one.
        q_range : Optional[np.array]
            Points to calculate at; the experiment's own q by default.

        Raises
        ------
        IndexError
            No experiment is loaded at `index`.
        ValueError
            The experiment has no model, or `channel` does not match the
            experiment's kind.
        """
        experiment = self.experimental_data_for_model_at_index(index)
        if isinstance(experiment, PolarizedDataSet):
            if channel is None:
                raise ValueError(f'Experiment at index {index} is polarized; a channel is required.')
            dataset = experiment[channel]
        else:
            if channel is not None:
                raise ValueError(f'Experiment at index {index} is not polarized; it has no channel.')
            dataset = experiment
        model = experiment.model
        if model is None:
            raise ValueError(f'Experiment at index {index} has no model.')
        model_index = next(i for i, candidate in enumerate(self.models) if candidate is model)
        return self.model_data_for_model_at_index(
            model_index,
            q_range=np.asarray(dataset.x) if q_range is None else q_range,
            channel=channel,
            resolution_function=getattr(dataset, 'resolution_function', None),
        )

    def experimental_data_for_model_at_index(
        self,
        index: int = 0,
        channel: Optional[Union[PolarizationChannel, str]] = None,
    ) -> Union[DataSet1D, PolarizedDataSet]:
        """Experimental data for model at index.

        Parameters
        ----------
        index : int
            Index of the experiment.
        channel : Optional[Union[PolarizationChannel, str]]
            Spin channel of a polarized experiment. `None` (the default) keeps
            the historical behavior and returns the stored experiment as is: a
            `DataSet1D` for an unpolarized experiment, the whole
            `PolarizedDataSet` for a polarized one.

        Returns
        -------
        Union[DataSet1D, PolarizedDataSet]
            The experiment, or the `DataSet1D` of the requested channel.

        Raises
        ------
        IndexError
            No experiment is loaded at `index`.
        KeyError
            `channel` is a valid spin channel but was not measured.
        ValueError
            `channel` was given for an unpolarized experiment, or is not one of
            'pp', 'pm', 'mp', 'mm'.
        """
        if index not in self._experiments.keys():
            raise IndexError(f'No experiment data for model at index {index}')

        experiment = self._experiments[index]
        if channel is None:
            return experiment

        if not isinstance(experiment, PolarizedDataSet):
            raise ValueError(
                f"Experiment at index {index} is not polarized; it has no '{channel}' channel. "
                'Call without a channel argument to get its data.'
            )
        try:
            channel = PolarizationChannel(channel)
        except ValueError as exception:
            known = ', '.join(member.value for member in PolarizationChannel)
            raise ValueError(f"Unknown spin channel '{channel}'; expected one of {known}.") from exception
        if channel not in experiment:
            measured = ', '.join(member.value for member in experiment.available_channels)
            raise KeyError(f"Channel '{channel.value}' was not measured in experiment {index} (measured: {measured}).")
        return experiment[channel]

    def experiment_is_polarized_at_index(self, index: int = 0) -> bool:
        """Whether the experiment at index holds per-channel (polarized) data.

        Returns False when no experiment is loaded at `index`, so consumers can
        use it as a plain predicate.
        """
        return isinstance(self._experiments.get(index), PolarizedDataSet)

    def experiment_channels_at_index(self, index: int = 0) -> List[PolarizationChannel]:
        """Measured spin channels of the experiment at index ([] when unpolarized)."""
        experiment = self._experiments.get(index)
        if not isinstance(experiment, PolarizedDataSet):
            return []
        return experiment.available_channels

    def experiment_supports_spin_asymmetry_at_index(self, index: int = 0) -> bool:
        """Whether SA can be formed for the experiment at index.

        Spin asymmetry needs both non-spin-flip channels; an NSF-incomplete or
        spin-flip-only experiment has no SA. The two channel datasets must also
        be structurally usable (non-empty, matching lengths, a strictly ordered
        q grid) — an application asks this before offering a spin-asymmetry
        view, so a dataset that cannot be turned into one must not be advertised
        and then fail.
        """
        channels = self.experiment_channels_at_index(index)
        if not _has_both_nsf_channels(channels):
            return False
        experiment = self._experiments[index]
        try:
            for channel in (PolarizationChannel.PP, PolarizationChannel.MM):
                _ordered_channel_arrays(experiment[channel])
        except ValueError as exception:
            logger.warning('Experiment %s cannot produce a spin asymmetry: %s', index, exception)
            return False
        return True

    def spin_asymmetry_for_experiment_at_index(self, index: int = 0) -> Dict[str, object]:
        """Spin asymmetry SA = (R⁺⁺ − R⁻⁻) / (R⁺⁺ + R⁻⁻) of a polarized experiment.

        The measured SA is formed on the q grid of the pp channel. A mm channel
        measured on a different grid is linearly interpolated onto it — values
        with the usual weights, variances with the *squared* weights, which is
        the propagation rule for independent endpoints (the covariance this
        introduces between neighbouring SA points is not representable in
        `DataSet1D` and is discarded). Interpolation happens **only inside the q
        range both channels cover**: `np.interp` would otherwise clamp to the
        edge value and turn extrapolated points into fabricated measurements.

        Points are dropped when the denominator cannot carry a meaningful
        asymmetry:

        - it is not positive, or is lost to cancellation between the two
          channels (see :data:`SPIN_ASYMMETRY_CANCELLATION_FRACTION`) — this
          guard needs no uncertainties and is what keeps a background-subtracted
          tail from throwing ±10³ values onto the axis;
        - it is not above :data:`SPIN_ASYMMETRY_SIGNIFICANCE` times its own
          uncertainty, when the channels carry usable uncertainties;
        - the point itself is not usable (non-finite reflectivity, or a negative
          or non-finite variance).

        Parameters
        ----------
        index : int
            Index of the experiment.

        Returns
        -------
        Dict[str, object]
            - ``'measured'``: `DataSet1D` of SA versus q. As everywhere in this
              library, ``ye`` holds **variances**, not standard deviations.
            - ``'calculated'``: `DataSet1D` of the model SA on the same q
              points, or None when the model is not magnetic (an unpolarized
              model has SA ≡ 0, which is not worth drawing).
            - ``'masked_points'``: how many measured points were dropped for any
              of the reasons above.
            - ``'low_significance_points'``: of those, how many had a
              denominator below the significance threshold.
            - ``'small_denominator_points'``: of those, how many had a
              denominator that was non-positive or lost to cancellation.
            - ``'invalid_points'``: of those, how many carried a non-finite
              reflectivity or an unusable variance.
            - ``'out_of_overlap_points'``: number of pp points dropped because
              the mm channel does not cover their q.

        Raises
        ------
        IndexError
            No experiment is loaded at `index`.
        ValueError
            The experiment does not carry both pp and mm channels, or their
            datasets are not structurally usable.
        """
        if index not in self._experiments.keys():
            raise IndexError(f'No experiment data for model at index {index}')
        channels = self.experiment_channels_at_index(index)
        if not _has_both_nsf_channels(channels):
            raise ValueError(f'Experiment {index} does not have both non-spin-flip channels; spin asymmetry needs pp and mm.')

        experiment = self._experiments[index]
        q, r_pp, var_pp = _ordered_channel_arrays(experiment[PolarizationChannel.PP])
        q_mm, r_mm_source, var_mm_source = _ordered_channel_arrays(experiment[PolarizationChannel.MM])

        out_of_overlap = 0
        if q.shape == q_mm.shape and np.allclose(q, q_mm, rtol=1e-9, atol=0.0):
            # The usual case: both channels come from one instrument scan.
            # A tolerance keeps grids that only differ by float round-trips
            # (text files) on this path.
            r_mm, var_mm = r_mm_source, var_mm_source
        else:
            # Different grids: put mm on the pp grid, but only where mm has data
            # — np.interp clamps outside its range, which would silently invent
            # measurements at the edges.
            inside = (q >= q_mm.min()) & (q <= q_mm.max())
            out_of_overlap = int(np.count_nonzero(~inside))
            if out_of_overlap:
                logger.warning(
                    'Spin asymmetry of experiment %s: %s of %s pp points lie outside the mm q range '
                    '[%.5g, %.5g] and are dropped.',
                    index,
                    out_of_overlap,
                    q.size,
                    q_mm.min(),
                    q_mm.max(),
                )
            q, r_pp, var_pp = q[inside], r_pp[inside], var_pp[inside]
            r_mm, var_mm = _interpolate_with_variance(q, q_mm, r_mm_source, var_mm_source)

        asymmetry, variance, keep, reasons = _spin_asymmetry(r_pp, r_mm, var_pp, var_mm)
        measured = DataSet1D(
            name=f'Spin asymmetry for Experiment {index}',
            x=q[keep],
            y=asymmetry[keep],
            ye=variance[keep],
            x_label='q (1/angstrom)',
            y_label='Spin asymmetry',
        )

        calculated = None
        model_index = self._model_index_for_experiment(experiment)
        if model_index is not None and self.models[model_index].has_magnetism and measured.x.size > 0:
            calculated_pp = self.model_data_for_model_at_index(model_index, q_range=measured.x, channel='pp').y
            calculated_mm = self.model_data_for_model_at_index(model_index, q_range=measured.x, channel='mm').y
            calculated_asymmetry, _, _, _ = _spin_asymmetry(calculated_pp, calculated_mm)
            calculated = DataSet1D(
                name=f'Calculated spin asymmetry for Experiment {index}',
                x=measured.x,
                y=calculated_asymmetry,
                x_label='q (1/angstrom)',
                y_label='Spin asymmetry',
            )

        return {
            'measured': measured,
            'calculated': calculated,
            'masked_points': int(np.count_nonzero(~keep)),
            'out_of_overlap_points': out_of_overlap,
            **reasons,
        }

    def _model_index_for_experiment(self, experiment) -> Optional[int]:
        """Index of the model an experiment is bound to, or None."""
        return self._model_index(getattr(experiment, 'model', None))

    def default_model(self):
        """Default model."""
        self._replace_collection(MaterialCollection(interface=self._calculator), self._materials)

        layers = [
            Layer(
                material=self._materials[0],
                thickness=0.0,
                roughness=0.0,
                name='Vacuum Layer',
                interface=self._calculator,
            ),
            Layer(
                material=self._materials[1],
                thickness=100.0,
                roughness=3.0,
                name='D2O Layer',
                interface=self._calculator,
            ),
            Layer(
                material=self._materials[2],
                thickness=0.0,
                roughness=1.2,
                name='Si Layer',
                interface=self._calculator,
            ),
        ]
        assemblies = [
            Multilayer(layers[0], name='Superphase', interface=self._calculator),
            Multilayer(layers[1], name='D2O', interface=self._calculator),
            Multilayer(layers[2], name='Subphase', interface=self._calculator),
        ]
        sample = Sample(*assemblies, interface=self._calculator)
        model = Model(sample=sample, interface=self._calculator)
        model.is_default = True
        self.models = ModelCollection([model])

    def is_default_model(self, index: int) -> bool:
        """Check if the model at the given index is a default model.

        Parameters
        ----------
        index : int
            Index of the model to check.

        Returns
        -------
        bool
            True if the model was created as a default placeholder.
        """
        if index < 0 or index >= len(self._models):
            return False

        return self._models[index].is_default

    def plan_model_removal(self, index: int) -> 'ModelRemoval':
        """What removing the model at `index` would affect; nothing is changed.

        Raises
        ------
        IndexError :
            If the index is out of range.
        """
        if index < 0 or index >= len(self._models):
            raise IndexError(f'Model index {index} out of range')
        model = self._models[index]
        survivors = [other for other in self._models if other is not model]
        surviving = {id(parameter) for other in survivors for parameter in other.get_all_parameters()}
        # Parameters only this model owns; one shared with a surviving model stays.
        removed = {id(parameter) for parameter in model.get_all_parameters() if id(parameter) not in surviving}
        dependents = [
            parameter
            for other in survivors
            for parameter in other.get_all_parameters()
            if not parameter.independent and any(id(leader) in removed for leader in parameter.dependency_map.values())
        ]
        inequality_constraints = [
            spec
            for spec in self._inequality_constraints
            if any(id(self._parameter_or_none(path)) in removed for path in spec.paths.values())
        ]
        return ModelRemoval(
            experiments=self.experiments_for_model(index),
            dependents=list({id(parameter): parameter for parameter in dependents}.values()),
            inequality_constraints=inequality_constraints,
        )

    def remove_model_at_index(self, index: int, experiments: Union[str, int, None] = None) -> 'ModelRemoval':
        """Remove the model at `index`, resolving everything that refers to it.

        Parameters of other models that follow a parameter only this model
        owns become independent, keeping their current value (or getting back
        the one from before the link, when :meth:`apply_link` made the tie). Inequality
        constraints on such parameters are removed; the others are re-pathed.
        Objects the model shares with other models stay with them. See
        :meth:`plan_model_removal`.

        Parameters
        ----------
        index : int
            Index of the model to remove.
        experiments : Union[str, int, None], optional
            What happens to the experiments bound to the model: ``'remove'``
            removes them, an int rebinds them to the model at that index
            (counted before the removal). Required when there are any.

        Returns
        -------
        ModelRemoval
            What was affected.

        Raises
        ------
        IndexError :
            If the index is out of range.
        ValueError :
            If trying to remove the last remaining model, or experiments are
            bound to it and `experiments` does not say what to do with them.
            Nothing is changed then.
        """
        plan = self.plan_model_removal(index)
        if len(self._models) <= 1:
            raise ValueError('Cannot remove the last model from the project')
        if plan.experiments and experiments != 'remove':
            if not isinstance(experiments, int) or experiments == index or not 0 <= experiments < len(self._models):
                raise ValueError(
                    f'Experiments {plan.experiments} use the model; pass experiments="remove" '
                    'or the index of another model to bind them to.'
                )

        for parameter in plan.dependents:
            self._untie(parameter)
        removed = self._models[index]
        self._links = [link for link in self._links if removed not in (link.follower, link.reference)]
        self._contrasts = [pair for pair in self._contrasts if removed not in pair]
        if experiments == 'remove':
            for key in reversed(plan.experiments):
                self.remove_experiment(key)
        elif plan.experiments:
            for key in plan.experiments:
                self._experiments[key].model = self._models[experiments]
        self._keeping_inequality_paths(lambda: self._models.pop(index))
        self._invalidate_fitter()

        # Adjust current model index if necessary
        if self._current_model_index >= len(self._models):
            self._current_model_index = len(self._models) - 1
        elif self._current_model_index > index:
            self._current_model_index -= 1

        # Reset assembly and layer indices for the new current model
        self._current_assembly_index = 0
        self._current_layer_index = 0
        return plan

    def move_model(self, index: int, new_index: int) -> None:
        """Move the model at `index` to `new_index`, keeping inequality constraints on their parameters."""
        if index == new_index:
            return
        current = self._models[self._current_model_index]
        self._keeping_inequality_paths(lambda: self._models.insert(new_index, self._models.pop(index)))
        self._current_model_index = self._model_index(current)
        self._invalidate_fitter()

    def _parameter_or_none(self, path: str):
        try:
            return self.resolve_parameter_path(path)
        except KeyError:
            return None

    def _keeping_inequality_paths(self, edit) -> List[InequalitySpec]:
        """Run a structural `edit`, then re-path every inequality constraint to the
        parameters it referred to before. Constraints whose parameters are gone are
        removed and returned."""
        targets = [
            (spec, {alias: self._parameter_or_none(path) for alias, path in spec.paths.items()})
            for spec in self._inequality_constraints
        ]
        edit()
        locations = {id(parameter): path for path, parameter in self._walk_parameters()}
        kept, removed = [], []
        for spec, parameters in targets:
            new_paths = {alias: locations.get(id(parameter)) for alias, parameter in parameters.items()}
            if None in new_paths.values():
                removed.append(spec)
                continue
            spec.lhs_paths = {alias: new_paths[alias] for alias in spec.lhs_paths}
            spec.rhs_paths = {alias: new_paths[alias] for alias in spec.rhs_paths}
            kept.append(spec)
        self._inequality_constraints = kept
        return removed

    def add_material(self, material: Material) -> None:
        """Add a material to the project material collection.

        :param material: the material to add.
        :raises ValueError: if the material is already in the collection. The collection is
            left untouched in that case, so a caller cannot mistake a refused add for a
            successful one.
        """
        if material in self._materials:
            raise ValueError(f'Material {material.name} is already in the material collection.')
        self._materials.append(material)

    def remove_material(self, index: int) -> None:
        """Remove the material at *index* from the project material collection.

        :param index: position of the material in the collection.
        :raises IndexError: if there is no material at *index*.
        :raises ValueError: if the material is used by one of the models. Removing it would
            leave the model referring to a material the project no longer holds, so the
            collection is left untouched.
        """
        material = self._materials[index]
        if material in self._get_materials_in_models():
            raise ValueError(f'Material {material.name} is used in models and cannot be removed.')
        self._materials.pop(index)

    def _default_info(self):
        """Default info."""
        return dict(
            name='DefaultEasyReflectometryProject',
            short_description='Reflectometry, 1D',
            modified=datetime.datetime.now().strftime('%d.%m.%Y %H:%M'),
        )

    def create(self):
        """Create the project directory tree on disk.

        :raises FileExistsError: if the project directory already exists. Nothing is written in
            that case and `created` stays False, so a caller cannot mistake a collision for a new
            project.
        """
        if os.path.exists(self.path):
            raise FileExistsError(f'Directory {self.path} already exists. Choose a different name or location.')
        os.makedirs(self.path)
        os.makedirs(self.path / 'experiments')
        self._created = True
        self._timestamp_modification()

    def save_as_json(self, overwrite=False):
        """Save the project as json.

        The write is atomic: the serialized project is written to a temporary file in the
        destination directory and then moved into place with `os.replace`. Any failure while
        serializing or writing therefore leaves a previously saved project file untouched.

        Failures are raised rather than reported on stdout, so that callers (notably the GUI)
        can tell a successful save from a failed one.

        :param overwrite: whether an existing project file may be replaced.
        :raises FileExistsError: if the project file exists and `overwrite` is False.
        :raises ValueError: if the project cannot be serialized, e.g. when a constraint depends
            on a parameter that is not reachable from the models.
        :raises OSError: if the file cannot be written or moved into place.
        """
        if self.path_json.exists() and not overwrite:
            raise FileExistsError(f'File already exists {self.path_json}. Pass overwrite=True to replace it.')
        # Serialize before touching the file system, so that a serialization failure
        # leaves no temporary file behind and no doubt about the existing file.
        try:
            project_json = json.dumps(self.as_dict(include_materials_not_in_model=True), indent=4)
        except TypeError as error:
            # json.dumps reports a value it cannot encode (a stray numpy scalar, say) as a
            # TypeError; to the caller that is the same "cannot be serialized" failure.
            raise ValueError(f'The project cannot be serialized: {error}') from error
        self.path_json.parent.mkdir(exist_ok=True, parents=True)
        # The temporary file must share a directory with the destination, otherwise the
        # replace below is not guaranteed to be atomic (and fails across volumes on Windows).
        # The name is not a dot-file so that a leftover is visible while debugging.
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path_json.parent, prefix=f'{self.path_json.name}.', suffix='.tmp'
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, mode='w') as file:
                file.write(project_json)
                file.flush()
                # Rename without fsync can leave an empty file behind after a power loss.
                os.fsync(file.fileno())
            self._apply_project_file_mode(temporary_path)
            os.replace(temporary_path, self.path_json)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        logger.debug(f'Saved project to {self.path_json}')

    def _apply_project_file_mode(self, temporary_path: Path) -> None:
        """Give the temporary file the permissions the project file should end up with.

        `tempfile.mkstemp` creates files owner-only (0600) and `os.replace` carries that mode
        onto the destination. Without this step every save on POSIX would strip group and other
        read access, and overwriting a shared 0644 project file would silently make it private.
        An existing file keeps its mode; a new file gets the ordinary umask-derived one.
        """
        if self.path_json.exists():
            mode = stat.S_IMODE(self.path_json.stat().st_mode)
        else:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        try:
            os.chmod(temporary_path, mode)
        except OSError:
            # Some file systems refuse chmod (network shares, some Windows mounts); the write
            # itself is still fine, so this is not a reason to fail the save.
            logger.debug(f'Could not set the mode of {temporary_path}')

    def load_from_json(self, path: Optional[Union[Path, str]] = None):
        """Load a project file, replacing the current project state.

        :param path: the project file; defaults to this project's own `path_json`.
        :raises FileNotFoundError: if there is no file at `path`. The current project is left
            untouched.
        :raises ValueError: if the file is not valid JSON or predates the supported file format.
        """
        if path is None:
            path = self.path_json
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'File {path} does not exist')
        with open(path, 'r') as file:
            project_dict = json.load(file)
            self.reset()
            self.from_dict(project_dict)
        self._path_project_parent = path.parents[1]
        self._created = True

    #: Schema version embedded in every serialized project. Bumped from 1 → 2
    #: when the sample/model classes migrated from the legacy
    #: ``easyscience.ObjBase``/``CollectionBase`` pipeline to
    #: ``ModelBase``/``EasyList``. The on-disk shape of nested objects (Layer,
    #: Material, MaterialMixture, MaterialSolvated, LayerAreaPerMolecule, etc.)
    #: changed in a way that is not backward-compatible with v1 files.
    FILE_FORMAT = 2
    #: Written instead of :attr:`FILE_FORMAT` when the models share objects
    #: (reference nodes, see :mod:`easyreflectometry.sample.references`).
    #: Readers that predate it cannot represent the sharing and refuse the file.
    FILE_FORMAT_SHARED = 3

    def as_dict(self, include_materials_not_in_model=False):
        """As dict.

        Objects used in several places (a material in two layers, an assembly
        in two models) are written once and referred to elsewhere. The
        materials palette is written in order under ``materials``.
        ``include_materials_not_in_model`` adds the unused materials once more
        under their older key, for readers that predate ``materials``.
        """
        project_dict = {}
        project_dict['file_format'] = self.FILE_FORMAT
        project_dict['info'] = self._info
        project_dict['with_experiments'] = self._with_experiments
        with references.writing():
            if self._models is not None:
                project_dict['models'] = self._models.as_dict()
                project_dict['models']['unique_name'] = self._models.unique_name + '_to_prevent_collisions_on_load'
            project_dict['materials'] = [material.as_dict() for material in self._materials]
        references.prune(project_dict)
        # Older readers ignore `materials`, so only references inside the models change the format.
        if references.contains_references(project_dict.get('models')):
            project_dict['file_format'] = self.FILE_FORMAT_SHARED
        if include_materials_not_in_model:
            self._as_dict_add_materials_not_in_model_dict(project_dict)
        if self._with_experiments:
            self._as_dict_add_experiments(project_dict)
        project_dict['fit_settings'] = self._fit_settings.to_dict()
        # Kept for readers that predate `fit_settings`.
        project_dict['fitter_minimizer'] = project_dict['fit_settings']['minimizer']
        if self._calculator is not None:
            project_dict['calculator'] = self._calculator.current_interface_name
        if self._colors is not None:
            project_dict['colors'] = self._colors
        if self._inequality_constraints:
            project_dict['inequality_constraints'] = [spec.to_dict() for spec in self._inequality_constraints]
        parameter_constraints = self._user_constraints()
        if parameter_constraints:
            project_dict['parameter_constraints'] = parameter_constraints
        sum_partner_max_backups = self._sum_partner_max_backups()
        if sum_partner_max_backups:
            project_dict['sum_partner_max_backups'] = sum_partner_max_backups
        self._as_dict_add_contrasts(project_dict)
        return project_dict

    def _model_index(self, model: Model) -> Optional[int]:
        return next((index for index, candidate in enumerate(self._models) if candidate is model), None)

    def _as_dict_add_contrasts(self, project_dict: dict) -> None:
        """Record contrast provenance and links by model index and parameter path.

        Entries whose models, or tied parameters, are no longer in the project
        (models replaced wholesale) are left out.
        """
        contrasts = [[self._model_index(derived), self._model_index(reference)] for derived, reference in self._contrasts]
        contrasts = [pair for pair in contrasts if None not in pair]
        if contrasts:
            project_dict['contrasts'] = contrasts
        parameter_paths = {id(parameter): path for path, parameter in self._walk_parameters()} if self._links else {}
        links = []
        for link in self._links:
            models = [self._model_index(link.follower), self._model_index(link.reference)]
            pairs = [
                [parameter_paths[id(follower)], parameter_paths[id(reference)], state]
                for follower, reference, state in link.pairs
                if id(follower) in parameter_paths and id(reference) in parameter_paths
            ]
            if None not in models and pairs:
                links.append({'follower': models[0], 'reference': models[1], 'pairs': pairs})
        if links:
            project_dict['links'] = links

    def _from_dict_restore_contrasts(self, project_dict: dict, report: list[str]) -> None:
        """Restore contrast provenance and links; an entry that does not fit the models is reported and dropped."""
        self._contrasts, self._links = [], []
        for derived, reference in project_dict.get('contrasts', []):
            try:
                self._contrasts.append((self._models[derived], self._models[reference]))
            except (IndexError, TypeError):
                report.append(f'Contrast provenance {derived} from {reference} refers to a missing model; dropped.')
        for raw in project_dict.get('links', []):
            try:
                pairs = [
                    (self.resolve_parameter_path(follower), self.resolve_parameter_path(reference), state)
                    for follower, reference, state in raw['pairs']
                ]
                self._links.append(LinkRecord(self._models[raw['follower']], self._models[raw['reference']], pairs))
            except (KeyError, IndexError, TypeError) as error:
                report.append(f'A link between models could not be restored: {error}')

    def _as_dict_add_materials_not_in_model_dict(self, project_dict: dict):
        """As dict add materials not in model dict."""
        materials_not_in_model = []
        for material in self._materials:
            if material not in self._get_materials_in_models():
                materials_not_in_model.append(material)
        if len(materials_not_in_model) > 0:
            project_dict['materials_not_in_model'] = MaterialCollection(materials_not_in_model).as_dict(skip=['interface'])

    def _as_dict_add_experiments(self, project_dict: dict):
        """As dict add experiments."""
        project_dict['experiments'] = {}
        project_dict['experiments_models'] = {}
        project_dict['experiments_model_indices'] = {}
        project_dict['experiments_names'] = {}
        project_dict['experiments_fit_scope'] = {}

        for key, experiment in self._experiments.items():
            if isinstance(experiment, PolarizedDataSet):
                self._as_dict_add_polarized_experiment(project_dict, key, experiment)
                continue
            project_dict['experiments'][key] = [
                list(experiment.x),
                list(experiment.y),
                list(experiment.ye),
            ]
            resolution = self._dataset_resolution_as_dict(experiment)
            if experiment.xe is not None or resolution is not None:
                # Fourth entry xe (None when cleared), fifth the dataset's own
                # resolution. The fifth is written even when None, an explicit
                # "use the model's resolution"; only its absence (files written
                # before it existed) means "derive it from xe".
                project_dict['experiments'][key].append(None if experiment.xe is None else list(experiment.xe))
                project_dict['experiments'][key].append(resolution)
            self._as_dict_add_experiment_model(project_dict, key, experiment)

    @staticmethod
    def _dataset_resolution_as_dict(dataset: DataSet1D) -> Optional[dict]:
        """The dataset's own resolution, serialized, or None when it uses the model's."""
        resolution_function = getattr(dataset, 'resolution_function', None)
        if resolution_function is None:
            return None
        return resolution_function.as_dict()

    @staticmethod
    def _dataset_resolution_from_dict(dataset: DataSet1D, arrays: list, position: int) -> None:
        """Restore a dataset's own resolution from ``arrays[position]``.

        A present entry is used as it is, ``None`` included (the dataset uses
        the model's resolution). Older files, written before the entry
        existed, get the measured resolution derived from ``xe``.
        """
        if len(arrays) <= position:
            dataset.resolution_function = resolution_from_dataset(dataset)
        elif arrays[position] is None:
            dataset.resolution_function = None
        else:
            dataset.resolution_function = ResolutionFunction.from_dict(arrays[position])

    def _as_dict_add_experiment_model(self, project_dict: dict, key: int, experiment) -> None:
        """Record the experiment's name, fit inclusion and model: by index (authoritative) and by name (older readers)."""
        project_dict['experiments_names'][key] = experiment.name
        project_dict['experiments_fit_scope'][key] = experiment.include_in_fit
        model_index = self._model_index_for_experiment(experiment)
        if model_index is not None:
            project_dict['experiments_model_indices'][key] = model_index
            project_dict['experiments_models'][key] = experiment.model.name

    def _as_dict_add_polarized_experiment(self, project_dict: dict, key: int, experiment: PolarizedDataSet) -> None:
        """Serialize a `PolarizedDataSet`: one (name, x, y, ye, xe, resolution) array set per measured channel.

        `experiments[key]` is a plain list for an ordinary `DataSet1D` (see
        `_as_dict_add_experiments`); a dict here — tagged `'polarized': True` —
        is how `_from_dict_extract_experiments` tells the two apart on load.
        """
        project_dict['experiments'][key] = {
            'polarized': True,
            'channels': {
                channel.value: [
                    experiment[channel].name,
                    list(experiment[channel].x),
                    list(experiment[channel].y),
                    list(experiment[channel].ye),
                    None if experiment[channel].xe is None else list(experiment[channel].xe),
                    # Written even when None; see `_dataset_resolution_from_dict`.
                    self._dataset_resolution_as_dict(experiment[channel]),
                ]
                for channel in experiment.available_channels
            },
        }
        self._as_dict_add_experiment_model(project_dict, key, experiment)

    def from_dict(self, project_dict: dict):
        """From dict."""
        keys = list(project_dict.keys())
        # Validate file format. v1 files were written by the legacy
        # `ObjBase`/`CollectionBase` pipeline; their inner shapes (Layer,
        # Material, MaterialMixture, …) are not compatible with the v2
        # `ModelBase`/`EasyList` deserializer. Older files must be re-created.
        file_format = project_dict.get('file_format')
        if file_format is None:
            raise ValueError(
                'This project file predates file_format=2 and cannot be loaded by '
                'this version of easyreflectometry. The serialization format changed '
                'when the sample/model classes migrated from the legacy ObjBase / '
                'CollectionBase pipeline. Please re-create the project from its '
                'underlying data using the current API.'
            )
        if file_format not in (self.FILE_FORMAT, self.FILE_FORMAT_SHARED):
            raise ValueError(
                f'Unsupported project file_format={file_format!r}; this version of '
                f'easyreflectometry only reads file_format={self.FILE_FORMAT} and {self.FILE_FORMAT_SHARED}. '
                'Please either update easyreflectometry or re-create the project.'
            )
        report: list[str] = []
        self._info = project_dict['info']
        self._with_experiments = project_dict['with_experiments']
        if 'calculator' in keys:
            self._calculator.switch(project_dict['calculator'])
        with references.reading(report):
            if 'models' in keys:
                self.models = ModelCollection.from_dict(project_dict['models'])
            palette = [references.deserialize(entry) for entry in project_dict.get('materials', [])]
        if 'materials' in keys:
            self._replace_collection(palette, self._materials)
        else:
            # A file predating `materials`: the models' materials, then the unused ones.
            self._replace_collection(self._get_materials_in_models(), self._materials)
            if 'materials_not_in_model' in keys:
                self._materials.extend(MaterialCollection.from_dict(project_dict['materials_not_in_model']))
        # Settings are replaced wholesale, never merged with the previous project's.
        # A file predating `fit_settings` only stored `fitter_minimizer`.
        legacy = {'minimizer': project_dict.get('fitter_minimizer')}
        self._fit_settings, settings_report = FitSettings.from_dict(project_dict.get('fit_settings', legacy))
        report.extend(settings_report)
        self._invalidate_fitter()
        if 'experiments' in keys:
            self._experiments = self._from_dict_extract_experiments(project_dict, report)
        else:
            self._experiments = {}
        self.load_report = report

        # Resolve any pending parameter dependencies parked by the core
        # deserializer. Only cores that serialize nested dependencies produce
        # them; on the others this is a no-op safety net and `parameter_constraints`
        # below carries the user constraints instead.
        resolve_all_parameter_dependencies(self)
        self._restore_user_constraints(project_dict.get('parameter_constraints', []))
        self._restore_sum_partner_max_backups(project_dict.get('sum_partner_max_backups', {}))
        self._warn_on_unreadable_dependencies(project_dict.get('models'))
        # Inequality constraints are declarative (paths), nothing to resolve yet:
        # they are bound to parameters when a fit starts.
        self._inequality_constraints = [InequalitySpec.from_dict(raw) for raw in project_dict.get('inequality_constraints', [])]
        self._from_dict_restore_contrasts(project_dict, self.load_report)

    @staticmethod
    def _warn_on_unreadable_dependencies(models_dict) -> None:
        """Warn about embedded dependencies this build cannot restore.

        A core that serializes dependencies inside each nested parameter writes
        ``_dependency_string`` there. Cores without that feature drop the field
        silently on load, so such a file would lose its equality constraints
        with no signal at all. Detect it and say so; the constraints have to be
        re-applied by hand.

        The field is written for internal dependencies too (material mixtures,
        conformal roughness, ``Model.total_thickness``), and those are rebuilt
        by their owning class regardless — so its presence does not prove
        anything was actually lost. The wording is hedged accordingly.
        """

        def _contains_dependency(node) -> bool:
            if isinstance(node, dict):
                return '_dependency_string' in node or any(_contains_dependency(v) for v in node.values())
            if isinstance(node, (list, tuple)):
                return any(_contains_dependency(item) for item in node)
            return False

        if _contains_dependency(models_dict):
            warnings.warn(
                'This project was saved by a build that stores parameter dependencies inside '
                'each parameter, which this build cannot restore. Internal constraints are '
                'rebuilt automatically, but any custom equality constraints have been dropped '
                'and must be re-applied.',
                stacklevel=2,
            )

    def _from_dict_extract_experiments(self, project_dict: dict, report: list[str]) -> Dict[int, DataSet1D | PolarizedDataSet]:
        """From dict extract experiments."""
        experiments = {}
        for key, raw in project_dict['experiments'].items():
            name = project_dict['experiments_names'].get(key, f'Experiment {key}')
            model = self._saved_experiment_model(project_dict, key, name, report)
            if isinstance(raw, dict) and raw.get('polarized'):
                experiment = self._polarized_experiment_from_dict(name, raw, model)
            else:
                experiment = self._dataset_from_dict(name, raw, model)
            experiment.include_in_fit = project_dict.get('experiments_fit_scope', {}).get(key, True) is not False
            experiments[int(key)] = experiment
        return experiments

    def _dataset_from_dict(self, name: str, raw: list, model: Optional[Model]) -> DataSet1D:
        """Reconstruct an unpolarized experiment serialized by `_as_dict_add_experiments`."""
        dataset = DataSet1D(
            name=name,
            x=raw[0],
            y=raw[1],
            ye=raw[2],
            xe=raw[3] if len(raw) > 3 else None,
            model=model,
            auto_background=False,
        )
        self._dataset_resolution_from_dict(dataset, raw, 4)
        return dataset

    def _saved_experiment_model(self, project_dict: dict, key, name: str, report: list[str]) -> Optional[Model]:
        """The model a saved experiment is bound to: by index, else by name (files that
        predate the index); an ambiguous or missing one is reported."""
        index = project_dict.get('experiments_model_indices', {}).get(key)
        if isinstance(index, int) and 0 <= index < len(self._models):
            return self._models[index]
        model_name = project_dict.get('experiments_models', {}).get(key)
        matches = [model for model in self._models if model.name == model_name]
        if not matches:
            report.append(f"Experiment '{name}' has no model; assign one before fitting it.")
            return None
        if len(matches) > 1:
            report.append(f"Experiment '{name}': several models are named '{model_name}'; the first is used.")
        return matches[0]

    def _polarized_experiment_from_dict(self, name: str, raw: dict, model: Optional[Model]) -> PolarizedDataSet:
        """Reconstruct a `PolarizedDataSet` serialized by `_as_dict_add_polarized_experiment`."""
        channels = {}
        for channel_value, arrays in raw['channels'].items():
            dataset = DataSet1D(
                name=arrays[0],
                x=arrays[1],
                y=arrays[2],
                ye=arrays[3],
                xe=arrays[4],
                model=model,
                auto_background=False,
            )
            self._dataset_resolution_from_dict(dataset, arrays, 5)
            channels[channel_value] = dataset
        return PolarizedDataSet(name=name, channels=channels, model=model)

    def _get_materials_in_models(self) -> MaterialCollection:
        """Get materials in models."""
        materials_in_model = MaterialCollection(populate_if_none=False)
        for model in self._models:
            for assembly in model.sample:
                for layer in assembly.layers:
                    materials_in_model.append(layer.material)
        return materials_in_model

    def _replace_collection(self, src_collection: BaseCollection, dst_collection: BaseCollection) -> None:
        """Replace collection."""
        # Clear the destination collection
        for i in range(len(dst_collection)):
            dst_collection.pop(0)

        for element in src_collection:
            dst_collection.append(element)

    def _timestamp_modification(self):
        """Timestamp modification."""
        self._info['modified'] = datetime.datetime.now().strftime('%d.%m.%Y %H:%M')


def _unwrapped_angle(angle: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Make an angle profile continuous along z within each magnetic region.

    The moment angle is periodic, so a profile that turns smoothly from 359° to
    1° comes back from `atan2` as a jump of nearly 360°. Drawn as a line that is
    a full sweep across the chart where the moment barely moves. Each contiguous
    run of `mask` (a magnetic region) is therefore unwrapped on its own and then
    shifted by whole turns so it sits as close to the conventional [0, 360)
    range as possible — neighbouring regions stay independent, since the angle
    between them is not defined.

    Parameters
    ----------
    angle : np.ndarray
        Wrapped angles in degrees.
    mask : np.ndarray
        Which points carry a moment.

    Returns
    -------
    np.ndarray
        Angles in degrees, continuous within each masked region. Points outside
        the mask are returned unchanged (callers drop them).
    """
    unwrapped = np.array(angle, dtype=float, copy=True)
    if mask.size == 0 or not np.any(mask):
        return unwrapped

    # Each contiguous run of masked points is one magnetic region.
    masked_indices = np.flatnonzero(mask)
    region_breaks = np.flatnonzero(np.diff(masked_indices) > 1) + 1
    for region in np.split(masked_indices, region_breaks):
        segment = np.unwrap(unwrapped[region], period=360.0)
        # Keep the drawn values near the usual range rather than at 720 deg.
        turns = np.round(np.median(segment) / 360.0 - 0.5)
        unwrapped[region] = segment - turns * 360.0
    return unwrapped


def _has_both_nsf_channels(channels: List[PolarizationChannel]) -> bool:
    """Whether both non-spin-flip channels (pp, mm) — the pair spin asymmetry needs — are present."""
    return PolarizationChannel.PP in channels and PolarizationChannel.MM in channels


def _ordered_channel_arrays(dataset: DataSet1D) -> tuple:
    """A channel's (q, reflectivity, variance) arrays, ordered and checked.

    Spin asymmetry pairs two channels point by point and interpolates one onto
    the other, both of which assume well-formed, strictly increasing q. Rather
    than trusting that, the arrays are checked here and sorted when needed —
    `np.interp` silently returns nonsense for a descending grid.

    Raises
    ------
    ValueError
        The dataset is empty, its arrays disagree in length or shape, its q
        values are not finite, or it visits the same q twice.
    """
    name = getattr(dataset, 'name', '?')
    q = np.asarray(getattr(dataset, 'x', np.empty(0)), dtype=float).ravel()
    y = np.asarray(getattr(dataset, 'y', np.empty(0)), dtype=float).ravel()
    if q.size == 0:
        raise ValueError(f"Channel '{name}' has no data points.")
    if q.size != y.size:
        raise ValueError(f"Channel '{name}' has {q.size} q values for {y.size} reflectivities.")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"Channel '{name}' has non-finite q values.")

    variance = np.asarray(getattr(dataset, 'ye', None) if getattr(dataset, 'ye', None) is not None else [], dtype=float)
    variance = variance.ravel()
    if variance.size == 0:
        variance = np.zeros_like(y)
    elif variance.size != y.size:
        logger.warning(
            "Channel '%s' has %s uncertainties for %s points; treating it as having none.",
            name,
            variance.size,
            y.size,
        )
        variance = np.zeros_like(y)

    order = np.argsort(q, kind='stable')
    if not np.array_equal(order, np.arange(q.size)):
        logger.warning("Channel '%s' is not ordered in q; sorting it before pairing.", name)
        q, y, variance = q[order], y[order], variance[order]
    # A single-point channel has an empty `np.diff`, so the duplicate-q check
    # below is vacuously satisfied and it passes through here unrejected. That
    # is intentional: a one-point channel is a legitimate (if degenerate) SA
    # pair when it lines up exactly with the other channel's grid, and
    # `_interpolate_with_variance` handles a size-1 `q_source` correctly (its
    # clipped `searchsorted` result always resolves to that single point).
    if np.any(np.diff(q) <= 0):
        raise ValueError(f"Channel '{name}' visits the same q more than once; the pairing would be ambiguous.")
    return q, y, variance


def _interpolate_with_variance(q: np.ndarray, q_source: np.ndarray, values: np.ndarray, variances: np.ndarray) -> tuple:
    """Linear interpolation of values and their variances onto `q`.

    Values use the linear weights (1−t, t); variances use their **squares**,
    which is the propagation rule for independent endpoints — interpolating a
    variance linearly (as `np.interp` would) overestimates it, by a factor 2 at
    the midpoint of two equal variances.

    `q` must lie inside `q_source`; the caller restricts it to the overlap.
    """
    upper = np.clip(np.searchsorted(q_source, q, side='left'), 1, q_source.size - 1)
    lower = upper - 1
    span = q_source[upper] - q_source[lower]
    # span is > 0 for a strictly increasing source grid; guard anyway.
    weight = np.where(span > 0, (q - q_source[lower]) / np.where(span > 0, span, 1.0), 0.0)
    interpolated = (1.0 - weight) * values[lower] + weight * values[upper]
    interpolated_variance = (1.0 - weight) ** 2 * variances[lower] + weight**2 * variances[upper]
    return interpolated, interpolated_variance


def _spin_asymmetry(
    r_pp: np.ndarray,
    r_mm: np.ndarray,
    var_pp: Optional[np.ndarray] = None,
    var_mm: Optional[np.ndarray] = None,
) -> tuple:
    """Spin asymmetry, its variance, which points to keep, and why not.

    Parameters
    ----------
    r_pp, r_mm : np.ndarray
        Non-spin-flip reflectivities on a common q grid.
    var_pp, var_mm : Optional[np.ndarray]
        Their variances (`DataSet1D.ye`), or None for a calculated curve.

    Returns
    -------
    tuple
        SA, its variance (zeros without input variances), a boolean mask of the
        points to keep, and a dict counting the dropped ones by reason.
    """
    r_pp = np.asarray(r_pp, dtype=float)
    r_mm = np.asarray(r_mm, dtype=float)
    denominator = r_pp + r_mm
    var_pp = np.zeros_like(r_pp) if var_pp is None else np.asarray(var_pp, dtype=float)
    var_mm = np.zeros_like(r_mm) if var_mm is None else np.asarray(var_mm, dtype=float)

    # A denominator of exactly zero would divide by zero; those points are
    # dropped by the masks below anyway.
    safe_denominator = np.where(denominator == 0, np.nan, denominator)
    with np.errstate(invalid='ignore', divide='ignore'):
        asymmetry = (r_pp - r_mm) / safe_denominator
        # sigma_SA = 2 sqrt(R--^2 sigma_++^2 + R++^2 sigma_--^2) / (R++ + R--)^2,
        # so the variance is its square. ye holds variances, hence no squaring
        # of var_pp / var_mm here.
        variance = 4.0 * (r_mm**2 * var_pp + r_pp**2 * var_mm) / safe_denominator**4

    asymmetry = np.nan_to_num(asymmetry, nan=0.0, posinf=0.0, neginf=0.0)
    variance = np.nan_to_num(variance, nan=0.0, posinf=0.0, neginf=0.0)

    # A point with a non-finite reflectivity, or an uncertainty that is not a
    # usable variance, cannot produce a meaningful SA — and must not be silently
    # demoted to "no uncertainty", which would also skip the significance test.
    invalid = ~np.isfinite(r_pp) | ~np.isfinite(r_mm)
    invalid |= ~np.isfinite(var_pp) | ~np.isfinite(var_mm) | (var_pp < 0) | (var_mm < 0)

    # Uncertainty-independent guard: the denominator must be positive and must
    # not be the small remainder of two much larger numbers. Reflectivities are
    # positive, so this only bites on background-subtracted data — which is
    # exactly where SA otherwise explodes to +/-1e3 and destroys the axis.
    magnitude = np.abs(r_pp) + np.abs(r_mm)
    with np.errstate(invalid='ignore'):
        degenerate = ~np.isfinite(denominator) | (denominator <= 0)
        degenerate |= np.abs(denominator) <= SPIN_ASYMMETRY_CANCELLATION_FRACTION * magnitude
    degenerate &= ~invalid

    # Uncertainty-based guard: is the denominator above the noise?
    denominator_sigma = np.sqrt(np.clip(var_pp + var_mm, 0.0, None))
    with_uncertainty = denominator_sigma > 0
    insignificant = with_uncertainty & (denominator <= SPIN_ASYMMETRY_SIGNIFICANCE * denominator_sigma)
    insignificant &= ~invalid & ~degenerate

    keep = ~invalid & ~degenerate & ~insignificant
    reasons = {
        'invalid_points': int(np.count_nonzero(invalid)),
        'small_denominator_points': int(np.count_nonzero(degenerate)),
        'low_significance_points': int(np.count_nonzero(insignificant)),
    }
    return asymmetry, variance, keep, reasons
