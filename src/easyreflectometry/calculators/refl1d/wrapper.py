# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from refl1d import names
from refl1d.profile import build_profile
from refl1d.sample.layers import Repeat

from easyreflectometry.model import ResolutionFunction

from ..polarization import POLARIZATION_CHANNEL_TO_INDEX
from ..stateless_wrapper_base import StatelessWrapperBase
from ..stateless_wrapper_base import repeats_of
from ..stateless_wrapper_base import value_of

if TYPE_CHECKING:
    from easyreflectometry.model import Model
    from easyreflectometry.sample import Layer

# The kernel is evaluated on OVERSAMPLING_FACTOR evenly spaced points per data
# point, reaching RESOLUTION_PADDING resolution widths beyond the data, before
# it is smeared onto the data points.
RESOLUTION_PADDING = 3.5
OVERSAMPLING_FACTOR = 21

# refl1d convention: with the default guide field (Aguide = 270 deg) a moment at
# thetaM = 270 deg is aligned with the field, i.e. produces no spin-flip.
DEFAULT_THETA_M = 270.0

_REQUIRES_MAGNETISM = (
    '{} requires magnetism: enable it on this calculator first '
    '(`include_magnetism = True` on the calculator / `magnetism = True` on the wrapper).'
)


class Refl1dWrapper(StatelessWrapperBase):
    """Builds the refl1d sample from the model on every evaluation.

    The sample is built from :func:`_kernel_inputs` alone: one tuple of plain
    numbers holding everything the kernel consumes. That tuple is also the
    model part of the key of the polarized cache, so an input cannot reach the
    kernel without reaching the key.
    """

    supports_magnetism = True

    def calculate(
        self, q_array: np.ndarray, model_id: str | None = None, resolution_function: ResolutionFunction | None = None
    ) -> np.ndarray:
        """Reflectivity of the model at *q_array*; the selected spin channel when magnetism is on."""
        model = self._get_model(model_id)
        if self._magnetism:
            reflectivities = self._polarized_reflectivities(q_array, model, resolution_function)
            # Copy: the arrays live in the polarized cache and must not be mutated.
            return reflectivities[POLARIZATION_CHANNEL_TO_INDEX[self._polarization_channel]].copy()

        inputs = _kernel_inputs(model, magnetism=False)
        # smearing() returns sigma, which is exactly what refl1d's probe.dQ expects.
        dq_array = self._resolution_for(model.unique_name, resolution_function).smearing(q_array)
        probe = _probe(q_array, dq_array, inputs, oversampled_regions=_oversampled_regions(q_array, dq_array))
        _, reflectivity = names.Experiment(probe=probe, sample=_build_sample(inputs)).reflectivity()
        return reflectivity

    def calculate_polarized(
        self, q_array: np.ndarray, model_id: str | None, resolution_function: ResolutionFunction | None = None
    ) -> dict[str, np.ndarray]:
        """Reflectivity of all four spin channels, keyed 'pp', 'pm', 'mp', 'mm' (in that order)."""
        if not self._magnetism:
            raise ValueError(_REQUIRES_MAGNETISM.format('Polarized reflectivity'))
        reflectivities = self._polarized_reflectivities(q_array, self._get_model(model_id), resolution_function)
        # Copies: the arrays live in the polarized cache and must not be mutated.
        return {channel.value: reflectivities[index].copy() for channel, index in POLARIZATION_CHANNEL_TO_INDEX.items()}

    def _polarized_reflectivities(
        self, q_array: np.ndarray, model: Model, resolution_function: ResolutionFunction | None
    ) -> list[np.ndarray]:
        """Reflectivity of the four spin cross-sections, in refl1d order (mm, mp, pm, pp).

        All four come from one kernel evaluation. They are cached in the
        model's entry: per model state (:func:`_kernel_inputs`) any number of
        (q, dq) grids, one per dataset resolution. A different state replaces
        the whole cache.
        """
        # Normalized dtype plus explicit shape in the key: raw bytes alone do not
        # uniquely identify an ndarray.
        q_array = np.asarray(q_array, dtype=np.float64)
        dq_array = np.asarray(self._resolution_for(model.unique_name, resolution_function).smearing(q_array), np.float64)
        inputs = _kernel_inputs(model, magnetism=True)
        grid = (q_array.shape, q_array.tobytes(), dq_array.shape, dq_array.tobytes())
        entry = self._models[model.unique_name]
        if entry.cache_state != inputs:
            entry.cache_state, entry.cache = inputs, {}
        elif grid in entry.cache:
            return entry.cache[grid]

        probe = _polarized_probe(q_array, dq_array, inputs)
        reflectivities = [r for _, r in names.Experiment(probe=probe, sample=_build_sample(inputs)).reflectivity()]
        if len(reflectivities) != 4:
            raise RuntimeError(f'refl1d returned {len(reflectivities)} polarized cross-sections; expected 4.')
        for channel, index in POLARIZATION_CHANNEL_TO_INDEX.items():
            if len(reflectivities[index]) != len(q_array) or not np.all(np.isfinite(reflectivities[index])):
                raise RuntimeError(f'refl1d returned a malformed {channel.value} cross-section.')
        entry.cache[grid] = reflectivities
        return reflectivities

    def sld_profile(self, model_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """z and sld(z) of the model."""
        z, sld, _ = self._profile_experiment(model_id).smooth_profile()
        # -1 to reverse the order
        return z, sld[::-1]

    def magnetic_sld_profile(self, model_id: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """z, sld(z), magnetic sld rhoM(z) and magnetic angle thetaM(z).

        The magnetic profile is built by smoothing the two in-plane components
        of the moment and converting back, not by smoothing its magnitude and
        angle separately -- see :meth:`_smoothed_magnetic_vector`.
        """
        if not self._magnetism:
            raise ValueError(_REQUIRES_MAGNETISM.format('The magnetic sld profile'))
        experiment = self._profile_experiment(model_id)
        z, sld, _, _, _ = experiment.magnetic_smooth_profile()
        sld_magnetic, theta_magnetic = self._smoothed_magnetic_vector(experiment, z)
        # -1 to reverse the order
        return z, sld[::-1], sld_magnetic[::-1], theta_magnetic[::-1]

    def _profile_experiment(self, model_id: str | None) -> names.Experiment:
        """An experiment that only renders the sample: the probe is a dummy.

        A plain probe also for a magnetic sample: profiles never compute a
        per-channel reflectivity, so the polarized probe is not needed.
        """
        inputs = _kernel_inputs(self._get_model(model_id), magnetism=self._magnetism)
        return names.Experiment(probe=_probe(np.array([1.0]), np.array([1.0]), inputs), sample=_build_sample(inputs))

    @staticmethod
    def _smoothed_magnetic_vector(experiment: names.Experiment, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Magnitude and angle of the smoothed in-plane moment.

        refl1d smooths the magnetic microslabs channel by channel, so the
        magnitude |rhoM| and the angle thetaM are interpolated independently
        across an interface. For two layers whose moments differ by a couple of
        degrees around 0/360 (e.g. 359 and 1) the angle then takes the long way
        round the circle, passing through the guide-field direction: the profile
        reports the *full* moment as longitudinal exactly where it is almost
        entirely transverse, which shows up as a spurious spin-up/spin-down
        splitting at the interface.

        Smoothing the Cartesian components instead and converting back is
        interpolation of the moment as a vector, which is what the physics does.
        The reference angle used for the decomposition cancels out.

        Raises
        ------
        NotImplementedError
            The installed refl1d does not expose the microslab data this needs.
            Falling back to its angle-smoothed profile is deliberately *not*
            done: that output is wrong in a way a user cannot see, and a silent
            change of results after a dependency update is worse than no
            profile at all.
        """
        try:
            slabs = experiment._render_slabs()
            offsets = np.cumsum(slabs.w[:-1]) + slabs._z_offset
            roughness = slabs.sigma
            rho_m = np.asarray(slabs.rhoM, dtype=float)
            theta_m = np.asarray(slabs.thetaM, dtype=float)
        except (AttributeError, IndexError, TypeError) as exception:  # pragma: no cover - refl1d internals
            raise NotImplementedError(
                'The installed refl1d does not provide the microslab data needed for a '
                f'component-safe magnetic profile ({exception}). The magnetic depth profile is '
                "unavailable; refl1d's own profile smooths the moment angle separately, which "
                'misreports the spin-up/spin-down splitting at interfaces between differently '
                'oriented moments.'
            ) from exception

        relative_angle = np.radians(theta_m - DEFAULT_THETA_M)
        parallel = build_profile(z, offsets, roughness, rho_m * np.cos(relative_angle))
        perpendicular = build_profile(z, offsets, roughness, rho_m * np.sin(relative_angle))
        magnitude = np.hypot(parallel, perpendicular)
        angle = (DEFAULT_THETA_M + np.degrees(np.arctan2(perpendicular, parallel))) % 360.0
        return magnitude, angle


def _kernel_inputs(model: Model, magnetism: bool) -> tuple:
    """Every number the kernel consumes, in sample order.

    ``(assemblies, scale, background, guide_field_angle, applied_field)`` with
    ``assemblies`` a tuple of ``(repeats, layers)`` and each layer
    ``(thickness, roughness, sld, isld, magnetic)``, where ``magnetic`` is
    ``(rho_m, theta_m)`` for a magnetic layer when *magnetism* is on, else ``None``.
    """
    assemblies = tuple(
        (repeats_of(assembly), tuple(_layer_inputs(layer, magnetism) for layer in assembly.layers)) for assembly in model.sample
    )
    return (
        assemblies,
        model.scale.value,
        model.background.value,
        model.guide_field_angle.value,
        model.applied_field.value,
    )


def _layer_inputs(layer: Layer, magnetism: bool) -> tuple:
    magnetic = None
    if magnetism and layer.magnetism is not None:
        magnetic = (layer.magnetism.rho_m.value, layer.magnetism.theta_m.value)
    material = layer.material
    return (
        value_of(layer, 'thickness'),
        value_of(layer, 'roughness'),
        value_of(material, 'sld'),
        value_of(material, 'isld'),
        magnetic,
    )


def _build_sample(inputs: tuple) -> names.Stack:
    """The refl1d sample, substrate first; an assembly repeated once is flattened into its slabs."""
    sample = names.Stack()
    for repeats, layers in reversed(inputs[0]):
        slabs = [_slab(*layer) for layer in reversed(layers)]
        if repeats == 1:
            for slab in slabs:
                sample |= slab
        else:
            stack = names.Stack()
            for slab in slabs:
                stack |= slab
            sample |= Repeat(stack, repeat=repeats)
    return sample


def _slab(thickness: float, roughness: float, sld: float, isld: float, magnetic: tuple | None) -> names.Slab:
    # A slab without magnetism is fine inside a polarized calculation.
    magnetism = None if magnetic is None else names.Magnetism(rhoM=magnetic[0], thetaM=magnetic[1])
    return names.Slab(names.SLD(rho=sld, irho=isld), thickness=thickness, interface=roughness, magnetism=magnetism)


def _oversampled_regions(q_array: np.ndarray, dq_array: np.ndarray) -> list[names.OversampledRegion]:
    """Where refl1d evaluates the kernel besides the data points; see `OVERSAMPLING_FACTOR`.

    The region covers every point's own resolution, not only that of the
    first and last point: merged datasets repeat q with different widths.
    """
    padding = RESOLUTION_PADDING * np.asarray(dq_array)
    return [names.OversampledRegion(np.min(q_array - padding), np.max(q_array + padding), OVERSAMPLING_FACTOR * len(q_array))]


def _probe(q_array: np.ndarray, dq_array: np.ndarray, inputs: tuple, polarized: bool = False, **options) -> names.QProbe:
    """A probe at the model's scale and background; *options* go to `QProbe`."""
    _, scale, background, *_ = inputs
    probe = names.QProbe(Q=q_array, dQ=dq_array, intensity=scale, background=background, **options)
    if polarized:
        # `PolarizedQProbe` reads `theta_offset` off each of its probes; a plain
        # `QProbe` never sets it.
        probe.theta_offset = names.Parameter.default(0, name='theta_offset')
    return probe


def _polarized_probe(q_array: np.ndarray, dq_array: np.ndarray, inputs: tuple) -> names.PolarizedQProbe:
    """All four cross-sections (refl1d order mm, mp, pm, pp), in the model's guide field.

    The polarized probe oversamples the union of its cross-sections' grids
    itself; their own oversampling would not be used.
    """
    _, _, _, guide_field_angle, applied_field = inputs
    probes = [_probe(q_array, dq_array, inputs, polarized=True) for _ in range(4)]
    arguments = {
        'xs': probes,
        'name': 'polarized',
        'Aguide': guide_field_angle,
        'H': applied_field,
        'oversampled_regions': _oversampled_regions(q_array, dq_array),
    }
    try:
        return names.PolarizedQProbe(**arguments)
    except AttributeError:
        # refl1d 1.0.0 bug: `PolarizedQProbe.__init__` reads `_union_cache_key`
        # before assigning it (fixed in 1.0.1). Pre-seed it and run __init__ again.
        probe = names.PolarizedQProbe.__new__(names.PolarizedQProbe)
        probe._union_cache_key = None
        probe.__init__(**arguments)
        return probe
