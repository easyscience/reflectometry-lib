# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Volume fraction (occupancy) profiles of the components of a sample.

A volume fraction profile shows how much of the sample volume each *component*
(each distinct material) occupies as a function of depth. It is the natural
companion of the scattering length density (SLD) profile: where the SLD
profile shows one blended curve, the volume fraction profile separates it into
the lipid heads, the tails, the solvent, the substrate and so on, in the style
of ``refnx.reflect.create_occupancy``.

The profile is built directly from the sample tree, without a calculator:

- every layer is decomposed into its constituent materials
  (:class:`~easyreflectometry.sample.MaterialSolvated` and
  :class:`~easyreflectometry.sample.MaterialMixture` are split with their
  fraction, repeating multilayers are expanded, gradient layers are read as a
  linear mixing gradient between their two end materials), and
- the per-slab fractions are smeared across each interface with the same
  error-function kernel the plotted SLD profile uses, so the fractions sum to
  one at every depth and ``sum_c rho_c * phi_c(z)`` reproduces the SLD profile
  whenever each component has the same SLD in every slab it occupies.

Conventions and caveats:

- ``z = 0`` is the interface between the superphase and the first film layer
  and ``z`` increases into the sample, the same convention as the refnx
  calculator's SLD profile.
- The error-function kernel is the *plotting* convention shared with the SLD
  profile, not the Névot-Croce factor the reflectivity calculation uses; the
  curves are a model-derived in-plane average, not a measured concentration
  profile.
- Smearing is a sum of independent steps, so with very different roughness on
  the two sides of a thin layer a fraction can dip below zero. The values are
  returned unchanged (clipping or renormalising would change the model) and a
  :class:`VolumeFractionWarning` reports the excursion.
- A ``solvent_fraction`` may describe lateral patchiness rather than
  solvation; the in-plane average is drawn either way.
- For a :class:`~easyreflectometry.sample.LayerAreaPerMolecule` the fractions
  are those of the current material model, in which the solvent fraction acts
  as coverage of a layer whose molecule SLD is ``b / (thickness * area)``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Sequence

import numpy as np
from scipy.special import erf

from .assemblies.gradient_layer import GradientLayer
from .assemblies.repeating_multilayer import RepeatingMultilayer
from .elements.materials.material_mixture import MaterialMixture

if TYPE_CHECKING:
    from .collections.sample import Sample
    from .elements.layers.layer import Layer
    from .elements.materials.material import Material

__all__ = [
    'DEFAULT_POINTS',
    'NEGATIVE_FRACTION_TOLERANCE',
    'VolumeFractionProfile',
    'VolumeFractionWarning',
    'volume_fraction_profile',
]

#: Number of depth points of the default grid (the refnx convention).
DEFAULT_POINTS = 500
#: A fraction below ``-NEGATIVE_FRACTION_TOLERANCE`` triggers a :class:`VolumeFractionWarning`.
NEGATIVE_FRACTION_TOLERANCE = 1e-6
# A gradient sub-layer must carry the i/N blend of the end materials to this precision.
_GRADIENT_RELATIVE_TOLERANCE = 1e-9
_GRADIENT_ABSOLUTE_TOLERANCE = 1e-12
# Two materials are "equivalent" when name and SLDs agree to this many decimals (in 1e-6 Å^-2).
_EQUIVALENCE_DECIMALS = 10


class VolumeFractionWarning(UserWarning):
    """A volume fraction profile left the physically admissible range ``[0, 1]``.

    Emitted by :func:`volume_fraction_profile` when the smeared fraction of a
    component dips below zero, which happens when the roughness on the two
    sides of a thin layer is very different. The values are reported
    unchanged: the slab model, not the plot, is inadmissible there.
    """


@dataclass(frozen=True)
class _Slab:
    """One flat layer as the occupancy calculation sees it.

    ``roughness`` is the roughness of the interface *above* the slab (between it
    and the previous slab), which is what ``Layer.roughness`` stores and what
    refnx's slab column 3 means.
    """

    thickness: float
    roughness: float
    composition: dict
    name: str


@dataclass
class VolumeFractionProfile:
    """Volume fraction of every component of a sample versus depth.

    Attributes
    ----------
    z : np.ndarray
        Depth in Å; ``z = 0`` at the superphase / first layer interface.
    fractions : dict
        Component key -> ``phi_c(z)``. Keys are stable identifiers (the
        ``unique_name`` of the material, or of the first member of a merged
        group), in order of first appearance from the front of the sample.
    labels : dict
        Component key -> display label, unique within the profile.
    """

    z: np.ndarray
    fractions: dict
    labels: dict

    @property
    def components(self) -> list:
        """The component keys, front to back."""
        return list(self.fractions)

    @property
    def total(self) -> np.ndarray:
        """``sum_c phi_c(z)``; ones up to floating point error."""
        if not self.fractions:
            return np.zeros_like(self.z)
        return np.sum(np.array(list(self.fractions.values())), axis=0)

    def grouped(self, groups: Mapping[str, Sequence[str]]) -> 'VolumeFractionProfile':
        """Merge components into named groups for presentation.

        Parameters
        ----------
        groups : Mapping[str, Sequence[str]]
            Group label -> members, given as component keys or labels, e.g.
            ``{'Heads': ['C10H18NO8P'], 'Water': ['D2O', 'H2O']}``. Each
            component may appear at most once over all groups. Components not
            listed keep their own trace. The result lists the groups first, in
            the given order, then the remaining components in their original
            order.

        Returns
        -------
        VolumeFractionProfile
            A new profile; this one is left unchanged.

        Raises
        ------
        ValueError
            A member is unknown, a component is listed twice, a group is
            empty, or a group label collides with a retained component.
        """
        label_to_key = {label: key for key, label in self.labels.items()}
        used: dict = {}
        fractions: dict = {}
        labels: dict = {}
        for group_label, members in groups.items():
            keys = []
            for member in members:
                key = member if member in self.fractions else label_to_key.get(member)
                if key is None:
                    raise ValueError(
                        f'Unknown component {member!r} in group {group_label!r}. '
                        f'Available keys: {list(self.fractions)}; labels: {list(self.labels.values())}.'
                    )
                if key in used:
                    raise ValueError(
                        f'Component {self.labels[key]!r} is listed in group {group_label!r} '
                        f'and already in group {used[key]!r}; groups must not overlap.'
                    )
                used[key] = group_label
                keys.append(key)
            if not keys:
                raise ValueError(f'Group {group_label!r} has no members.')
            fractions[group_label] = np.sum(np.array([self.fractions[key] for key in keys]), axis=0)
            labels[group_label] = group_label
        for key, values in self.fractions.items():
            if key in used:
                continue
            label = self.labels[key]
            if key in fractions or label in labels.values():
                raise ValueError(
                    f'Group label {label!r} collides with the retained component {label!r}; '
                    'include that component in the group or rename the group.'
                )
            fractions[key] = values.copy()
            labels[key] = label
        return VolumeFractionProfile(z=self.z.copy(), fractions=fractions, labels=labels)


def volume_fraction_profile(
    sample: 'Sample',
    z: Optional[np.ndarray] = None,
    *,
    max_delta_z: Optional[float] = None,
    merge_equivalent: bool = True,
) -> VolumeFractionProfile:
    """Erf-smeared volume fraction of every component of ``sample`` versus depth.

    Parameters
    ----------
    sample : Sample
        The sample. It does not need to be bound to a calculator.
    z : np.ndarray, optional
        Depths (Å) at which to evaluate the profile, ``z = 0`` being the
        superphase / first layer interface. By default a grid of
        :data:`DEFAULT_POINTS` points spanning the sample plus four times the
        outer roughnesses on each side (the refnx convention; the extent only
        looks at the two outer interfaces, so pass a wider ``z`` when a broad
        internal interface matters).
    max_delta_z : float, optional
        Maximum spacing of the default grid; raises the point count above
        :data:`DEFAULT_POINTS` when needed. Cannot be combined with ``z``.
    merge_equivalent : bool, optional
        Merge materials the calculator cannot tell apart (equal name, SLD and
        imaginary SLD) into one component, so that the several default solvent
        objects of a bilayer come out as one "D2O" trace and its two head
        groups as one trace. ``False`` keeps every material object separate.
        Materials that share a name but differ in SLD are never merged; use
        :meth:`VolumeFractionProfile.grouped` for that. By default, True.

    Returns
    -------
    VolumeFractionProfile
        The profile. ``sum_c phi_c(z) == 1`` at every depth.

    Raises
    ------
    ValueError
        The sample has fewer than two layers, both ``z`` and ``max_delta_z``
        are given, a grid argument is invalid, a parameter value is not
        finite, or a :class:`~easyreflectometry.sample.GradientLayer` was
        edited after construction so that its sub-layers no longer carry the
        linear blend of its end materials.

    Warns
    -----
    VolumeFractionWarning
        A fraction dips below ``-NEGATIVE_FRACTION_TOLERANCE`` somewhere on the
        grid (strongly unequal roughness around a thin layer).
    """
    if z is not None and max_delta_z is not None:
        raise ValueError('Give either z or max_delta_z, not both.')
    slabs, labels = _resolve_slabs(sample, merge_equivalent=merge_equivalent)
    grid = _default_grid(slabs, max_delta_z) if z is None else _validated_grid(z)

    thickness = np.array([slab.thickness for slab in slabs], dtype=float)
    interfaces = np.cumsum(thickness[:-1])  # z of the interface above slab 1, 2, ..., M
    sigmas = np.array([slab.roughness for slab in slabs[1:]], dtype=float)
    fractions = {}
    for key in labels:
        values = np.array([slab.composition.get(key, 0.0) for slab in slabs], dtype=float)
        fractions[key] = _smear(grid, interfaces, sigmas, values)
    _warn_if_inadmissible(grid, fractions, labels)
    return VolumeFractionProfile(z=grid, fractions=fractions, labels=labels)


# ----- composition --------------------------------------------------------


def _composition(material) -> list:
    """Leaf materials of ``material`` with their volume fractions, which sum to one.

    A ``MaterialMixture`` (and so a ``MaterialSolvated``) is split into its two
    parts weighted by its fraction; the recursion covers nested mixtures should
    they become constructible. Anything else is a leaf.
    """
    if isinstance(material, MaterialMixture):
        p = float(material.fraction.value)
        parts = [(leaf, weight * (1.0 - p)) for leaf, weight in _composition(material.material_a)]
        parts += [(leaf, weight * p) for leaf, weight in _composition(material.material_b)]
        return parts
    return [(material, 1.0)]


class _ComponentRegistry:
    """Assigns stable keys and unique labels to the leaf materials of a sample."""

    def __init__(self, merge_equivalent: bool):
        self._merge_equivalent = merge_equivalent
        self._key_by_identity: dict = {}
        self._key_by_equivalence: dict = {}
        self._label_counts: dict = {}
        self.labels: dict = {}

    def key_for(self, material: 'Material') -> str:
        identity = material.unique_name
        key = self._key_by_identity.get(identity)
        if key is not None:
            return key
        key = identity
        if self._merge_equivalent:
            equivalence = (
                material.name,
                round(_scalar(material.sld), _EQUIVALENCE_DECIMALS),
                round(_scalar(material.isld), _EQUIVALENCE_DECIMALS),
            )
            key = self._key_by_equivalence.setdefault(equivalence, identity)
        self._key_by_identity[identity] = key
        if key == identity:
            self.labels[key] = self._unique_label(material.name)
        return key

    def _unique_label(self, name: str) -> str:
        count = self._label_counts.get(name, 0) + 1
        self._label_counts[name] = count
        return name if count == 1 else f'{name} ({count})'


def _scalar(value) -> float:
    """Float value of a Parameter or a plain number."""
    return float(getattr(value, 'value', value))


def _layer_entries(sample: 'Sample') -> Iterator:
    """Yield ``(layer, composition)`` front to back, with repetitions expanded."""
    for assembly in sample:
        if isinstance(assembly, GradientLayer):
            yield from _gradient_entries(assembly)
        elif isinstance(assembly, RepeatingMultilayer):
            repetitions = int(round(_scalar(assembly.repetitions)))
            for _ in range(repetitions):
                for layer in assembly.layers:
                    yield layer, _composition(layer.material)
        else:
            for layer in assembly.layers:
                yield layer, _composition(layer.material)


def _gradient_entries(assembly: GradientLayer) -> Iterator:
    """Sub-layer ``i`` of ``N`` is the ``i/N`` blend of the two end materials.

    That is how ``GradientLayer`` builds its sub-layers, but it freezes them at
    construction, so the blend is checked against each sub-layer's actual SLD
    and a stale gradient is refused rather than silently reinterpreted.
    """
    front, back = assembly.front_material, assembly.back_material
    layers = list(assembly.layers)
    elements = int(assembly.discretisation_elements)
    if len(layers) != elements:
        raise ValueError(
            f'GradientLayer {assembly.name!r} has {len(layers)} sub-layers but {elements} discretisation '
            'elements; its composition is undefined. Rebuild the gradient layer.'
        )
    for index, layer in enumerate(layers):
        fraction_back = index / elements
        for attribute in ('sld', 'isld'):
            expected = (1.0 - fraction_back) * _scalar(getattr(front, attribute)) + fraction_back * _scalar(
                getattr(back, attribute)
            )
            actual = _scalar(getattr(layer.material, attribute))
            if not math.isclose(actual, expected, rel_tol=_GRADIENT_RELATIVE_TOLERANCE, abs_tol=_GRADIENT_ABSOLUTE_TOLERANCE):
                raise ValueError(
                    f'GradientLayer {assembly.name!r}: sub-layer {index} has {attribute} {actual:g}, but the linear '
                    f'blend of its end materials gives {expected:g}. Its end materials or sub-layers were edited '
                    'after construction, so its composition is undefined. Rebuild the gradient layer.'
                )
        yield layer, [(front, 1.0 - fraction_back), (back, fraction_back)]


def _resolve_slabs(sample: 'Sample', merge_equivalent: bool = True) -> tuple:
    """Flatten ``sample`` into ``(slabs, labels)``.

    The first and last slabs are the superphase and subphase of the *expanded*
    list; their thickness is forced to zero as the calculators do.
    """
    registry = _ComponentRegistry(merge_equivalent)
    slabs = []
    for layer, parts in _layer_entries(sample):
        composition: dict = {}
        for material, weight in parts:
            key = registry.key_for(material)
            composition[key] = composition.get(key, 0.0) + float(weight)
        thickness = abs(_scalar(layer.thickness))
        roughness = abs(_scalar(layer.roughness))
        _check_finite(layer, thickness, roughness, composition)
        slabs.append(_Slab(thickness=thickness, roughness=roughness, composition=composition, name=layer.name))
    if len(slabs) < 2:
        raise ValueError(
            'A volume fraction profile needs at least a superphase and a subphase; the sample has fewer than two layers.'
        )
    for index in (0, -1):
        bounding = slabs[index]
        slabs[index] = _Slab(thickness=0.0, roughness=bounding.roughness, composition=bounding.composition, name=bounding.name)
    return slabs, registry.labels


def _check_finite(layer: 'Layer', thickness: float, roughness: float, composition: dict) -> None:
    if not (math.isfinite(thickness) and math.isfinite(roughness)):
        raise ValueError(f'Layer {layer.name!r} has a non-finite thickness or roughness.')
    if not all(math.isfinite(value) for value in composition.values()):
        raise ValueError(f'Layer {layer.name!r} has a non-finite material fraction.')


# ----- grid and smearing --------------------------------------------------


def _default_grid(slabs: list, max_delta_z: Optional[float]) -> np.ndarray:
    """refnx's grid: the film plus four outer roughnesses and 5 Å on each side."""
    total = sum(slab.thickness for slab in slabs)
    start = -5.0 - 4.0 * slabs[1].roughness
    end = 5.0 + total + 4.0 * slabs[-1].roughness
    points = DEFAULT_POINTS
    if max_delta_z is not None:
        max_delta_z = float(max_delta_z)
        if not (math.isfinite(max_delta_z) and max_delta_z > 0):
            raise ValueError('max_delta_z must be a finite, positive spacing in Å.')
        points = max(DEFAULT_POINTS, int(math.ceil((end - start) / max_delta_z)) + 1)
    return np.linspace(start, end, num=points)


def _validated_grid(z) -> np.ndarray:
    grid = np.asarray(z, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError('z must be a one-dimensional, non-empty array of depths in Å.')
    if not np.all(np.isfinite(grid)):
        raise ValueError('z must contain only finite depths.')
    return grid


def _smear(z: np.ndarray, interfaces: np.ndarray, sigmas: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum of erf steps: ``values[0] + sum_i (values[i+1] - values[i]) * H_sigma_i(z - interfaces[i])``.

    This is refnx's ``sld_profile`` loop applied to an arbitrary per-slab
    quantity. It is O(interfaces x points); plenty for hundreds of interfaces.
    Should a many-thousand-repetition stack ever need it, vectorise as
    ``steps @ H(z[None, :], interfaces[:, None], sigmas[:, None])``.
    """
    profile = np.full_like(z, values[0], dtype=float)
    for location, sigma, step in zip(interfaces, sigmas, np.diff(values)):
        if sigma == 0:
            # '>=' on purpose: refnx's Step gives H(0) = 1, and a grid point can land on an interface.
            kernel = np.where(z >= location, 1.0, 0.0)
        else:
            kernel = 0.5 * (1.0 + erf((z - location) / (sigma * math.sqrt(2.0))))
        profile += step * kernel
    return profile


def _warn_if_inadmissible(z: np.ndarray, fractions: dict, labels: dict) -> None:
    offenders = []
    for key, values in fractions.items():
        index = int(np.argmin(values))
        if values[index] < -NEGATIVE_FRACTION_TOLERANCE:
            offenders.append(f'{labels[key]!r} reaches {values[index]:.3g} at z = {z[index]:.2f} Å')
    if offenders:
        warnings.warn(
            'Volume fraction below zero: ' + '; '.join(offenders) + '. This happens when the roughness on the '
            'two sides of a thin layer is very different, so the slab model is not physically admissible there. '
            'The values are reported unchanged; a grid check cannot certify every depth in between.',
            VolumeFractionWarning,
            stacklevel=3,
        )
