# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import logging
from typing import Tuple

import numpy as np
from refl1d import names
from refl1d.profile import build_profile
from refl1d.sample.layers import Repeat

from ..polarization import POLARIZATION_CHANNEL_TO_INDEX
from ..wrapper_base import WrapperBase

logger = logging.getLogger(__name__)

RESOLUTION_PADDING = 3.5
OVERSAMPLING_FACTOR = 21

# refl1d convention: with the default guide field (Aguide = 270 deg) a moment at
# thetaM = 270 deg is aligned with the field, i.e. produces no spin-flip.
DEFAULT_THETA_M = 270.0


class Refl1dWrapper(WrapperBase):
    supports_magnetism = True

    def __init__(self):
        """Constructor."""
        super().__init__()
        # Magnetic values per layer name, kept outside the slabs so they survive
        # magnetism being toggled off/on and can be set before it is enabled.
        self._layer_magnetism: dict[str, dict[str, float]] = {}
        # Per-model cache of polarized reflectivities: all four cross-sections come
        # from a single kernel evaluation, so a simultaneous multi-channel fit costs
        # one evaluation per iteration instead of one per channel. Keyed on a token
        # of every model input; entries per (q, dq) grid, so channels measured on
        # different grids coexist within one iteration.
        self._polarized_cache: dict[str, dict] = {}

    def reset_storage(self):
        """Reset the storage area (including stored magnetic values) to blank."""
        super().reset_storage()
        self._layer_magnetism = {}
        self._polarized_cache = {}

    def create_material(self, name: str):
        """Create a material using SLD.

        Parameters
        ----------
        name : str
            The name of the material.
        """
        self.storage['material'][name] = names.SLD(str(name))

    def create_layer(self, name: str):
        """Create a layer using Slab.

        Parameters
        ----------
        name : str
            The name of the layer.
        """
        if self._magnetism:
            values = self._layer_magnetism.get(name, {})
            magnetism = names.Magnetism(
                rhoM=values.get('rhoM', 0.0),
                thetaM=values.get('thetaM', DEFAULT_THETA_M),
            )
        else:
            magnetism = None
        self.storage['layer'][name] = names.Slab(name=str(name), magnetism=magnetism)

    def create_item(self, name: str):
        """Create an item using Repeat.

        Parameters
        ----------
        name : str
            The name of the item.
        """
        self.storage['item'][name] = Repeat(names.Stack(names.Slab(names.SLD(), thickness=0, interface=0)), name=str(name))
        del self.storage['item'][name].stack[0]

    def update_layer(self, name: str, **kwargs):
        """Update a layer in a given item.

        Magnetic keys (`magnetism_rhoM`, `magnetism_thetaM`) may be passed alone or
        together; values are stored per layer and attached to the slab when
        magnetism is enabled.

        Parameters
        ----------
        name : str
            The layer name.
        **kwargs :
        """
        magnetic_values = {k.removeprefix('magnetism_'): v for k, v in kwargs.items() if k.startswith('magnetism_')}
        kwargs_no_magnetism = {k: v for k, v in kwargs.items() if not k.startswith('magnetism_')}
        super().update_layer(name, **kwargs_no_magnetism)
        if magnetic_values:
            stored = self._layer_magnetism.setdefault(name, {'rhoM': 0.0, 'thetaM': DEFAULT_THETA_M})
            stored.update(magnetic_values)
            if self._magnetism:
                self._apply_magnetism_to_layer(name)

    def get_layer_value(self, name: str, key: str) -> float:
        """A function to get a given layer value.

        Parameters
        ----------
        name : str
            The layer name.
        key : str
            The given value keys.
        """
        if key in ['magnetism_rhoM', 'magnetism_thetaM']:
            defaults = {'rhoM': 0.0, 'thetaM': DEFAULT_THETA_M}
            magnetic_key = key.removeprefix('magnetism_')
            return self._layer_magnetism.get(name, defaults).get(magnetic_key, defaults[magnetic_key])
        return super().get_layer_value(name, key)

    def _remove_magnetism_from_layers(self) -> None:
        """Detach Magnetism objects from all slabs.

        Called when magnetism is disabled: slabs carrying Magnetism objects would
        crash refl1d's plain (unpolarized) QProbe path. The magnetic values remain
        stored and are re-attached when magnetism is re-enabled.
        """
        for layer in self.storage['layer'].values():
            layer.magnetism = None

    def _apply_magnetism_to_layers(self) -> None:
        """Attach stored magnetic values to all slabs (called when magnetism is enabled)."""
        for name in self.storage['layer']:
            self._apply_magnetism_to_layer(name)

    def _apply_magnetism_to_layer(self, name: str) -> None:
        """Attach the stored magnetic values (or defaults) of one layer to its slab."""
        values = self._layer_magnetism.get(name, {})
        slab = self.storage['layer'][name]
        slab.magnetism = names.Magnetism(
            rhoM=values.get('rhoM', 0.0),
            thetaM=values.get('thetaM', DEFAULT_THETA_M),
        )

    def remove_layer_magnetism(self, name: str) -> None:
        """Remove the magnetic state of one layer; disable magnetism when none is left.

        Keeps `magnetism` (the calculator flag) in sync with the model: once no
        layer holds magnetic values any more, the polarized calculation path is
        switched off entirely.

        Parameters
        ----------
        name : str
            The layer name.
        """
        self._layer_magnetism.pop(name, None)
        slab = self.storage['layer'].get(name)
        if slab is not None:
            # A non-magnetic slab is fine inside a polarized calculation.
            slab.magnetism = None
        if self._magnetism and not self._layer_magnetism:
            # Goes through the `magnetism` property setter, which calls
            # `_remove_magnetism_from_layers()` again (a no-op here since
            # `_layer_magnetism` is already empty) and, more importantly,
            # resets `_polarization_channel` back to PP.
            self.magnetism = False

    def create_model(self, name: str):
        """Create a model for analysis.

        Parameters
        ----------
        name : str
            Name for the model.
        """
        self.storage['model'][name] = {'scale': 1, 'bkg': 0, 'items': []}

    def update_model(self, name: str, **kwargs):
        """Update the non-structural parameters of the model.

        Parameters
        ----------
        **kwargs :
        name : str
            Name of the model.
        """
        model = self.storage['model'][name]
        for key in kwargs.keys():
            model[key] = kwargs[key]

    def get_model_value(self, name: str, key: str) -> float:
        """A function to get a given model value.

        Parameters
        ----------
        name : str
            Name of the model.
        key : str
            The given value keys.

        Returns
        -------
        float
            The desired value.
        """
        model = self.storage['model'][name]
        return model[key]

    def assign_material_to_layer(self, material_name: str, layer_name: str):
        """Assign a material to a layer.

        Parameters
        ----------
        material_name : str
            The material name.
        layer_name : str
            The layer name.
        """
        self.storage['layer'][layer_name].material = self.storage['material'][material_name]

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
        item.stack.add(self.storage['layer'][layer_name])

    def add_item(self, item_name: str, model_name: str):
        """Add an item to the model.

        Parameters
        ----------
        item_name : str
            Items to add to model.
        model_name : str
            Name for the model.
        """
        self.storage['model'][model_name]['items'].append(self.storage['item'][item_name])

    def remove_layer_from_item(self, layer_name: str, item_name: str):
        """Remove a layer in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        layer_idx = list(self.storage['item'][item_name].stack).index(self.storage['layer'][layer_name])
        del self.storage['item'][item_name].stack[layer_idx]

    def remove_item(self, item_name: str, model_name: str):
        """Remove a given item.

        Parameters
        ----------
        item_name : str
            The item name.
        model_name : str
            The model name.
        """
        item_idx = self.storage['model'][model_name]['items'].index(self.storage['item'][item_name])
        del self.storage['model'][model_name]['items'][item_idx]
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
        if self._magnetism:
            reflectivities = self._polarized_reflectivities(q_array, model_name)
            # Copy: the arrays live in the polarized cache and must not be mutated.
            return reflectivities[POLARIZATION_CHANNEL_TO_INDEX[self._polarization_channel]].copy()

        sample = _build_sample(self.storage, model_name)
        # smearing() returns sigma, which is exactly what refl1d's probe.dQ expects.
        dq_array = self._resolution_function.smearing(q_array)
        probe = _get_probe(
            q_array=q_array,
            dq_array=dq_array,
            model_name=model_name,
            storage=self.storage,
            oversampling_factor=OVERSAMPLING_FACTOR,
        )
        # returns q, reflectivity
        _, reflectivity = names.Experiment(probe=probe, sample=sample).reflectivity()
        return reflectivity

    def calculate_polarized(self, q_array: np.ndarray, model_name: str) -> dict[str, np.ndarray]:
        """For a given q array calculate the reflectivity of all four spin channels.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_name : str
            The model name.

        Returns
        -------
        dict[str, np.ndarray]
            Reflectivity per spin channel, keyed 'pp', 'pm', 'mp', 'mm' (in that order).
        """
        if not self._magnetism:
            raise ValueError(
                'Polarized reflectivity requires magnetism: enable it on this calculator first '
                '(`include_magnetism = True` on the calculator / `magnetism = True` on the wrapper).'
            )
        reflectivities = self._polarized_reflectivities(q_array, model_name)
        # Copies: the arrays live in the polarized cache and must not be mutated.
        return {channel.value: reflectivities[index].copy() for channel, index in POLARIZATION_CHANNEL_TO_INDEX.items()}

    def _model_state_token(self, model_name: str) -> tuple:
        """A token of every model input that affects the reflectivity.

        Two calls with equal tokens (and equal q/dq grids) are guaranteed to
        produce the same reflectivity, so cached cross-sections can be reused.
        The resolution function needs no entry here — it enters through the dq
        part of the per-grid cache key. Must be extended whenever a new slab or
        material attribute starts reaching the kernel: a forgotten kernel input
        would silently serve stale reflectivities whenever only that input
        changes, since the token would compare equal.
        """
        model = self.storage['model'][model_name]
        values: list = [model['scale'], model['bkg']]
        for item in model['items']:
            values.append(item.repeat.value)
            for slab in item.stack:
                values.extend((
                    slab.thickness.value,
                    slab.interface.value,
                    slab.material.rho.value,
                    slab.material.irho.value,
                ))
                if slab.magnetism is None:
                    values.append(None)
                else:
                    values.extend((slab.magnetism.rhoM.value, slab.magnetism.thetaM.value))
        return tuple(values)

    def _polarized_reflectivities(self, q_array: np.ndarray, model_name: str) -> list:
        """Reflectivity of the four spin cross-sections, in refl1d order (mm, mp, pm, pp).

        The list follows `PolarizedNeutronProbe._xs_names`; use
        `POLARIZATION_CHANNEL_TO_INDEX` to pick a channel out of it.
        Results are cached per model state and (q, dq) grid; see `_polarized_cache`.
        """
        # Normalized dtype plus explicit shape in the key: raw bytes alone do not
        # uniquely identify an ndarray (equal bytes can encode different
        # dtype/shape combinations), which could return a wrong-length hit.
        q_array = np.asarray(q_array, dtype=np.float64)
        dq_array = np.asarray(self._resolution_function.smearing(q_array), dtype=np.float64)

        token = self._model_state_token(model_name)
        grid_key = (q_array.shape, q_array.tobytes(), dq_array.shape, dq_array.tobytes())
        cache = self._polarized_cache.get(model_name)
        if cache is not None and cache['token'] == token:
            cached = cache['entries'].get(grid_key)
            if cached is not None:
                return cached
        else:
            cache = {'token': token, 'entries': {}}
            self._polarized_cache[model_name] = cache

        sample = _build_sample(self.storage, model_name)
        polarized_probe = _get_polarized_probe(
            q_array=q_array,
            dq_array=dq_array,
            model_name=model_name,
            storage=self.storage,
            oversampling_factor=OVERSAMPLING_FACTOR,
        )
        polarized_reflectivity = names.Experiment(probe=polarized_probe, sample=sample).reflectivity()

        # returns (q, reflectivity) per cross-section
        reflectivities = [reflectivity for _, reflectivity in polarized_reflectivity]
        if len(reflectivities) != 4:
            raise RuntimeError(f'refl1d returned {len(reflectivities)} polarized cross-sections; expected 4.')
        for channel, index in POLARIZATION_CHANNEL_TO_INDEX.items():
            if len(reflectivities[index]) != len(q_array) or not np.all(np.isfinite(reflectivities[index])):
                raise RuntimeError(f'refl1d returned a malformed {channel.value} cross-section.')
        cache['entries'][grid_key] = reflectivities
        return reflectivities

    def sld_profile(self, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_name : str
            The model name.

        Returns
        -------

            Z and sld(z).
        """
        sample = _build_sample(self.storage, model_name)
        probe = _get_probe(
            q_array=np.array([1]),  # dummy value
            dq_array=np.array([1]),  # dummy value
            model_name=model_name,
            storage=self.storage,
        )
        z, sld, _ = names.Experiment(probe=probe, sample=sample).smooth_profile()
        # -1 to reverse the order
        return z, sld[::-1]

    def magnetic_sld_profile(self, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the nuclear and magnetic scattering length density profiles.

        The magnetic profile is built by smoothing the two in-plane components
        of the moment and converting back, not by smoothing its magnitude and
        angle separately — see :meth:`_smoothed_magnetic_vector`.

        Parameters
        ----------
        model_name : str
            The model name.

        Returns
        -------

            z, sld(z), magnetic sld rhoM(z) and magnetic angle thetaM(z).
        """
        if not self._magnetism:
            raise ValueError(
                'The magnetic sld profile requires magnetism: enable it on this calculator first '
                '(`include_magnetism = True` on the calculator / `magnetism = True` on the wrapper).'
            )
        sample = _build_sample(self.storage, model_name)
        # Plain (non-polarized) probe: unlike `_polarized_reflectivities`, this
        # only renders slabs for `magnetic_smooth_profile()`/`_render_slabs()`,
        # never computes a per-channel reflectivity, so the `theta_offset` that
        # `magnetism=True` would add (needed by `PolarizedQProbe`) is not required.
        probe = _get_probe(
            q_array=np.array([1]),  # dummy value
            dq_array=np.array([1]),  # dummy value
            model_name=model_name,
            storage=self.storage,
        )
        experiment = names.Experiment(probe=probe, sample=sample)
        z, sld, _, _, _ = experiment.magnetic_smooth_profile()
        sld_magnetic, theta_magnetic = self._smoothed_magnetic_vector(experiment, z)
        # -1 to reverse the order
        return z, sld[::-1], sld_magnetic[::-1], theta_magnetic[::-1]

    @staticmethod
    def _smoothed_magnetic_vector(experiment, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def _get_oversampling_q(q_array: np.ndarray, dq_array: np.ndarray, oversampling_factor: int) -> np.ndarray:
    """Get oversampling q."""
    argmin = np.argmin(q_array)  # index of the smallest q element
    argmax = np.argmax(q_array)  # index of the largest q element
    return np.linspace(
        q_array[argmin] - RESOLUTION_PADDING * dq_array[argmin],  # dq element at the smallest q index
        q_array[argmax] + RESOLUTION_PADDING * dq_array[argmax],  # dq element at the largest q index
        oversampling_factor * len(q_array),
    )


def _get_probe(
    q_array: np.ndarray,
    dq_array: np.ndarray,
    model_name: str,
    storage: dict,
    oversampling_factor: int = 1,
    magnetism: bool = False,
) -> names.QProbe:
    """Get probe."""
    probe = names.QProbe(
        Q=q_array,
        dQ=dq_array,
        intensity=storage['model'][model_name]['scale'],
        background=storage['model'][model_name]['bkg'],
    )

    # Add theta_offset attribute if magnetism is enabled
    # This is required for PolarizedQProbe to work correctly: refl1d's
    # `PolarizedNeutronQProbe.__init__` -> `_calculate_union` reads `theta_offset`
    # off each constituent probe, so a QProbe destined for a PolarizedQProbe must
    # carry it even though the plain (unpolarized) QProbe path never touches it.
    if magnetism:
        probe.theta_offset = names.Parameter.default(0, name='theta_offset')

    if oversampling_factor > 1:
        probe.calc_Qo = _get_oversampling_q(q_array, dq_array, oversampling_factor)
    return probe


def _get_polarized_probe(
    q_array: np.ndarray,
    dq_array: np.ndarray,
    model_name: str,
    storage: dict,
    oversampling_factor: int = 1,
) -> names.PolarizedNeutronQProbe:
    """Get polarized probe with all four cross-sections (pp, pm, mp, mm)."""
    four_probes = [
        _get_probe(
            q_array=q_array,
            dq_array=dq_array,
            model_name=model_name,
            storage=storage,
            oversampling_factor=oversampling_factor,
            magnetism=True,  # Enable magnetism for polarized probes
        )
        for _ in range(4)
    ]

    try:
        polarized_probe = names.PolarizedNeutronQProbe(xs=four_probes, name='polarized')
    except AttributeError:
        # refl1d 1.0.0 bug: PolarizedQProbe.__init__ calls _calculate_union(), which
        # reads self._union_cache_key before the attribute is ever assigned (the
        # non-Q PolarizedNeutronProbe assigns it in __init__; the Q variant does
        # not). Pre-seed the attribute and re-run __init__. The try/except makes
        # the workaround self-removing once refl1d fixes the initialization.
        polarized_probe = names.PolarizedNeutronQProbe.__new__(names.PolarizedNeutronQProbe)
        polarized_probe._union_cache_key = None
        polarized_probe.__init__(xs=four_probes, name='polarized')
    return polarized_probe


def _build_sample(storage: dict, model_name: str) -> names.Stack:
    """Build sample."""
    sample = names.Stack()
    # -1 to reverse the order
    for i in storage['model'][model_name]['items'][::-1]:
        if i.repeat.value == 1:
            # -1 to reverse the order
            for j in range(len(i.stack))[::-1]:
                sample |= i.stack[j]
        else:
            stack = names.Stack()
            # -1 to reverse the order
            for j in range(len(i.stack))[::-1]:
                stack |= i.stack[j]
            sample |= Repeat(stack, repeat=i.repeat.value)
    return sample
