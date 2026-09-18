# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Polarized experiment data: per-spin-channel datasets and channel detection.

A polarized neutron reflectometry experiment measures up to four spin
cross-sections (non-spin-flip 'pp'/'mm', spin-flip 'pm'/'mp'), typically stored
as one file per channel. `PolarizedDataSet` groups those per-channel
`DataSet1D` objects into a single experiment sharing one model.
"""

from __future__ import annotations

import os
import re
from types import MappingProxyType
from typing import Mapping
from typing import Optional
from typing import Union

from easyreflectometry.calculators.polarization import POLARIZATION_CHANNEL_TO_INDEX
from easyreflectometry.calculators.polarization import PolarizationChannel

from .data_store import DataSet1D

# ORSO instrument_settings.polarization values → spin channel.
# Only the four fully-analysed cross-sections are mapped. Partially-analysed
# observables are deliberately NOT: 'po' (incident plus, no outgoing analysis)
# measures the sum pp + pm — a physically different observable from the pp
# cross-section — and likewise 'mo' = mp + mm; 'op'/'om' analyse only the
# outgoing spin; 'unpolarized' is no channel at all. A header declaring any of
# those suppresses the filename fallback and the user must assign (or the data
# must be modelled as a channel sum, which this formalism does not do yet).
_ORSO_POLARIZATION_TO_CHANNEL = {
    'pp': PolarizationChannel.PP,
    'pm': PolarizationChannel.PM,
    'mp': PolarizationChannel.MP,
    'mm': PolarizationChannel.MM,
}

# Filename tokens → spin channel (used when there is no ORSO header).
_TOKEN_TO_CHANNEL = {
    'pp': PolarizationChannel.PP,
    'uu': PolarizationChannel.PP,
    'upup': PolarizationChannel.PP,
    'plusplus': PolarizationChannel.PP,
    'mm': PolarizationChannel.MM,
    'dd': PolarizationChannel.MM,
    'downdown': PolarizationChannel.MM,
    'minusminus': PolarizationChannel.MM,
    'pm': PolarizationChannel.PM,
    'ud': PolarizationChannel.PM,
    'updown': PolarizationChannel.PM,
    'mp': PolarizationChannel.MP,
    'du': PolarizationChannel.MP,
    'downup': PolarizationChannel.MP,
}

# Single tokens for 2-channel (NSF-only) naming like 'sample_up.dat' / 'sample_down.dat'.
_SINGLE_TOKEN_TO_CHANNEL = {
    'u': PolarizationChannel.PP,
    'up': PolarizationChannel.PP,
    'p': PolarizationChannel.PP,
    'plus': PolarizationChannel.PP,
    'd': PolarizationChannel.MM,
    'down': PolarizationChannel.MM,
    'm': PolarizationChannel.MM,
    'minus': PolarizationChannel.MM,
}

# Adjacent token pairs like 'up_down' → channel.
_TOKEN_PAIR_TO_CHANNEL = {
    ('up', 'up'): PolarizationChannel.PP,
    ('up', 'down'): PolarizationChannel.PM,
    ('down', 'up'): PolarizationChannel.MP,
    ('down', 'down'): PolarizationChannel.MM,
    ('u', 'u'): PolarizationChannel.PP,
    ('u', 'd'): PolarizationChannel.PM,
    ('d', 'u'): PolarizationChannel.MP,
    ('d', 'd'): PolarizationChannel.MM,
    ('plus', 'plus'): PolarizationChannel.PP,
    ('plus', 'minus'): PolarizationChannel.PM,
    ('minus', 'plus'): PolarizationChannel.MP,
    ('minus', 'minus'): PolarizationChannel.MM,
}


class PolarizedDataSet:
    """A polarized experiment: one dataset per measured spin channel, one shared model.

    Parameters
    ----------
    name : str, optional
        Name of the experiment. By default, 'PolarizedSeries'.
    channels : dict[PolarizationChannel | str, DataSet1D]
        The measured channels; at least one. Keys are 'pp', 'pm', 'mp', 'mm'
        (or the corresponding enum members). Two-channel NSF-only experiments
        simply provide 'pp' and 'mm'.
    model : Model, optional
        The model shared by all channels. By default, None.
    """

    def __init__(
        self,
        name: str = 'PolarizedSeries',
        channels: Optional[dict[Union[PolarizationChannel, str], DataSet1D]] = None,
        model=None,
    ):
        if not channels:
            raise ValueError('A PolarizedDataSet requires at least one channel dataset.')
        normalized: dict[PolarizationChannel, DataSet1D] = {}
        for channel, dataset in channels.items():
            channel = PolarizationChannel(channel)
            if channel in normalized:
                raise ValueError(f"Duplicate channel '{channel.value}'.")
            normalized[channel] = self._validated_dataset(channel, dataset)
        self._channels = self._in_canonical_order(normalized)
        self.name = name
        self.model = model

    @staticmethod
    def _validated_dataset(channel: PolarizationChannel, dataset: DataSet1D) -> DataSet1D:
        if not isinstance(dataset, DataSet1D):
            raise ValueError(f"Channel '{channel.value}' must be a DataSet1D, got {type(dataset).__name__}.")
        return dataset

    @staticmethod
    def _in_canonical_order(channels: dict[PolarizationChannel, DataSet1D]) -> dict[PolarizationChannel, DataSet1D]:
        # Canonical channel order: pp, pm, mp, mm.
        return {channel: channels[channel] for channel in POLARIZATION_CHANNEL_TO_INDEX if channel in channels}

    @property
    def model(self):
        """The model shared by all channels."""
        return self._model

    @model.setter
    def model(self, new_model) -> None:
        self._model = new_model
        for dataset in self._channels.values():
            dataset.model = new_model

    @property
    def channels(self) -> Mapping[PolarizationChannel, DataSet1D]:
        """The measured channels, in canonical order (pp, pm, mp, mm).

        Read-only view: use :meth:`set_channel` / :meth:`remove_channel` to
        modify the channel set, so validation and model propagation apply.
        """
        return MappingProxyType(self._channels)

    @property
    def available_channels(self) -> list[PolarizationChannel]:
        """The measured channels, in canonical order (pp, pm, mp, mm)."""
        return list(self._channels.keys())

    def set_channel(self, channel: Union[PolarizationChannel, str], dataset: DataSet1D) -> None:
        """Add or replace one channel dataset.

        The dataset is validated, adopts the shared model, and the canonical
        channel order is preserved.

        Parameters
        ----------
        channel : Union[PolarizationChannel, str]
            One of 'pp', 'pm', 'mp', 'mm' (or the corresponding enum member).
        dataset : DataSet1D
            The channel data.
        """
        channel = PolarizationChannel(channel)
        dataset = self._validated_dataset(channel, dataset)
        dataset.model = self._model
        merged = dict(self._channels)
        merged[channel] = dataset
        self._channels = self._in_canonical_order(merged)

    def remove_channel(self, channel: Union[PolarizationChannel, str]) -> None:
        """Remove one channel dataset; the last remaining channel cannot be removed.

        Parameters
        ----------
        channel : Union[PolarizationChannel, str]
            One of 'pp', 'pm', 'mp', 'mm' (or the corresponding enum member).
        """
        channel = PolarizationChannel(channel)
        if channel not in self._channels:
            raise ValueError(f"No '{channel.value}' channel in this dataset.")
        if len(self._channels) == 1:
            raise ValueError('A PolarizedDataSet requires at least one channel dataset.')
        del self._channels[channel]

    def __getitem__(self, channel: Union[PolarizationChannel, str]) -> DataSet1D:
        return self._channels[PolarizationChannel(channel)]

    def __contains__(self, channel: Union[PolarizationChannel, str]) -> bool:
        try:
            return PolarizationChannel(channel) in self._channels
        except ValueError:
            return False

    def __len__(self) -> int:
        return len(self._channels)

    @property
    def is_experiment(self) -> bool:
        """Is experiment."""
        return self._model is not None

    @property
    def is_simulation(self) -> bool:
        """Is simulation."""
        return self._model is None

    def __repr__(self) -> str:
        channel_names = ', '.join(channel.value for channel in self._channels)
        return f"Polarized dataset '{self.name}' with channels: {channel_names}"


def detect_polarization_channel(path: str) -> Optional[PolarizationChannel]:
    """Detect the spin channel of a data file.

    Tries the ORSO header (`data_source.measurement.instrument_settings.polarization`)
    first, then filename heuristics ('_uu'/'_pp'/'_up' → pp, '_dd'/'_mm'/'_down' → mm,
    '_ud'/'_pm' → pm, '_du'/'_mp' → mp, ...). Returns None when neither yields a
    channel; a GUI should then ask the user to assign one.

    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    Optional[PolarizationChannel]
        The detected channel, or None.
    """
    header_declared, channel = _channel_from_orso_header(path)
    if channel is not None:
        return channel
    if header_declared:
        # The header explicitly declares a non-channel polarization (e.g.
        # 'unpolarized'): trust it over any channel-looking filename tokens.
        return None
    return _channel_from_filename(path)


def channel_from_orso_polarization(polarization) -> Optional[PolarizationChannel]:
    """Map an ORSO ``instrument_settings.polarization`` value to a spin channel.

    Parameters
    ----------
    polarization :
        The header value (orsopy ``Polarization`` enum, string, or None).

    Returns
    -------
    Optional[PolarizationChannel]
        The mapped channel, or None for absent/unmapped values (``po``, ``mo``,
        ``op``, ``om``, ``unpolarized``, ``vector`` are deliberately unmapped).
    """
    if polarization is None:
        return None
    value = getattr(polarization, 'value', polarization)
    return _ORSO_POLARIZATION_TO_CHANNEL.get(str(value).lower())


def _dataset_polarization(orso_dataset):
    """The declared polarization of one parsed ORSO dataset, or None."""
    try:
        return orso_dataset.info.data_source.measurement.instrument_settings.polarization
    except AttributeError:
        return None


def detect_polarization_channels_per_dataset(
    orso_data,
) -> list[tuple[bool, Optional[PolarizationChannel]]]:
    """Classify every dataset of a parsed ORSO file by its own header.

    Unlike :func:`detect_polarization_channel`, which reads only the first
    dataset, this honours per-dataset ``polarization:`` overrides in
    multi-dataset files.

    Parameters
    ----------
    orso_data : list
        Parsed ORSO dataset list (as returned by ``orso.load_orso``).

    Returns
    -------
    list[tuple[bool, Optional[PolarizationChannel]]]
        Per dataset: (header declares a polarization, mapped channel or None).
    """
    result = []
    for orso_dataset in orso_data:
        polarization = _dataset_polarization(orso_dataset)
        result.append((polarization is not None, channel_from_orso_polarization(polarization)))
    return result


def _channel_from_orso_header(path: str) -> tuple[bool, Optional[PolarizationChannel]]:
    """Read the polarization of the first dataset in an ORSO file.

    Returns
    -------
    tuple[bool, Optional[PolarizationChannel]]
        (header declares a polarization, mapped channel or None). The flag is
        False when the file is unreadable or carries no polarization field.
    """
    try:
        from easyreflectometry.orso_utils import _load_orso_any

        orso_data = _load_orso_any(str(path))
        polarization = orso_data[0].info.data_source.measurement.instrument_settings.polarization
    except Exception:
        return False, None
    if polarization is None:
        return False, None
    return True, channel_from_orso_polarization(polarization)


def _channel_from_filename(path: str) -> Optional[PolarizationChannel]:
    """Guess the spin channel from separator-delimited tokens in the file name."""
    basename = os.path.splitext(os.path.basename(str(path)))[0].lower()
    tokens = [token for token in re.split(r'[^a-z0-9]+', basename) if token]

    # Prefer the most specific match, scanning from the end (suffix convention).
    for token in reversed(tokens):
        if token in _TOKEN_TO_CHANNEL:
            return _TOKEN_TO_CHANNEL[token]
    for first, second in reversed(list(zip(tokens, tokens[1:]))):
        if (first, second) in _TOKEN_PAIR_TO_CHANNEL:
            return _TOKEN_PAIR_TO_CHANNEL[(first, second)]
    for token in reversed(tokens):
        if token in _SINGLE_TOKEN_TO_CHANNEL:
            return _SINGLE_TOKEN_TO_CHANNEL[token]
    return None
