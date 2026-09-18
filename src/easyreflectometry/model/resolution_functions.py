# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution functions for the resolution of the experiment.
When a percentage is provided we assume that the resolution is a
Gaussian distribution with a FWHM of the percentage of the q value.
To convert from a sigma value to a FWHM value we use the formula
FWHM = 2.35 * sigma [2 * np.sqrt(2 * np.log(2)) * sigma].

The :meth:`ResolutionFunction.smearing` contract returns **sigma**
(the standard deviation of the Gaussian resolution) for every resolution
type.  This matches the ``sQz`` convention used by data reduction and the
natural output of :class:`Pointwise`.  Each calculation engine wrapper is
responsible for converting sigma to the width convention of its backend
(FWHM for refnx, sigma for refl1d), so that vector resolutions are
interpreted consistently across engines (see GitHub issue #367).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Union

import numpy as np

DEFAULT_RESOLUTION_FWHM_PERCENTAGE = 5.0

# Conversion factor between sigma and FWHM for a Gaussian: FWHM = SIGMA_TO_FWHM * sigma.
SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))


class ResolutionFunction:
    @abstractmethod
    def smearing(self, q: Union[np.array, float]) -> np.array:
        """Return the resolution as sigma (standard deviation) at each ``q``."""
        ...

    @abstractmethod
    def as_dict(self, skip: Optional[List[str]] = None) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> ResolutionFunction:
        """Smearing function."""
        if data['smearing'] == 'PercentageFwhm':
            return PercentageFwhm(data['constant'])
        if data['smearing'] == 'LinearSpline':
            return LinearSpline(data['q_data_points'], data['fwhm_values'])
        if data['smearing'] == 'Pointwise':
            return Pointwise([
                data['q_data_points'],
                data['R_data_points'],
                data['sQz_data_points'],
            ])
        raise ValueError('Unknown resolution function type')


class PercentageFwhm(ResolutionFunction):
    def __init__(self, constant: Union[None, float] = None):
        """Init function."""
        if constant is None:
            constant = DEFAULT_RESOLUTION_FWHM_PERCENTAGE
        self.constant = constant

    def smearing(self, q: Union[np.array, float]) -> np.array:
        """Return per-point sigma values from the constant FWHM percentage.

        ``constant`` is a FWHM percentage of ``q``; it is converted to an
        absolute sigma so the smearing() contract is sigma for all types.
        """
        q_array = np.asarray(q, dtype=float)
        fwhm = (self.constant / 100.0) * q_array
        return fwhm / SIGMA_TO_FWHM

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        """As dict."""
        return {'smearing': 'PercentageFwhm', 'constant': self.constant}


class LinearSpline(ResolutionFunction):
    def __init__(self, q_data_points: np.array, fwhm_values: np.array):
        """Init function."""
        self.q_data_points = q_data_points
        self.fwhm_values = fwhm_values

    def smearing(self, q: Union[np.array, float]) -> np.array:
        """Return per-point sigma values from the FWHM knots.

        The stored ``fwhm_values`` are FWHM widths; they are interpolated
        onto ``q`` and converted to sigma to satisfy the smearing() contract.
        """
        fwhm = np.interp(np.asarray(q, dtype=float), self.q_data_points, self.fwhm_values)
        return fwhm / SIGMA_TO_FWHM

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        """As dict."""
        return {
            'smearing': 'LinearSpline',
            'q_data_points': list(self.q_data_points),
            'fwhm_values': list(self.fwhm_values),
        }


class Pointwise(ResolutionFunction):
    """Pointwise resolution defined by a per-point resolution provided with the data.

    The resolution is supplied as the variance of the Qz values (``sQz``) at the
    measured Qz data points, which is the form produced by data reduction (e.g.
    ``Qz_0.variances``).  The resolution width at each point is ``sqrt(sQz)``.
    For a requested ``q`` the width is obtained by linearly interpolating onto
    ``q``, exactly as :class:`LinearSpline` does for explicitly provided widths.

    This is a convenience wrapper around :class:`LinearSpline` that derives the
    widths from the ``[Qz, R, sQz]`` triple loaded from a data file; the returned
    widths are consumed by the calculators (refnx ``x_err`` / refl1d ``dq``),
    which perform the actual convolution against the model.

    Serialization contract: ``as_dict``/``from_dict`` store ``sQz_data_points``
    as **variances** (sigma squared). This is deliberately unchanged by the
    ORSO ``value_is: FWHM`` support — FWHM columns are converted to sigma at
    load time, so stored values are always sigma squared and saved projects
    round-trip without migration.
    """

    def __init__(self, q_data_points: List[np.ndarray]):
        """Init function.

        Parameters
        ----------
        q_data_points : List[np.ndarray]
            ``[Qz, R, sQz]`` where ``Qz`` are the measured Qz values, ``R`` the
            measured reflectivity (kept only for serialization round-trips) and
            ``sQz`` the variance of ``Qz`` at each point.
        """
        self.q_data_points = q_data_points

    def smearing(self, q: Optional[Union[np.ndarray, float]] = None) -> np.ndarray:
        """Return the resolution sigma interpolated onto ``q``.

        ``sQz`` is the variance of ``Qz``, so the sigma at each data point is
        ``sqrt(sQz)``; values are linearly interpolated onto the requested
        ``q``.  This already satisfies the sigma smearing() contract, so no
        FWHM conversion is applied.  When ``q`` is ``None`` the sigma values
        are returned at the stored data points.
        """
        Qz = np.asarray(self.q_data_points[0], dtype=float)
        sQz = np.asarray(self.q_data_points[2], dtype=float)
        q_eval = Qz if q is None else np.asarray(q, dtype=float)
        widths = np.sqrt(sQz)
        return np.asarray(np.interp(q_eval, Qz, widths))

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        """As dict."""
        return {
            'smearing': 'Pointwise',
            'q_data_points': list(self.q_data_points[0]),
            'R_data_points': list(self.q_data_points[1]),
            'sQz_data_points': list(self.q_data_points[2]),
        }
