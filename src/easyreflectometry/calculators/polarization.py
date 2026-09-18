# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum


class PolarizationChannel(str, Enum):
    """Spin cross-section channels for polarized neutron reflectometry.

    The accepted spellings are exactly the enum values ('pp', 'pm', 'mp', 'mm');
    uppercase strings are rejected.
    """

    PP = 'pp'  # non-spin-flip, up-up
    PM = 'pm'  # spin-flip, up-down
    MP = 'mp'  # spin-flip, down-up
    MM = 'mm'  # non-spin-flip, down-down


# Mapping to refl1d cross-section indices. refl1d returns the polarized
# cross-sections in the order of `PolarizedNeutronProbe._xs_names`, which is
# ['mm', 'mp', 'pm', 'pp'] (refl1d/probe/probe.py); `magnetic_amplitude` returns
# (--, -+, +-, ++) accordingly (refl1d/sample/reflectivity.py). The pp/mm
# assignment is pinned by a physics test (moment aligned with the guide field:
# pp must see rho + rhoM); pm vs mp rests on the refl1d source alone (they are
# identical by symmetry for non-chiral, non-absorptive samples).
# The key order (pp, pm, mp, mm) is the canonical channel order used throughout.
POLARIZATION_CHANNEL_TO_INDEX = {
    PolarizationChannel.PP: 3,
    PolarizationChannel.PM: 2,
    PolarizationChannel.MP: 1,
    PolarizationChannel.MM: 0,
}
