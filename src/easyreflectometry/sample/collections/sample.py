# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import List
from typing import Optional

import numpy as np

from ..assemblies.base_assembly import BaseAssembly
from ..assemblies.multilayer import Multilayer
from ..assemblies.repeating_multilayer import RepeatingMultilayer
from ..assemblies.surfactant_layer import SurfactantLayer
from ..elements.layers.layer import Layer
from ..volume_fraction import VolumeFractionProfile
from ..volume_fraction import volume_fraction_profile
from .base_collection import BaseCollection


# Needs to be a function, elements are added to the global_object.map
def DEFAULT_ELEMENTS(interface):
    """:meta private:."""
    return (
        Multilayer(interface=interface),
        Multilayer(interface=interface),
    )


class Sample(BaseCollection):
    """A sample is a collection of assemblies that represent the structure for which experimental measurements exist."""

    def __init__(
        self,
        *assemblies: Optional[List[BaseAssembly]],
        name: str = 'EasySample',
        interface=None,
        unique_name: Optional[str] = None,
        populate_if_none: bool = True,
        **kwargs,
    ):
        """Constructor.

        Parameters
        ----------
        **kwargs :
        populate_if_none : bool, optional
            By default, True.
        unique_name : Optional[str], optional
            By default, None.
        *assemblies : Optional[List[BaseAssembly]]
        args :
            The assemblies in the sample.
        name : str, optional
            Name of the sample. By default, 'EasySample'.
        interface :
            Calculator interface. By default, None.
        """
        # `from_dict` (via `EasyList.from_dict`) passes the items as a single
        # list-positional arg; unpack that so validation and super() agree.
        if len(assemblies) == 1 and isinstance(assemblies[0], list):
            assemblies = tuple(assemblies[0])

        if not assemblies:
            if populate_if_none:
                assemblies = DEFAULT_ELEMENTS(interface)
            else:
                assemblies = []

        for assembly in assemblies:
            if not issubclass(type(assembly), BaseAssembly):
                raise ValueError('The elements must be an Assembly.')
        super().__init__(
            name,
            interface,
            *assemblies,
            unique_name=unique_name,
            populate_if_none=populate_if_none,
            **kwargs,
        )

    def add_assembly(self, assembly: Optional[BaseAssembly] = None):
        """Add an assembly to the sample.

        Parameters
        ----------
        assembly : Optional[BaseAssembly], optional
            Assembly to add. By default, None.
        """
        if assembly is None:
            assembly = Multilayer(
                name='EasyMultilayer added',
                interface=self.interface,
            )
        self.append(assembly)

    def duplicate_assembly(self, index: int):
        """Add an assembly to the sample.

        Parameters
        ----------
        index : int
        assembly :
            Assembly to add.
        """
        # Order matters: RepeatingMultilayer and SurfactantLayer are subclasses of
        # BaseAssembly but not Multilayer; however a RepeatingMultilayer IS a
        # Multilayer, so the most-specific check must come first to avoid
        # serialising it through the wrong `from_dict`.
        to_be_duplicated = self[index]
        if isinstance(to_be_duplicated, RepeatingMultilayer):
            duplicate = RepeatingMultilayer.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        elif isinstance(to_be_duplicated, SurfactantLayer):
            duplicate = SurfactantLayer.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        elif isinstance(to_be_duplicated, Multilayer):
            duplicate = Multilayer.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        else:
            raise TypeError(f'Cannot duplicate assembly of type {type(to_be_duplicated).__name__}')
        duplicate.name = duplicate.name + ' duplicate'
        self.append(duplicate)

    def move_up(self, index: int):
        """Move the assembly at the given index up in the sample.

        Parameters
        ----------
        index : int
            Index of the assembly to move up.
        """
        super().move_up(index)

    def move_down(self, index: int):
        """Move the assembly at the given index down in the sample.

        Parameters
        ----------
        index : int
            Index of the assembly to move down.
        """
        super().move_down(index)

    def remove_assembly(self, index: int):
        """Remove the assembly at the given index from the sample.

        Parameters
        ----------
        index : int
            Index of the assembly to remove.
        """
        self.pop(index)

    @property
    def superphase(self) -> Layer:
        """The superphase of the sample."""
        return self[0].front_layer

    @property
    def subphase(self) -> Layer:
        """The subphase of the sample."""
        # This assembly only got one layer
        if self[-1].back_layer is None:
            return self[-1].front_layer
        else:
            return self[-1].back_layer

    def volume_fraction_profile(
        self,
        z: Optional[np.ndarray] = None,
        *,
        max_delta_z: Optional[float] = None,
        merge_equivalent: bool = True,
    ) -> VolumeFractionProfile:
        """Volume fraction (occupancy) of every component of the sample versus depth.

        A thin wrapper around :func:`easyreflectometry.sample.volume_fraction.volume_fraction_profile`;
        see there for the parameters, the depth convention and the caveats.

        Parameters
        ----------
        z : np.ndarray, optional
            Depths (Å) to evaluate at; ``z = 0`` is the superphase / first layer interface.
            By default a 500-point grid spanning the sample.
        max_delta_z : float, optional
            Maximum spacing of the default grid. Cannot be combined with ``z``.
        merge_equivalent : bool, optional
            Merge materials with equal name and SLD into one component. By default, True.

        Returns
        -------
        VolumeFractionProfile
            The per-component profiles, summing to one at every depth.
        """
        return volume_fraction_profile(self, z, max_delta_z=max_delta_z, merge_equivalent=merge_equivalent)
