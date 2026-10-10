# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Optional

from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.constraints import constrain_equal

from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer_area_per_molecule import LayerAreaPerMolecule
from ..elements.materials.material import Material
from .base_assembly import BaseAssembly


class SurfactantLayer(BaseAssembly):
    """A surfactant layer constructs a series of layers representing the
    head and tail groups of a surfactant.
    This assembly allows the definition of a surfactant or lipid using the chemistry
    of the head (head_layer) and tail (tail_layer) regions, additionally
    this approach will make the application of constraints such as conformal roughness
    or area per molecule more straight forward.

    More information about the usage of this assembly is available in the
    `surfactant documentation`_

    .. _`surfactant documentation`: ../sample/assemblies_library.html#surfactantlayer
    """

    def __init__(
        self,
        tail_layer: Optional[LayerAreaPerMolecule] = None,
        head_layer: Optional[LayerAreaPerMolecule] = None,
        name: str = 'EasySurfactantLayer',
        unique_name: Optional[str] = None,
        constrain_area_per_molecule: bool = False,
        conformal_roughness: bool = False,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        tail_layer : Optional[LayerAreaPerMolecule], optional
            Layer representing the tail part of the surfactant layer. By default, None.
        head_layer : Optional[LayerAreaPerMolecule], optional
            Layer representing the head part of the surfactant layer. By default, None.
        name : str, optional
            Name for surfactant layer. By default, 'EasySurfactantLayer'.
        constrain_area_per_molecule : bool, optional
            Constrain the area per molecule. By default, False.
        conformal_roughness : bool, optional
            Constrain the roughness to be the same for both layers. By default, False.
        interface :
            Calculator interface. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if tail_layer is None:
            air = Material(
                sld=0,
                isld=0,
                name='Air',
                unique_name=unique_name + '_MaterialTail',
                interface=interface,
            )
            tail_layer = LayerAreaPerMolecule(
                molecular_formula='C32D64',
                thickness=16,
                solvent=air,
                solvent_fraction=0.0,
                area_per_molecule=48.2,
                roughness=3,
                name='DPPC Tail',
                unique_name=unique_name + '_LayerAreaPerMoleculeTail',
                interface=interface,
            )
        if head_layer is None:
            d2o = Material(
                sld=6.36,
                isld=0,
                name='D2O',
                unique_name=unique_name + '_MaterialHead',
                interface=interface,
            )
            head_layer = LayerAreaPerMolecule(
                molecular_formula='C10H18NO8P',
                thickness=10.0,
                solvent=d2o,
                solvent_fraction=0.2,
                area_per_molecule=48.2,
                roughness=3.0,
                name='DPPC Head',
                unique_name=unique_name + '_LayerAreaPerMoleculeHead',
                interface=interface,
            )
        surfactant = LayerCollection(
            tail_layer,
            head_layer,
            name='Layers',
            unique_name=unique_name + '_LayerCollection',
            interface=interface,
        )
        super().__init__(
            name=name,
            unique_name=unique_name,
            type='Surfactant Layer',
            layers=surfactant,
            interface=interface,
        )

        self.conformal = False

        if constrain_area_per_molecule:
            self.constrain_area_per_molecule = True
        if conformal_roughness:
            self._enable_roughness_constraints()
            self.conformal = True

    @property
    def tail_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the tail layer of the surfactant surface."""
        return self.front_layer

    @tail_layer.setter
    def tail_layer(self, layer: LayerAreaPerMolecule) -> None:
        """Set the tail layer of the surfactant surface."""
        self.front_layer = layer

    @property
    def head_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the head layer of the surfactant surface."""
        return self.back_layer

    @head_layer.setter
    def head_layer(self, layer: LayerAreaPerMolecule) -> None:
        """Set the head layer of the surfactant surface."""
        self.back_layer = layer

    @property
    def constrain_area_per_molecule(self) -> bool:
        """Get the area per molecule constraint status."""
        constrained = not self.head_layer._area_per_molecule.independent
        return constrained

    @constrain_area_per_molecule.setter
    def constrain_area_per_molecule(self, status: bool):
        """Set the status for the area per molecule constraint such that the head and tail layers have the
        same area per molecule.

        Parameters
        ----------
        status : bool
            Boolean description the wanted of the constraint.
        """
        if status:
            independent_param = self.tail_layer._area_per_molecule
            self.head_layer._area_per_molecule.make_dependent_on(
                dependency_expression='a', dependency_map={'a': independent_param}
            )
        else:
            self.head_layer._area_per_molecule.make_independent()
        return

    @property
    def conformal_roughness(self) -> bool:
        """Get the roughness constraint status."""
        return self.conformal

    @conformal_roughness.setter
    def conformal_roughness(self, status: bool):
        """Set the status for the roughness to be the same for both layers.

        Parameters
        ----------
        status : bool
            Boolean description the wanted of the constraint.
        """
        if status:
            self._enable_roughness_constraints()
            self.conformal = True
        else:
            self._disable_roughness_constraints()
            self.conformal = False

    def constrain_solvent_roughness(self, solvent_roughness: Parameter):
        """Add the constraint to the solvent roughness.

        Parameters
        ----------
        solvent_roughness : Parameter
            The solvent roughness parameter.
        """
        if not self.conformal_roughness:
            raise ValueError('Roughness must be conformal to use this function.')
        solvent_roughness.value = self.tail_layer.roughness.value
        solvent_roughness.make_dependent_on(dependency_expression='a', dependency_map={'a': self.tail_layer.roughness})

    def constrain_multiple_contrast(
        self,
        another_contrast: SurfactantLayer,
        head_layer_thickness: bool = True,
        tail_layer_thickness: bool = True,
        head_layer_area_per_molecule: bool = True,
        tail_layer_area_per_molecule: bool = True,
        head_layer_fraction: bool = True,
        tail_layer_fraction: bool = True,
    ):
        """Constrain structural parameters between surfactant layer objects.

        Parameters
        ----------
        tail_layer_fraction : bool, optional
            By default, True.
        head_layer_fraction : bool, optional
            By default, True.
        tail_layer_area_per_molecule : bool, optional
            By default, True.
        head_layer_area_per_molecule : bool, optional
            By default, True.
        tail_layer_thickness : bool, optional
            By default, True.
        head_layer_thickness : bool, optional
            By default, True.
        another_contrast : SurfactantLayer
            The surfactant layer to constrain.
        """
        if head_layer_thickness:
            constrain_equal(self.head_layer.thickness, another_contrast.head_layer.thickness)

        if tail_layer_thickness:
            constrain_equal(self.tail_layer.thickness, another_contrast.tail_layer.thickness)

        if head_layer_area_per_molecule:
            constrain_equal(self.head_layer._area_per_molecule, another_contrast.head_layer._area_per_molecule)

        if tail_layer_area_per_molecule:
            constrain_equal(self.tail_layer._area_per_molecule, another_contrast.tail_layer._area_per_molecule)

        if head_layer_fraction:
            constrain_equal(self.head_layer.material._fraction, another_contrast.head_layer.material._fraction)

        if tail_layer_fraction:
            constrain_equal(self.tail_layer.material._fraction, another_contrast.tail_layer.material._fraction)

    @property
    def _dict_repr(self) -> dict:
        """A simplified dict representation."""
        return {
            self.name: {
                'head_layer': self.head_layer._dict_repr,
                'tail_layer': self.tail_layer._dict_repr,
                'area per molecule constrained': self.constrain_area_per_molecule,
                'conformal roughness': self.conformal_roughness,
            }
        }

    def to_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Serialize, dropping the derived ``layers`` field (it is rebuilt
        from ``tail_layer`` and ``head_layer`` in ``__init__``).
        """
        this_dict = super().to_dict(skip=skip)
        this_dict.pop('layers', None)
        return this_dict

    def as_dict(self, skip: Optional[list[str]] = None) -> dict:
        return self.to_dict(skip=skip)
