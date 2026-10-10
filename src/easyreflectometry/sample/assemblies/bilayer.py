# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.constraints import constrain_equal

from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer_area_per_molecule import LayerAreaPerMolecule
from ..elements.materials.material import Material
from .base_assembly import BaseAssembly

DEFAULTS = {
    'head': {
        'molecular_formula': 'C10H18NO8P',
        'thickness': 10.0,
        'solvent_fraction': 0.2,
        'area_per_molecule': 48.2,
        'roughness': 3.0,
    },
    'tail': {
        'molecular_formula': 'C32D64',
        'thickness': 16.0,
        'solvent_fraction': 0.0,
        'area_per_molecule': 48.2,
        'roughness': 3.0,
    },
    'solvent': {
        'sld': 6.36,
        'isld': 0,
        'name': 'D2O',
    },
}


class Bilayer(BaseAssembly):
    """A lipid bilayer consisting of two surfactant layers where one is inverted.

    The bilayer structure is: Front Head - Front Tail - Back Tail - Back Head

    This assembly comes pre-populated with physically meaningful constraints:
    - Both tail layers are constrained to share the same structural parameters
      (thickness, area per molecule, and solvent fraction).
    - Head layers are constrained to share thickness and area per molecule,
      while solvent fraction (hydration) remains independent on each side.
    - A single roughness parameter applies to all interfaces (conformal roughness).

    More information about the usage of this assembly is available in the
    `bilayer documentation`_

    .. _`bilayer documentation`: ../sample/assemblies_library.html#bilayer
    """

    def __init__(
        self,
        front_head_layer: LayerAreaPerMolecule | None = None,
        front_tail_layer: LayerAreaPerMolecule | None = None,
        back_head_layer: LayerAreaPerMolecule | None = None,
        back_tail_layer: LayerAreaPerMolecule | None = None,
        name: str = 'EasyBilayer',
        unique_name: str | None = None,
        constrain_heads: bool = True,
        conformal_roughness: bool = True,
        interface: Any = None,
    ):
        """Constructor.

        Parameters
        ----------
        front_head_layer : LayerAreaPerMolecule | None, optional
            Layer representing the front head part of the bilayer. By default, None.
        front_tail_layer : LayerAreaPerMolecule | None, optional
            Layer representing the front tail part of the bilayer.
            The back tail layer's thickness, area per molecule, and solvent fraction are
            constrained to match this layer. By default, None.
        back_head_layer : LayerAreaPerMolecule | None, optional
            Layer representing the back head part of the bilayer. By default, None.
        back_tail_layer : LayerAreaPerMolecule | None, optional
            Layer representing the back tail part of the bilayer. If omitted, a back tail
            is created from the front tail (same molecular_formula, solvent, roughness, etc.).
            Independent state (solvent, molecular_formula, name, roughness when
            ``conformal_roughness`` is False) is preserved across serialization; the
            structural parameters listed above are derived from the front tail. By default, None.
        name : str, optional
            Name for bilayer. By default, 'EasyBilayer'.
        unique_name : str | None, optional
            Unique name for internal object tracking. By default, None.
        constrain_heads : bool, optional
            When `True`, the back head layer thickness and area per
            molecule are constrained to match the front head layer. Solvent fraction
            (hydration) remains independent on each side. By default, True.
        conformal_roughness : bool, optional
            When `True`, all four layer interfaces share
            the same roughness value, controlled by the front head layer. By default, True.
        interface : Any, optional
            Calculator interface. By default, None.
        """
        # Generate unique name for nested objects
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        # Create default layers if not provided
        if front_head_layer is None:
            front_head_layer = self._create_default_head_layer(
                unique_name=unique_name,
                name_suffix='Front',
                interface=interface,
            )

        if front_tail_layer is None:
            front_tail_layer = self._create_default_tail_layer(
                unique_name=unique_name,
                interface=interface,
            )

        # If no back tail is supplied, derive one from the front tail. The structural
        # parameters (thickness, area_per_molecule, solvent_fraction) get constrained to
        # the front tail in `_setup_tail_constraints` below regardless of which path
        # produced this layer.
        if back_tail_layer is None:
            back_tail_layer = self._create_back_tail_layer(
                front_tail_layer=front_tail_layer,
                unique_name=unique_name,
                interface=interface,
            )

        if back_head_layer is None:
            back_head_layer = self._create_default_head_layer(
                unique_name=unique_name,
                name_suffix='Back',
                interface=interface,
            )

        # Create layer collection: front_head, front_tail, back_tail, back_head
        bilayer_layers = LayerCollection(
            front_head_layer,
            front_tail_layer,
            back_tail_layer,
            back_head_layer,
            name='Layers',
            unique_name=unique_name + '_LayerCollection',
            interface=interface,
        )

        super().__init__(
            name=name,
            unique_name=unique_name,
            type='Bilayer',
            layers=bilayer_layers,
            interface=interface,
        )

        self._conformal_roughness = False
        self._constrain_heads = False
        self._tail_constraints_setup = False

        # Setup tail layer constraints (back tail depends on front tail)
        self._setup_tail_constraints()

        # Apply head constraints if requested
        if constrain_heads:
            self.constrain_heads = True

        # Apply conformal roughness if requested
        if conformal_roughness:
            self.conformal_roughness = True

    @staticmethod
    def _create_default_head_layer(
        unique_name: str,
        name_suffix: str,
        interface: Any = None,
    ) -> LayerAreaPerMolecule:
        """Create a default head layer with DPPC head group parameters.

        Parameters
        ----------
        unique_name : str
            Base unique name for internal object tracking.
        name_suffix : str
            Suffix for layer name ('Front' or 'Back').
        interface : Any, optional
            Calculator interface. By default, None.

        Returns
        -------
        LayerAreaPerMolecule
            A new LayerAreaPerMolecule for the head group.
        """
        solvent = Material(
            sld=DEFAULTS['solvent']['sld'],
            isld=DEFAULTS['solvent']['isld'],
            name=DEFAULTS['solvent']['name'],
            unique_name=unique_name + f'_Material{name_suffix}Head',
            interface=interface,
        )
        return LayerAreaPerMolecule(
            molecular_formula=DEFAULTS['head']['molecular_formula'],
            thickness=DEFAULTS['head']['thickness'],
            solvent=solvent,
            solvent_fraction=DEFAULTS['head']['solvent_fraction'],
            area_per_molecule=DEFAULTS['head']['area_per_molecule'],
            roughness=DEFAULTS['head']['roughness'],
            name=f'DPPC Head {name_suffix}',
            unique_name=unique_name + f'_LayerAreaPerMolecule{name_suffix}Head',
            interface=interface,
        )

    @staticmethod
    def _create_default_tail_layer(
        unique_name: str,
        interface: Any = None,
    ) -> LayerAreaPerMolecule:
        """Create a default tail layer with DPPC tail group parameters.

        Parameters
        ----------
        unique_name : str
            Base unique name for internal object tracking.
        interface : Any, optional
            Calculator interface. By default, None.

        Returns
        -------
        LayerAreaPerMolecule
            A new LayerAreaPerMolecule for the tail group.
        """
        solvent = Material(
            sld=DEFAULTS['solvent']['sld'],
            isld=DEFAULTS['solvent']['isld'],
            name=DEFAULTS['solvent']['name'],
            unique_name=unique_name + '_MaterialTail',
            interface=interface,
        )
        return LayerAreaPerMolecule(
            molecular_formula=DEFAULTS['tail']['molecular_formula'],
            thickness=DEFAULTS['tail']['thickness'],
            solvent=solvent,
            solvent_fraction=DEFAULTS['tail']['solvent_fraction'],
            area_per_molecule=DEFAULTS['tail']['area_per_molecule'],
            roughness=DEFAULTS['tail']['roughness'],
            name='DPPC Tail',
            unique_name=unique_name + '_LayerAreaPerMoleculeTail',
            interface=interface,
        )

    @staticmethod
    def _create_back_tail_layer(
        front_tail_layer: LayerAreaPerMolecule,
        unique_name: str,
        interface: Any = None,
    ) -> LayerAreaPerMolecule:
        """Create a back tail layer with initial values copied from the front tail layer.

        Parameters
        ----------
        front_tail_layer : LayerAreaPerMolecule
            The front tail layer to copy initial values from.
        unique_name : str
            Base unique name for internal object tracking.
        interface : Any, optional
            Calculator interface. By default, None.

        Returns
        -------
        LayerAreaPerMolecule
            A new LayerAreaPerMolecule for the back tail.
        """
        solvent = Material(
            sld=DEFAULTS['solvent']['sld'],
            isld=DEFAULTS['solvent']['isld'],
            name=DEFAULTS['solvent']['name'],
            unique_name=unique_name + '_MaterialBackTail',
            interface=interface,
        )
        return LayerAreaPerMolecule(
            molecular_formula=front_tail_layer.molecular_formula,
            thickness=front_tail_layer.thickness.value,
            solvent=solvent,
            solvent_fraction=front_tail_layer.solvent_fraction.value,
            area_per_molecule=front_tail_layer.area_per_molecule.value,
            roughness=front_tail_layer.roughness.value,
            name=front_tail_layer.name + ' Back',
            unique_name=unique_name + '_LayerAreaPerMoleculeBackTail',
            interface=interface,
        )

    def _setup_tail_constraints(self) -> None:
        """Setup constraints so back tail layer parameters depend on front tail layer.

        Constrains thickness, area per molecule, and solvent fraction of the
        back tail layer to match the front tail layer.
        """
        front_tail = self.front_tail_layer
        back_tail = self.back_tail_layer

        # Constrain thickness
        back_tail.thickness.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_tail.thickness},
        )

        # Constrain area per molecule
        back_tail.area_per_molecule_parameter.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_tail.area_per_molecule_parameter},
        )

        # Constrain solvent fraction
        back_tail.solvent_fraction_parameter.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_tail.solvent_fraction_parameter},
        )

        self._tail_constraints_setup = True

    @property
    def front_head_layer(self) -> LayerAreaPerMolecule:
        """Get the front head layer of the bilayer."""
        return self.layers[0]

    @front_head_layer.setter
    def front_head_layer(self, layer: LayerAreaPerMolecule) -> None:
        """Set the front head layer of the bilayer."""
        self.layers[0] = layer

    @property
    def front_tail_layer(self) -> LayerAreaPerMolecule:
        """Get the front tail layer of the bilayer."""
        return self.layers[1]

    @property
    def back_tail_layer(self) -> LayerAreaPerMolecule:
        """Get the back tail layer of the bilayer."""
        return self.layers[2]

    @property
    def back_head_layer(self) -> LayerAreaPerMolecule:
        """Get the back head layer of the bilayer."""
        return self.layers[3]

    @back_head_layer.setter
    def back_head_layer(self, layer: LayerAreaPerMolecule) -> None:
        """Set the back head layer of the bilayer."""
        self.layers[3] = layer

    @property
    def constrain_heads(self) -> bool:
        """Get the head layer constraint status."""
        return self._constrain_heads

    @constrain_heads.setter
    def constrain_heads(self, status: bool) -> None:
        """Set the status for head layer constraints.

        When enabled, the back head layer thickness and area per molecule
        are constrained to match the front head layer. Solvent fraction
        (hydration) remains independent.

        Parameters
        ----------
        status : bool
            Boolean for the constraint status.
        """
        if status:
            self._enable_head_constraints()
        else:
            self._disable_head_constraints()
        self._constrain_heads = status

    def _enable_head_constraints(self) -> None:
        """Enable head layer constraints (thickness and area per molecule)."""
        front_head = self.front_head_layer
        back_head = self.back_head_layer

        # Constrain thickness
        back_head.thickness.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_head.thickness},
        )

        # Constrain area per molecule
        back_head.area_per_molecule_parameter.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_head.area_per_molecule_parameter},
        )

    def _disable_head_constraints(self) -> None:
        """Disable head layer constraints."""
        self.back_head_layer.thickness.make_independent()
        self.back_head_layer.area_per_molecule_parameter.make_independent()

    @property
    def conformal_roughness(self) -> bool:
        """Get the roughness constraint status."""
        return self._conformal_roughness

    @conformal_roughness.setter
    def conformal_roughness(self, status: bool) -> None:
        """Set the status for conformal roughness.

        When enabled, all layers share the same roughness parameter
        (controlled by the front head layer).

        Parameters
        ----------
        status : bool
            Boolean for the constraint status.
        """
        if status:
            self._setup_roughness_constraints()
            self._enable_roughness_constraints()
        else:
            if self._roughness_constraints_setup:
                self._disable_roughness_constraints()
        self._conformal_roughness = status

    def constrain_solvent_roughness(self, solvent_roughness: Parameter) -> None:
        """Add the constraint to the solvent roughness.

        Parameters
        ----------
        solvent_roughness : Parameter
            The solvent roughness parameter.
        """
        if not self.conformal_roughness:
            raise ValueError('Roughness must be conformal to use this function.')
        solvent_roughness.value = self.front_head_layer.roughness.value
        solvent_roughness.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': self.front_head_layer.roughness},
        )

    def constrain_multiple_contrast(
        self,
        another_contrast: Bilayer,
        front_head_thickness: bool = True,
        back_head_thickness: bool = True,
        tail_thickness: bool = True,
        front_head_area_per_molecule: bool = True,
        back_head_area_per_molecule: bool = True,
        tail_area_per_molecule: bool = True,
        front_head_fraction: bool = True,
        back_head_fraction: bool = True,
        tail_fraction: bool = True,
    ) -> None:
        """Constrain structural parameters between bilayer objects.

        Makes this bilayer's parameters dependent on another_contrast's parameters,
        so that changes to another_contrast propagate to this bilayer.

        Parameters
        ----------
        another_contrast : Bilayer
            The bilayer to constrain to.
        front_head_thickness : bool, optional
            Constrain front head thickness. By default, True.
        back_head_thickness : bool, optional
            Constrain back head thickness. By default, True.
        tail_thickness : bool, optional
            Constrain tail thickness. By default, True.
        front_head_area_per_molecule : bool, optional
            Constrain front head area per molecule. By default, True.
        back_head_area_per_molecule : bool, optional
            Constrain back head area per molecule. By default, True.
        tail_area_per_molecule : bool, optional
            Constrain tail area per molecule. By default, True.
        front_head_fraction : bool, optional
            Constrain front head solvent fraction. By default, True.
        back_head_fraction : bool, optional
            Constrain back head solvent fraction. By default, True.
        tail_fraction : bool, optional
            Constrain tail solvent fraction. By default, True.
        """
        if front_head_thickness:
            constrain_equal(self.front_head_layer.thickness, another_contrast.front_head_layer.thickness)

        if back_head_thickness:
            constrain_equal(self.back_head_layer.thickness, another_contrast.back_head_layer.thickness)

        if tail_thickness:
            constrain_equal(self.front_tail_layer.thickness, another_contrast.front_tail_layer.thickness)

        if front_head_area_per_molecule:
            constrain_equal(
                self.front_head_layer.area_per_molecule_parameter, another_contrast.front_head_layer.area_per_molecule_parameter
            )

        if back_head_area_per_molecule:
            constrain_equal(
                self.back_head_layer.area_per_molecule_parameter, another_contrast.back_head_layer.area_per_molecule_parameter
            )

        if tail_area_per_molecule:
            constrain_equal(
                self.front_tail_layer.area_per_molecule_parameter, another_contrast.front_tail_layer.area_per_molecule_parameter
            )

        if front_head_fraction:
            constrain_equal(
                self.front_head_layer.solvent_fraction_parameter, another_contrast.front_head_layer.solvent_fraction_parameter
            )

        if back_head_fraction:
            constrain_equal(
                self.back_head_layer.solvent_fraction_parameter, another_contrast.back_head_layer.solvent_fraction_parameter
            )

        if tail_fraction:
            constrain_equal(
                self.front_tail_layer.solvent_fraction_parameter, another_contrast.front_tail_layer.solvent_fraction_parameter
            )

    @property
    def _dict_repr(self) -> dict:
        """A simplified dict representation."""
        return {
            self.name: {
                'front_head_layer': self.front_head_layer._dict_repr,
                'front_tail_layer': self.front_tail_layer._dict_repr,
                'back_tail_layer': self.back_tail_layer._dict_repr,
                'back_head_layer': self.back_head_layer._dict_repr,
                'constrain_heads': self.constrain_heads,
                'conformal_roughness': self.conformal_roughness,
            }
        }

    def to_dict(self, skip: list[str] | None = None) -> dict:
        """Serialize, dropping the derived `layers` collection.

        ``__init__`` rebuilds `layers` from the four layer arguments. The
        `back_tail_layer` is kept: it is a constructor argument and may hold
        state of its own (a formula, or a parameter made independent).
        """
        this_dict = super().to_dict(skip=skip)
        this_dict.pop('layers', None)
        return this_dict

    def as_dict(self, skip: list[str] | None = None) -> dict:
        """Compatibility alias for :meth:`to_dict`."""
        return self.to_dict(skip=skip)
