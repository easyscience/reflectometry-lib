# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from numbers import Number
from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.limits import apply_default_limits
from easyreflectometry.sample import BaseAssembly
from easyreflectometry.sample import Sample
from easyreflectometry.sample.base_core import BaseCore
from easyreflectometry.utils import get_as_parameter

from .resolution_functions import PercentageFwhm
from .resolution_functions import ResolutionFunction

DEFAULTS = {
    'scale': {
        'description': 'Scaling of the reflectomety profile',
        'url': 'https://github.com/reflectivity/edu_outreach/blob/master/refl_maths/paper.tex',
        'value': 1.0,
        'min': 0,
        'max': np.inf,
        'fixed': True,
    },
    'background': {
        'description': 'Linear background to include in reflectometry data',
        'url': 'https://github.com/reflectivity/edu_outreach/blob/master/refl_maths/paper.tex',
        'value': 1e-8,
        'min': 0.0,
        'max': np.inf,
        'fixed': True,
    },
    'resolution': {
        'value': 5.0,
    },
}

COLORS = [
    '#0173B2',
    '#DE8F05',
    '#029E73',
    '#D55E00',
    '#CC78BC',
    '#CA9161',
    '#FBAFE4',
    '#949494',
    '#ECE133',
    '#56B4E9',
]


class Model(BaseCore):
    """Model is the class that represents the experiment.

    It is used to store the information about the experiment and to perform the calculations.
    """

    def __init__(
        self,
        sample: Union[Sample, None] = None,
        scale: Union[Parameter, Number, None] = None,
        background: Union[Parameter, Number, None] = None,
        resolution_function: Union[ResolutionFunction, None] = None,
        name: str = 'Model',
        color: str = COLORS[0],
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        color : str, optional
            By default, COLORS[0].
        sample : Union[Sample, None], optional
            The sample being modelled. By default, None.
        scale : Union[Parameter, Number, None], optional
            Scaling factor of profile. By default, None.
        background : Union[Parameter, Number, None], optional
            Linear background magnitude. By default, None.
        name : str, optional
            Name of the model. By default, 'Model'.
        resolution_function : Union[ResolutionFunction, None], optional
            Resolution function. By default, None.
        interface :
            Calculator interface. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if sample is None:
            sample = Sample(interface=interface)
        if resolution_function is None:
            resolution_function = PercentageFwhm(DEFAULTS['resolution']['value'])

        scale = get_as_parameter('scale', scale, DEFAULTS)
        apply_default_limits(scale, 'scale')
        background = get_as_parameter('background', background, DEFAULTS)
        self.color = color
        self._is_default = False
        self._resolution_function = resolution_function

        super().__init__(name=name, unique_name=unique_name)
        self._sample = sample
        self._scale = scale
        self._background = background

        # Derived, read-only "calculation" parameter (see `total_thickness`).
        # Rebuilt by every constructor call — including `from_dict` — and
        # deliberately *not* serialized: it is a function of the layers.
        self._total_thickness = Parameter(
            name='total_thickness',
            value=0.0,
            unit='angstrom',
            fixed=True,
            description='Total thickness of the film: the sum of all layer thicknesses '
            'between the superphase and the subphase (derived, read-only).',
        )
        self._total_thickness_sources: list[Parameter] = []
        self.refresh_derived()

        # Set interface last — propagates to children via BaseCore.generate_bindings
        # and then sets the resolution function on the calculator (see setter).
        if interface is not None:
            self.interface = interface

    # ----- @property accessors for serialization round-trip -----

    @property
    def sample(self) -> Sample:
        return self._sample

    @sample.setter
    def sample(self, value: Sample) -> None:
        self._sample = value

    @property
    def scale(self) -> Parameter:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale.value = value

    @property
    def background(self) -> Parameter:
        return self._background

    @background.setter
    def background(self, value: float) -> None:
        self._background.value = value

    # ----- derived parameters -----

    def _film_layers(self) -> list:
        """Layers between the superphase and the subphase (first and last layer of the sample)."""
        layers = [layer for assembly in self.sample for layer in assembly.layers]
        return layers[1:-1] if len(layers) >= 3 else []

    def refresh_derived(self) -> None:
        """Re-derive the model's computed parameters from the current layer structure.

        Cheap (a few string operations) and idempotent: the dependency is
        only rebuilt when the set of contributing layers changed. Called on
        every access to :attr:`total_thickness`, so layers added or removed
        through *any* path (`Model.add_assemblies`, `assembly.layers.append`,
        ...) are picked up without explicit notification.
        """
        sources = [layer.thickness for layer in self._film_layers()]
        current = self._total_thickness_sources
        if len(sources) == len(current) and all(a is b for a, b in zip(sources, current)):
            return
        self._total_thickness_sources = sources
        total = self._total_thickness
        if not sources:
            if not total.independent:
                total.make_independent()
            # `make_dependent_on` clears `fixed`, and `make_independent` does not
            # put it back; without this the parameter would resurface as a free
            # fit parameter once the film is emptied.
            total.fixed = True
            total.value = 0.0
            return
        dependency_map = {f't{index}': source for index, source in enumerate(sources)}
        total.make_dependent_on(
            dependency_expression=' + '.join(dependency_map.keys()),
            dependency_map=dependency_map,
        )

    @property
    def total_thickness(self) -> Parameter:
        """Read-only parameter: the summed thickness of the film layers.

        The superphase (first layer of the sample) and the subphase (last
        layer) are semi-infinite and excluded. The parameter is *dependent*:
        it tracks the layer thicknesses, never enters a fit and cannot be
        set; it can be referenced from constraint expressions and
        inequality constraints like any other parameter.
        """
        self.refresh_derived()
        return self._total_thickness

    # ----- assembly management -----

    def add_assemblies(self, *assemblies: list[BaseAssembly]) -> None:
        """Add assemblies to the model sample.

        Parameters
        ----------
        *assemblies : list[BaseAssembly]
            Assemblies to add to model sample.
        """
        if not assemblies:
            self.sample.add_assembly()
            if self.interface is not None:
                self.interface().add_item_to_model(self.sample[-1].unique_name, self.unique_name)
        else:
            for assembly in assemblies:
                if issubclass(assembly.__class__, BaseAssembly):
                    self.sample.add_assembly(assembly)
                    if self.interface is not None:
                        self.interface().add_item_to_model(self.sample[-1].unique_name, self.unique_name)
                else:
                    raise ValueError(f'Object {assembly} is not a valid type, must be a child of BaseAssembly.')

    def duplicate_assembly(self, index: int) -> None:
        """Duplicate a given item or layer in a sample.

        Parameters
        ----------
        index : int
        idx :
            Index of the item or layer to duplicate.
        """
        self.sample.duplicate_assembly(index)
        if self.interface is not None:
            self.interface().add_item_to_model(self.sample[-1].unique_name, self.unique_name)

    def remove_assembly(self, index: int) -> None:
        """Remove an assembly from the model.

        Parameters
        ----------
        index : int
        idx :
            Index of the item to remove.
        """
        assembly_unique_name = self.sample[index].unique_name
        self.sample.remove_assembly(index)
        if self.interface is not None:
            self.interface().remove_item_from_model(assembly_unique_name, self.unique_name)

    @property
    def has_magnetism(self) -> bool:
        """Whether any layer in the sample carries magnetic properties."""
        return any(getattr(layer, 'magnetism', None) is not None for assembly in self.sample for layer in assembly.layers)

    @property
    def is_default(self) -> bool:
        """Whether this model was created as a default placeholder."""
        return self._is_default

    @is_default.setter
    def is_default(self, value: bool) -> None:
        """Set whether this model is a default placeholder."""
        self._is_default = value

    # ----- resolution function -----

    @property
    def resolution_function(self) -> ResolutionFunction:
        """Return the resolution function."""
        return self._resolution_function

    @resolution_function.setter
    def resolution_function(self, resolution_function: ResolutionFunction) -> None:
        """Set the resolution function for the model."""
        self._resolution_function = resolution_function
        if self.interface is not None:
            self.interface().set_resolution_function(self._resolution_function)

    # ----- interface (override BaseCore's to add resolution-function side effect) -----

    @BaseCore.interface.setter
    def interface(self, new_interface) -> None:
        """Set the interface; runs `generate_bindings` and then refreshes the
        calculator's resolution function.
        """
        # Call BaseCore.interface.setter for the binding propagation.
        BaseCore.interface.fset(self, new_interface)
        if new_interface is not None:
            new_interface().set_resolution_function(self._resolution_function)

    # ----- representation -----

    @property
    def _dict_repr(self) -> dict[str, dict[str, str]]:
        """A simplified dict representation."""
        if isinstance(self._resolution_function, PercentageFwhm):
            resolution_value = self._resolution_function.as_dict()['constant']
            resolution = f'{resolution_value} %'
        else:
            resolution = 'function of Q'

        return {
            self.name: {
                'scale': float(self.scale.value),
                'background': float(self.background.value),
                'resolution': resolution,
                'color': self.color,
                'sample': self.sample._dict_repr,
            }
        }

    # ----- serialization (custom because resolution_function + interface need special handling) -----

    def to_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Serialize the model, encoding the resolution function and interface name."""
        if skip is None:
            skip = []
        # Sample/resolution_function/interface get bespoke encoding below.
        skip_for_super = list(skip) + ['sample', 'resolution_function', 'interface']
        this_dict = super().to_dict(skip=skip_for_super)
        this_dict['sample'] = self.sample.as_dict(skip=skip)
        this_dict['resolution_function'] = self.resolution_function.as_dict(skip=skip)
        if self.interface is None:
            this_dict['interface'] = None
        else:
            this_dict['interface'] = self.interface().name
        return this_dict

    def as_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Compatibility alias for :meth:`to_dict`."""
        return self.to_dict(skip=skip)

    def as_orso(self) -> dict:
        """The sample as an ORSO simple-model (``sample.model``) dictionary.

        Slab representation: lengths in angstrom, SLDs in 1/angstrom^2,
        repeating multilayers via the inline ``N ( ... )`` stack syntax.

        Returns
        -------
        dict
            The ORSO model-language dictionary (the content of an .ort file's
            ``data_source.sample.model`` section).
        """
        # Circular import if hoisted to module-top.
        from orsopy.fileio import Header

        from easyreflectometry.orso_utils import sample_to_orso_model

        return Header.asdict(sample_to_orso_model(self.sample))

    @classmethod
    def from_dict(cls, passed_dict: dict) -> Model:
        """Create a Model from a dictionary."""
        # Circular import if hoisted to module-top.
        from easyreflectometry.calculators import CalculatorFactory

        this_dict = copy.deepcopy(passed_dict)
        resolution_function = ResolutionFunction.from_dict(this_dict.pop('resolution_function'))
        interface_name = this_dict.pop('interface')
        if interface_name is not None:
            interface = CalculatorFactory()
            interface.switch(interface_name)
        else:
            interface = None

        model = super().from_dict(this_dict)

        model.resolution_function = resolution_function
        model.interface = interface
        return model
