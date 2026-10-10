# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Contrasts: one structure measured under several SLD conditions.

Each contrast is an ordinary :class:`~easyreflectometry.model.Model` with its
own scale, background and resolution. What the contrasts have in common is
shared in one of two ways:

- *by identity*: an assembly or material is one object used by several
  models, so it has one set of parameters;
- *by equality*: a parameter of one model follows the corresponding
  parameter of another (``constrain_equal``), where the objects must differ,
  e.g. a hydrated layer whose solvent differs between contrasts.

:func:`derive_contrast` builds a new contrast from a reference model and a
list of substitutions; :func:`plan_link` lays out the equalities that would
tie the parameters of two existing models. :class:`~easyreflectometry.project.Project`
wraps both (``add_contrast``, ``plan_link``/``apply_link``/``unlink``,
``detach``).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Union

import periodictable
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easyreflectometry.constraints import USER_CONSTRAINT_FLAG
from easyreflectometry.constraints import constrain_equal
from easyreflectometry.model import Model
from easyreflectometry.model.resolution_functions import ResolutionFunction
from easyreflectometry.sample import LayerAreaPerMolecule
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialDensity
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Sample
from easyreflectometry.sample import paths
from easyreflectometry.sample import references

#: Parameters describing a material's chemistry: two contrasts usually differ
#: in them, so linking never ties them without being told to.
CHEMISTRY = frozenset({'sld', 'isld', 'density', 'scattering_length_real', 'scattering_length_imag'})


class UnsupportedSubstitution(ValueError):
    """A substitution :func:`derive_contrast` cannot carry out; nothing was built."""


class LayoutMismatch(ValueError):
    """Two models whose parameters do not correspond one to one."""


@dataclass(frozen=True)
class ReplaceMaterial:
    """Use `new` wherever the reference uses `old` (a material, or a mixture's component)."""

    old: Union[Material, MaterialMixture]
    new: Union[Material, MaterialMixture]


@dataclass(frozen=True)
class ReplaceFormula:
    """Give the contrast's copy of `target` another chemical formula (isotopic substitution)."""

    target: Union[LayerAreaPerMolecule, MaterialDensity]
    formula: str


def derive_contrast(
    reference: Model,
    *,
    name: str,
    substitutions: Optional[Sequence[Union[ReplaceMaterial, ReplaceFormula]]] = None,
) -> Model:
    """A new contrast of `reference`'s structure; `reference` is not changed.

    Assemblies the substitutions do not touch are shared with `reference`.
    An assembly they touch is copied: its materials are shared except the
    replaced ones, its layers are new, and every independent parameter of the
    copy is tied to the reference's except those of the replacements (and the
    scattering lengths of a re-formulated material). So a hydrated layer in
    another solvent keeps the reference's thickness, roughness and solvent
    fraction, and its effective SLD follows the dry material it still shares.
    Scale, background and resolution are copied by value and stay independent.

    Raises
    ------
    UnsupportedSubstitution
        A target is not part of `reference`, or a formula target is neither a
        :class:`LayerAreaPerMolecule` nor a :class:`MaterialDensity`.
    """
    substitutions = list(substitutions or [])
    swaps = {id(item.old): item.new for item in substitutions if isinstance(item, ReplaceMaterial)}
    formulas = {id(item.target): item for item in substitutions if isinstance(item, ReplaceFormula)}
    targets = set(swaps) | set(formulas)
    in_reference = {id(obj) for obj in _objects(reference.sample)}
    for item in substitutions:
        target = item.old if isinstance(item, ReplaceMaterial) else item.target
        if id(target) not in in_reference:
            raise UnsupportedSubstitution(f"'{getattr(target, 'name', target)}' is not part of '{reference.name}'.")
        if isinstance(item, ReplaceFormula):
            if not isinstance(target, (LayerAreaPerMolecule, MaterialDensity)):
                raise UnsupportedSubstitution(f"'{target.name}' has no chemical formula to replace.")
            _check_formula(item.formula)

    assemblies = []
    for assembly in reference.sample:
        if not any(id(obj) in targets for obj in _objects(assembly)):
            assemblies.append(assembly)
            continue
        copy = references.rebuild(assembly, kept_children(assembly, swaps, targets))
        corresponding = _corresponding_objects(assembly, copy)
        excluded = {id(parameter) for new in swaps.values() for parameter in new.get_all_parameters()}
        for target_id, item in formulas.items():
            if target_id in corresponding:
                target_copy = corresponding[target_id]
                if isinstance(target_copy, LayerAreaPerMolecule):
                    target_copy.molecular_formula = item.formula
                else:
                    target_copy.chemical_structure = item.formula
                    excluded |= {id(target_copy.scattering_length_real), id(target_copy.scattering_length_imag)}
        tie_corresponding(copy, assembly, excluded)
        assemblies.append(copy)

    model = Model(
        sample=Sample(*assemblies),
        name=name,
        resolution_function=ResolutionFunction.from_dict(reference.resolution_function.as_dict()),
    )
    for attribute in ('scale', 'background'):
        restore(getattr(model, attribute), snapshot(getattr(reference, attribute)))
    return model


def substitution_candidates(model: Model) -> tuple[list[Any], list[Any]]:
    """What :func:`derive_contrast` can substitute in `model`, each once, in sample order.

    Returns
    -------
    tuple[list, list]
        The materials and mixture components a :class:`ReplaceMaterial` can
        replace (not those a :class:`LayerAreaPerMolecule` builds for itself),
        and the layers and materials a :class:`ReplaceFormula` can re-formulate.
    """
    materials: list[Any] = []
    formulas: list[Any] = []
    seen: set[int] = set()

    def visit(node: Any, owned: bool) -> None:
        if id(node) in seen or isinstance(node, DescriptorNumber):
            return
        seen.add(id(node))
        if isinstance(node, (LayerAreaPerMolecule, MaterialDensity)):
            formulas.append(node)
        if isinstance(node, (Material, MaterialMixture)) and not owned:
            materials.append(node)
        for _, child in paths.children(node):
            internal = isinstance(node, LayerAreaPerMolecule) and child is node.material
            visit(child, internal or (owned and child is not getattr(node, 'solvent', None)))

    visit(model.sample, False)
    return materials, formulas


def _check_formula(formula: str) -> None:
    try:
        atoms = periodictable.formula(formula).atoms
    except Exception as error:  # periodictable raises its own parse errors
        raise UnsupportedSubstitution(f"'{formula}' is not a chemical formula: {error}") from None
    if not atoms:
        raise UnsupportedSubstitution('A chemical formula is needed.')


def tie_corresponding(copy: Any, reference: Any, excluded: set[int] = frozenset()) -> list[Parameter]:
    """Tie every independent parameter of `copy` to the reference parameter at the same path.

    Parameters that are the same object, have no counterpart or are listed in
    `excluded` (by id) are left alone. Returns the parameters tied.
    """
    leaders = _parameters(reference, [])
    tied = []
    for path, parameter in _parameters(copy, []).items():
        leader = leaders.get(path)
        if leader is None or leader is parameter or id(parameter) in excluded or not parameter.independent:
            continue
        constrain_equal(parameter, leader)
        tied.append(parameter)
    return tied


@dataclass
class LinkRow:
    """One pair of corresponding parameters of two models and what linking does with it.

    `action` is ``'tie'``, ``'skip'``, ``'already_tied'``, ``'conflict'`` or
    ``'undecided'`` (a chemistry parameter: set it to ``'tie'`` or ``'skip'``).
    """

    path: str
    follower: Parameter
    reference: Parameter
    action: str
    reason: str = ''
    #: Indices of the other models the follower parameter belongs to as well.
    shared_with: list[int] = field(default_factory=list)


@dataclass
class LinkPlan:
    follower: Model
    reference: Model
    rows: list[LinkRow]


@dataclass
class LinkRecord:
    """The equalities one :meth:`Project.apply_link` made, so they can be undone together."""

    follower: Model
    reference: Model
    #: (follower parameter, reference parameter, its value/min/max/fixed before the tie)
    pairs: list[tuple[Parameter, Parameter, dict]]


def snapshot(parameter: Parameter) -> dict:
    """The state :func:`restore` gives back to `parameter` (a tie replaces its value and bounds)."""
    return {'value': parameter.value, 'min': parameter.min, 'max': parameter.max, 'fixed': parameter.fixed}


def restore(parameter: Parameter, state: dict) -> None:
    parameter.min, parameter.max = -float('inf'), float('inf')
    parameter.value = state['value']
    parameter.min, parameter.max = state['min'], state['max']
    parameter.fixed = state['fixed']


def plan_link(
    follower: Model,
    reference: Model,
    *,
    owners: Callable[[Parameter], list[int]],
    follower_index: int,
    materials: Optional[str] = None,
) -> LinkPlan:
    """The equalities that would tie `follower`'s parameters to `reference`'s; nothing is changed.

    Parameters correspond by their path within the model, so the two layouts
    must match. Scale and background stay per contrast; parameters that are
    one object, derived, disabled or followed by something else are skipped
    with a reason; a parameter the follower shares with a third model is
    skipped as well (tying it would move that model too). Chemistry
    parameters (:data:`CHEMISTRY`) are ``'undecided'`` unless `materials` is
    ``'tie'`` or ``'skip'``.

    Raises
    ------
    LayoutMismatch
        The models' parameters do not correspond one to one.
    """
    # Every path: a material used by two layers of one model and by two distinct
    # objects in the other still corresponds layer by layer.
    follower_parameters = _parameters(follower.sample, ['sample'], every_path=True)
    reference_parameters = _parameters(reference.sample, ['sample'], every_path=True)
    if follower_parameters.keys() != reference_parameters.keys():
        mismatch = sorted(follower_parameters.keys() ^ reference_parameters.keys())[0]
        raise LayoutMismatch(f"'{follower.name}' and '{reference.name}' differ in layout at {mismatch}.")
    rows = [
        LinkRow(name, getattr(follower, name), getattr(reference, name), 'skip', 'per contrast')
        for name in ('scale', 'background')
    ]
    planned: dict[int, str] = {}
    for path, parameter in follower_parameters.items():
        row = LinkRow(path, parameter, reference_parameters[path], 'tie')
        if id(parameter) in planned:
            row.action, row.reason = 'skip', f'the same parameter as {planned[id(parameter)]}'
        else:
            planned[id(parameter)] = path
            _classify(row, follower_index, owners, materials)
        rows.append(row)
    return LinkPlan(follower, reference, rows)


def _classify(row: LinkRow, follower_index: int, owners: Callable[[Parameter], list[int]], materials: Optional[str]) -> None:
    follower, reference = row.follower, row.reference
    if follower is reference:
        row.action, row.reason = 'skip', 'shared object'
    elif not follower.independent:
        if not getattr(follower, USER_CONSTRAINT_FLAG, False):
            row.action, row.reason = 'skip', 'derived'
        elif follows(follower, reference):
            row.action, row.reason = 'already_tied', 'already follows the reference'
        else:
            row.action, row.reason = 'conflict', f'follows {follower._clean_dependency_string}'
    elif not (getattr(follower, 'enabled', True) and getattr(reference, 'enabled', True)):
        row.action, row.reason = 'skip', 'not used'
    elif _depends_on(reference, follower):
        row.action, row.reason = 'conflict', 'the reference follows it'
    else:
        row.shared_with = [index for index in owners(follower) if index != follower_index]
        if row.shared_with:
            row.action, row.reason = 'skip', 'shared with other models'
        elif follower.name in CHEMISTRY:
            row.action = 'undecided' if materials is None else materials
            row.reason = 'material chemistry'


def follows(parameter: Parameter, leader: Parameter) -> bool:
    """Whether `parameter` is a plain equality tie to `leader`."""
    dependency_map = parameter.dependency_map or {}
    return (
        not parameter.independent
        and len(dependency_map) == 1
        and next(iter(dependency_map.values())) is leader
        and parameter._clean_dependency_string.strip() == next(iter(dependency_map))
    )


def _depends_on(parameter: Parameter, other: Parameter) -> bool:
    """Whether `parameter` follows `other`, directly or through other dependencies."""
    pending, seen = [parameter], set()
    while pending:
        current = pending.pop()
        if current is other:
            return True
        if id(current) in seen or current.independent:
            continue
        seen.add(id(current))
        pending.extend(value for value in (current.dependency_map or {}).values() if isinstance(value, Parameter))
    return False


def _parameters(root: Any, tokens: list[str], every_path: bool = False) -> dict[str, Parameter]:
    """Path to parameter under `root` (descriptors that cannot be fitted or tied left out);
    see :func:`paths.walk` for `every_path`."""
    return {
        path: parameter
        for path, parameter in paths.walk(root, tokens, every_path=every_path)
        if isinstance(parameter, Parameter)
    }


def _objects(root: Any) -> Iterator[Any]:
    """`root` and every sample object under it, each once."""
    seen: set[int] = set()
    pending = [root]
    while pending:
        obj = pending.pop()
        if id(obj) in seen or isinstance(obj, DescriptorNumber):
            continue
        seen.add(id(obj))
        yield obj
        pending.extend(child for _, child in paths.children(obj))


def kept_children(assembly: Any, swaps: dict[int, Any], targets: set[int]) -> dict[int, Any]:
    """What a copy of `assembly` keeps (see :func:`references.rebuild`).

    Each replaced material maps to its replacement; every other material that
    holds no target is shared. Layers are copied, and so are the materials a
    :class:`LayerAreaPerMolecule` builds for itself (all but its solvent),
    because their constraints belong to that layer.
    """
    keep: dict[int, Any] = {}
    seen: set[int] = set()

    def visit(node: Any, owned: bool) -> None:
        if id(node) in seen or isinstance(node, DescriptorNumber):
            return
        seen.add(id(node))
        if id(node) in swaps:
            keep[id(node)] = swaps[id(node)]
            return
        if (
            isinstance(node, (Material, MaterialMixture))
            and not owned
            and not any(id(obj) in targets for obj in _objects(node))
        ):
            keep[id(node)] = node
            return
        for _, child in paths.children(node):
            internal = isinstance(node, LayerAreaPerMolecule) and child is node.material
            visit(child, internal or (owned and child is not getattr(node, 'solvent', None)))

    visit(assembly, False)
    return keep


def _corresponding_objects(reference: Any, copy: Any) -> dict[int, Any]:
    """``id(reference object)`` to the object at the same place in `copy`."""
    found: dict[int, Any] = {}

    def visit(original: Any, duplicate: Any) -> None:
        if id(original) in found or isinstance(original, DescriptorNumber):
            return
        found[id(original)] = duplicate
        duplicates = dict(paths.children(duplicate))
        for token, child in paths.children(original):
            if token in duplicates:
                visit(child, duplicates[token])

    visit(reference, copy)
    return found
