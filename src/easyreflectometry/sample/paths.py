# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Structural paths of parameters, e.g. ``models/0/sample/1/layers/0/thickness``.

Paths are stable across save/load (unlike unique names) and are how a
project addresses parameters in saved constraints. Within one model the
relative paths (``sample/1/layers/0/thickness``) also match the parameters
of two models with the same layout.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from typing import Iterator
from typing import Optional

from easyscience.variable import DescriptorNumber

logger = logging.getLogger(__name__)

#: Properties not descended into when *generating* paths: non-structural
#: objects and convenience aliases of ``layers[i]`` (so a layer parameter is
#: always addressed as ``.../layers/<i>/...``). Resolving a path still accepts them.
SKIPPED_PROPERTIES = frozenset({'interface', 'parent', 'front_layer', 'back_layer', 'head_layer', 'tail_layer'})


def children(obj: Any) -> list[tuple[str, Any]]:
    """``(token, child)`` pairs to descend into from `obj`."""
    from easyreflectometry.sample.base_core import BaseCore
    from easyreflectometry.sample.collections.base_collection import BaseCollection

    if isinstance(obj, (BaseCollection, list, tuple)) or (isinstance(obj, Sequence) and not isinstance(obj, str)):
        return [(str(index), item) for index, item in enumerate(obj)]
    if isinstance(obj, BaseCore) or hasattr(obj, 'get_all_parameters'):
        candidates = []
        for attr_name in dir(type(obj)):
            if attr_name.startswith('_') or attr_name in SKIPPED_PROPERTIES:
                continue
            class_attr = getattr(type(obj), attr_name, None)
            if not isinstance(class_attr, property):
                continue
            try:
                value = getattr(obj, attr_name)
            except Exception as exception:
                logger.debug("Skipping property '%s' on %r: %s", attr_name, obj, exception)
                continue
            if isinstance(value, (DescriptorNumber, BaseCore, BaseCollection, list, tuple)):
                candidates.append((attr_name, value))
        return candidates
    return []


def walk(
    root: Any, tokens: list[str], visited: Optional[set[int]] = None, every_path: bool = False
) -> Iterator[tuple[str, Any]]:
    """Yield ``(path, parameter)`` for every parameter under `root`, paths starting with `tokens`.

    By default each object is descended into once (per `visited`, shared
    between calls when given), so an object reachable from several places is
    reported at its first path only. With `every_path`, it is reported at each
    of them; only a cycle back to an ancestor is cut.
    """
    visited = set() if visited is None else visited
    if isinstance(root, DescriptorNumber):
        yield '/'.join(tokens), root
        return
    if id(root) in visited:
        return
    visited.add(id(root))
    for token, child in children(root):
        yield from walk(child, tokens + [token], visited, every_path)
    if every_path:
        visited.discard(id(root))
