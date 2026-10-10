# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Shared sample objects in a serialized tree.

A material, layer or assembly may sit in several places of a project (two
layers of one model, two contrasts sharing a structure). Serialized naively,
each place gets its own copy and the sharing is lost on reload. Inside
:func:`writing`, the first occurrence of an object is written in full and
tagged with ``@ref_id``; every later occurrence is written as a reference node
``{'@ref': id, '@module': ..., '@class': ...}``. Inside :func:`reading`, a
reference node resolves to the object built for that id, so each shared
object is constructed once and its owners refer to that one object.

The same tables let a caller build a copy that keeps chosen children: seed the
writer with the objects to keep (or to swap) and the reader with what each id
should resolve to (:func:`rebuild`).
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any
from typing import Iterator
from typing import Optional

REF = '@ref'
REF_ID = '@ref_id'

_writer: contextvars.ContextVar[Optional['_Writer']] = contextvars.ContextVar('_reference_writer', default=None)
_reader: contextvars.ContextVar[Optional['_Reader']] = contextvars.ContextVar('_reference_reader', default=None)


class _Writer:
    def __init__(self) -> None:
        # id(obj) -> (ref id, obj); the object is held so its id is not reused.
        self.ids: dict[int, tuple[str, Any]] = {}

    def key_for(self, obj: Any) -> str:
        entry = self.ids.get(id(obj))
        if entry is None:
            entry = (str(len(self.ids)), obj)
            self.ids[id(obj)] = entry
        return entry[0]


class _Reader:
    def __init__(self, report: list[str]) -> None:
        self.objects: dict[str, Any] = {}
        self.report = report


@contextlib.contextmanager
def writing() -> Iterator[_Writer]:
    """Serialize shared objects once; drop unused ``@ref_id`` tags afterwards with :func:`prune`."""
    writer = _Writer()
    token = _writer.set(writer)
    try:
        yield writer
    finally:
        _writer.reset(token)


@contextlib.contextmanager
def reading(report: Optional[list[str]] = None) -> Iterator[_Reader]:
    """Resolve reference nodes while deserializing; problems are appended to `report`."""
    reader = _Reader(report if report is not None else [])
    token = _reader.set(reader)
    try:
        yield reader
    finally:
        _reader.reset(token)


@contextlib.contextmanager
def suspended() -> Iterator[None]:
    """No tagging inside: for a serialization pass whose output is discarded."""
    token = _writer.set(None)
    try:
        yield
    finally:
        _writer.reset(token)


def is_reference(value: Any) -> bool:
    return isinstance(value, dict) and REF in value


def tag(obj: Any, d: dict) -> dict:
    """Serializer hook: `d` is `obj` serialized; return it tagged, or a reference node."""
    writer = _writer.get()
    if writer is None:
        return d
    seen = id(obj) in writer.ids
    key = writer.key_for(obj)
    if seen:
        return {REF: key, '@module': d['@module'], '@class': d['@class']}
    d[REF_ID] = key
    return d


def resolve(value: Any) -> Any:
    """The object a reference node stands for; anything else is returned unchanged.

    An unknown id is reported and the node is returned as is, so it
    deserializes to a default object of its class.
    """
    if not is_reference(value):
        return value
    reader = _reader.get()
    if reader is None:
        raise ValueError(f'Reference {value[REF]!r} found outside a project load; it cannot be resolved.')
    obj = reader.objects.get(value[REF])
    if obj is None:
        reader.report.append(f'Shared {value["@class"]} {value[REF]!r} is missing from the file; a default one is used.')
        return value
    return obj


def deserialize(value: Any) -> Any:
    """`value` deserialized: a reference node resolves to its object, anything
    else (including an unresolvable reference, built as a default of its class)
    goes through the serializer."""
    from easyscience.io import SerializerBase

    obj = resolve(value)
    # A single-key dict routes the value through the serializer's per-value
    # path, which uses the value's own class `from_dict`.
    return obj if obj is not value else SerializerBase.deserialize_dict({'item': value})['item']


def build_children(obj_dict: dict) -> dict:
    """`obj_dict` with its sample-object children built, in document order.

    Building them here, one after the other as they were written, means a reference
    always meets an object that is already built, also when two sibling slots hold
    the same object (``MaterialMixture(a, a)``). Other children (parameters,
    collections) are left to the regular deserialization: a built collection would
    be turned into a plain list by it.
    """
    return {key: deserialize(value) if _is_sample_object(value) else value for key, value in obj_dict.items()}


def _is_sample_object(value: Any) -> bool:
    if is_reference(value):
        return True
    if not (isinstance(value, dict) and '@module' in value and '@class' in value):
        return False
    from easyscience.io import SerializerBase

    from easyreflectometry.sample.base_core import BaseCore

    try:
        cls = SerializerBase._import_class(value['@module'], value['@class'])
    except ImportError:
        return False
    return isinstance(cls, type) and issubclass(cls, BaseCore)


def register(obj_dict: dict, obj: Any) -> None:
    """Record `obj` as built from `obj_dict`, so later reference nodes resolve to it."""
    reader = _reader.get()
    if reader is not None and REF_ID in obj_dict:
        reader.objects[obj_dict[REF_ID]] = obj


def prune(node: Any) -> Any:
    """Drop the ``@ref_id`` tags no reference node in `node` points to, in place; returns `node`."""
    referenced: set[str] = set()

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            if REF in item:
                referenced.add(item[REF])
            for value in item.values():
                collect(value)
        elif isinstance(item, list):
            for value in item:
                collect(value)

    def strip(item: Any) -> None:
        if isinstance(item, dict):
            if REF_ID in item and item[REF_ID] not in referenced:
                del item[REF_ID]
            for value in item.values():
                strip(value)
        elif isinstance(item, list):
            for value in item:
                strip(value)

    collect(node)
    strip(node)
    return node


def contains_references(node: Any) -> bool:
    if isinstance(node, dict):
        return REF in node or any(contains_references(value) for value in node.values())
    if isinstance(node, list):
        return any(contains_references(value) for value in node)
    return False


def rebuild(obj: Any, keep: dict[int, Any]) -> Any:
    """A copy of `obj` built from its serialized form, keeping chosen children.

    `keep` maps ``id(child)`` to the object the copy should hold in its place:
    the child itself to share it with `obj`, or another object to swap it in.
    Everything else under `obj` is copied.
    """
    writer = _Writer()
    for child_id in keep:
        # Mark as seen, so every occurrence is written as a reference node.
        writer.ids[child_id] = (str(len(writer.ids)), None)
    token = _writer.set(writer)
    try:
        data = obj.as_dict(skip=['unique_name'])
    finally:
        _writer.reset(token)
    with reading() as reader:
        for child_id, replacement in keep.items():
            reader.objects[writer.ids[child_id][0]] = replacement
        return type(obj).from_dict(prune(data))
